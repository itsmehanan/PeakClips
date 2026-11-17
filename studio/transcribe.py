"""Speech to text with word timestamps: uses Groq Whisper API (fast, zero local download) when GROQ_API_KEY is set,
or falls back to Parakeet 0.6b through sherpa-onnx."""
import io
import os
import queue
import threading
import time
import wave

import httpx
import numpy as np

from . import config

_lock = threading.Lock()

GROQ_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.environ.get("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo").strip()

_CPUS = os.cpu_count() or 4
PROVIDER = os.environ.get("PEAKCUT_ASR_PROVIDER", "cpu").strip().lower() or "cpu"
THREADS = int(os.environ.get("PEAKCUT_ASR_THREADS") or (4 if _CPUS >= 8 else max(2, _CPUS - 2)))
COPIES = int(os.environ.get("PEAKCUT_ASR_COPIES") or (1 if PROVIDER != "cpu" else max(1, _CPUS // THREADS) if _CPUS >= 8 else 1))
_free: "queue.LifoQueue" = queue.LifoQueue()
_made = 0

REMOTE = os.environ.get("PEAKCUT_ASR_URL", "").strip().rstrip("/")
REMOTE_TOKEN = os.environ.get("PEAKCUT_ASR_TOKEN", "").strip()
REMOTE_BATCH = int(os.environ.get("PEAKCUT_ASR_REMOTE_BATCH", "8"))
REMOTE_PARALLEL = int(os.environ.get("PEAKCUT_ASR_REMOTE_PARALLEL", "4"))
REMOTE_SLOTS = threading.BoundedSemaphore(max(1, int(os.environ.get("PEAKCUT_ASR_REMOTE_INFLIGHT", "6"))))
_remote_down_until = 0.0


def _groq_key() -> str:
    return (os.environ.get("GROQ_API_KEY") or config.env("GROQ_API_KEY") or GROQ_KEY).strip()


def model_ready() -> bool:
    if _groq_key():
        return True
    from . import fetch_model
    return fetch_model.ready()


def _pcm_to_wav_bytes(pcm_int16: np.ndarray, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_int16.astype("<i2").tobytes())
    return buf.getvalue()


def _transcribe_groq_chunk(pcm_int16: np.ndarray, offset_sec: float, key: str) -> list:
    wav_data = _pcm_to_wav_bytes(pcm_int16, 16000)
    files = {"file": ("chunk.wav", wav_data, "audio/wav")}
    data = {
        "model": GROQ_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities[]": "word",
    }
    headers = {"Authorization": f"Bearer {key}"}
    for attempt in range(3):
        try:
            r = httpx.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
                timeout=60.0,
            )
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            out = []
            for item in payload.get("words") or []:
                w = str(item.get("word") or "").strip()
                if not w:
                    continue
                s = round(offset_sec + float(item.get("start", 0.0)), 2)
                e = round(offset_sec + float(item.get("end", s + 0.15)), 2)
                out.append({"w": w, "s": s, "e": max(s + 0.08, e)})
            # Fallback if a segment returned text without word-level array
            if not out and payload.get("text"):
                parts = str(payload["text"]).split()
                dur = len(pcm_int16) / 16000.0
                step = dur / max(1, len(parts))
                for i, w in enumerate(parts):
                    s = round(offset_sec + i * step, 2)
                    out.append({"w": w, "s": s, "e": round(s + step, 2)})
            return out
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    return []


def _transcribe_groq(audio: np.ndarray, key: str, on_progress=None) -> list:
    """Splits audio into ~5-minute WAV chunks (well under Groq's 25 MB limit) and extracts word timestamps."""
    sr = 16000
    chunk_samples = 300 * sr  # 5 minutes = ~9.6 MB WAV
    total = len(audio)
    if total == 0:
        return []
    slices = [(pos, min(total, pos + chunk_samples)) for pos in range(0, total, chunk_samples)]
    words = []
    for idx, (a, b) in enumerate(slices):
        seg = audio[a:b]
        # Skip near-silent chunks
        rms = float(np.sqrt(((seg.astype(np.float32) / 32768.0) ** 2).mean()))
        if len(seg) >= 4800 and rms >= 0.002:
            words.extend(_transcribe_groq_chunk(seg, a / sr, key))
        if on_progress:
            on_progress((idx + 1) / len(slices))
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.4
        w["e"] = round(max(w["s"] + 0.08, min(nxt, max(w["e"], w["s"] + 0.6))), 2)
    return words


def _model_files(d) -> dict:
    kinds = ("fp16", "", "int8") if PROVIDER != "cpu" else ("int8", "", "fp16")
    for k in kinds:
        names = {p: d / (f"{p}.{k}.onnx" if k else f"{p}.onnx") for p in ("encoder", "decoder", "joiner")}
        if all(f.is_file() for f in names.values()):
            return {p: str(f) for p, f in names.items()}
    raise FileNotFoundError(f"No speech model files in {d}")


def _new():
    import sherpa_onnx
    from . import fetch_model
    fetch_model.ensure()
    d = config.ASR_MODEL_DIR
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        **_model_files(d),
        tokens=str(d / "tokens.txt"),
        num_threads=THREADS,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        model_type="nemo_transducer",
        provider=PROVIDER,
    )


def _take():
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if make:
        try:
            return _new()
        except Exception:
            with _lock:
                _made -= 1
            raise
    return _free.get()


def _take_now():
    global _made
    try:
        return _free.get_nowait()
    except queue.Empty:
        pass
    with _lock:
        make = _made < COPIES
        if make:
            _made += 1
    if not make:
        return None
    try:
        return _new()
    except Exception:
        with _lock:
            _made -= 1
        raise


def _recognizer():
    rec = _take()
    _free.put(rec)
    return rec


def _chunks(audio: np.ndarray, lo=12.0, hi=24.0):
    sr, hop = 16000, 800
    n = len(audio)
    frames = n // hop
    if frames == 0:
        return
    x = audio[: frames * hop].reshape(frames, hop).astype(np.float32)
    rms = np.sqrt((x ** 2).mean(axis=1))
    pos = 0
    while pos < n:
        if n - pos <= hi * sr:
            yield pos, n
            return
        a, b = (pos + int(lo * sr)) // hop, (pos + int(hi * sr)) // hop
        cut = (a + int(np.argmin(rms[a:b]))) * hop + hop // 2
        yield pos, cut
        pos = cut


BATCH = int(os.environ.get("PEAKCUT_ASR_BATCH", "1"))


def _words(tokens, timestamps, a: int, b: int) -> list:
    out = []
    off, seg_end = a / 16000.0, b / 16000.0
    cur, gap = None, False
    for tok, t in zip(tokens, timestamps):
        new_word = tok.startswith((" ", "▁")) or gap
        piece = tok.replace("▁", " ").strip()
        if not piece:
            gap = True
            continue
        gap = False
        is_punct = all(not c.isalnum() for c in piece)
        if cur is None or (new_word and not is_punct):
            if cur:
                out.append(cur)
            cur = {"w": piece, "s": round(off + t, 2), "e": round(off + t + 0.12, 2)}
        else:
            cur["w"] += piece
            cur["e"] = round(off + t + 0.12, 2)
    if cur:
        cur["e"] = min(cur["e"], round(seg_end, 2))
        out.append(cur)
    return out


def decode_pieces(pieces: list, rec=None) -> list:
    streams = []
    rec = rec or _take()
    try:
        for seg in pieces:
            st = rec.create_stream()
            st.accept_waveform(16000, seg)
            streams.append(st)
        rec.decode_streams(streams)
    finally:
        _free.put(rec)
    return [(list(st.result.tokens), [float(t) for t in st.result.timestamps]) for st in streams]


def _remote_ok() -> bool:
    return bool(REMOTE) and time.time() >= _remote_down_until


def _decode_remote(pieces: list) -> list:
    raw = [np.clip(seg * 32768.0, -32768, 32767).astype("<i2") for seg in pieces]
    r = httpx.post(REMOTE + "/decode", content=b"".join(x.tobytes() for x in raw), timeout=httpx.Timeout(120, connect=4),
                   headers={"Authorization": "Bearer " + REMOTE_TOKEN, "X-Lens": ",".join(str(len(x)) for x in raw),
                            "Content-Type": "application/octet-stream"})
    r.raise_for_status()
    got = r.json()["pieces"]
    if len(got) != len(pieces):
        raise ValueError("wrong number of pieces came back")
    return [(p["tokens"], p["timestamps"]) for p in got]


def transcribe(audio: np.ndarray, on_progress=None) -> list:
    """Returns [{'w': word, 's': start_sec, 'e': end_sec}, ...] for the whole audio."""
    gkey = _groq_key()
    if gkey:
        return _transcribe_groq(audio, gkey, on_progress=on_progress)

    global _remote_down_until
    words = []
    todo = []
    for a, b in _chunks(audio):
        seg = audio[a:b].astype(np.float32) / 32768.0
        if len(seg) < 4800 or float(np.sqrt((seg ** 2).mean())) < 0.003:
            continue
        todo.append((a, b, seg))
    remote = _remote_ok()
    size = max(1, REMOTE_BATCH if remote else BATCH)
    groups = [todo[i:i + size] for i in range(0, len(todo), size)]
    done = [0]

    def decode(group):
        global _remote_down_until
        segs = [seg for _, _, seg in group]
        got = None
        while got is None:
            if remote and _remote_ok() and REMOTE_SLOTS.acquire(blocking=False):
                try:
                    got = _decode_remote(segs)
                except Exception:
                    _remote_down_until = time.time() + 60
                finally:
                    REMOTE_SLOTS.release()
                continue
            rec = _take_now()
            if rec is None and not (remote and _remote_ok()):
                rec = _take()
            if rec is not None:
                got = []
                step = max(1, BATCH)
                try:
                    for i in range(0, len(segs), step):
                        got += decode_pieces(segs[i:i + step], rec)
                        rec = None if i + step >= len(segs) else _take()
                except Exception:
                    got = None
                    raise
                continue
            time.sleep(0.02)
        out = [_words(tok, ts, a, b) for (a, b, _), (tok, ts) in zip(group, got)]
        done[0] += 1
        if on_progress:
            on_progress(done[0] / len(groups))
        return out

    workers = REMOTE_PARALLEL if remote else COPIES
    if workers > 1 and len(groups) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(workers, len(groups)), thread_name_prefix="asr") as ex:
            results = list(ex.map(decode, groups))
    else:
        results = [decode(g) for g in groups]
    for out in results:
        for piece in out:
            for w in piece:
                if words and words[-1]["w"] == w["w"] and words[-1]["s"] == w["s"]:
                    continue
                words.append(w)
    for i, w in enumerate(words):
        nxt = words[i + 1]["s"] if i + 1 < len(words) else w["e"] + 0.4
        w["e"] = round(max(w["s"] + 0.08, min(nxt, max(w["e"], w["s"] + 0.6))), 2)
    return words