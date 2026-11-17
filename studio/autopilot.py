"""Global Trend Hunter Autopilot: discovers new creators globally by niche & keywords,
rotates through different channels, and auto-clips/posts up to the daily reel quota."""
import re
import threading
import time
import yt_dlp

from . import creators, ctx, db, jobs, live

_STOP = threading.Event()
_SEEN_VIDEOS: set[str] = set()
_SEEN_CHANNELS: set[str] = set()  # Ensures rotation across different creators

_NON_LATIN = re.compile(r"[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]")


def _matches_language(entry: dict, target_lang: str) -> bool:
    lang = (target_lang or "en").strip().lower()
    if lang in ("", "any", "all"):
        return True
    meta_lang = str(entry.get("language") or "").strip().lower()
    title = str(entry.get("title") or "")
    if meta_lang:
        return meta_lang.startswith(lang)
    if lang.startswith("en") and _NON_LATIN.search(title):
        return False
    return True


def _build_global_queries(niche: str, keywords_csv: str, lang: str) -> list[str]:
    terms = [niche.strip()] if niche else []
    for kw in (keywords_csv or "").split(","):
        kw = kw.strip()
        if kw and kw.lower() not in [t.lower() for t in terms]:
            terms.append(kw)
    if not terms:
        terms = ["GTA 6"]

    lang_hint = " English" if (lang or "en").lower().startswith("en") else ""
    queries = []
    for t in terms:
        # Search for live streams and recent high-velocity videos across YouTube globally
        queries.append(f"ytsearch10:{t} live stream{lang_hint}")
        queries.append(f"ytsearch10:trending {t}{lang_hint}")
        queries.append(f"ytsearch10:new {t} gameplay{lang_hint}")
    return queries


def _today_clips_count(cid: str) -> int:
    cutoff = time.time() - 86400
    total = 0
    for j in jobs.list_jobs():
        if j.get("creator") != cid:
            continue
        if j.get("created", 0) >= cutoff:
            total += len(j.get("clips") or [])
    return total


def _hunt_global_trending(niche: str, keywords: str, lang: str) -> list[dict]:
    candidates = {}
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": 10,
    }
    queries = _build_global_queries(niche, keywords, lang)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for q in queries:
            try:
                info = ydl.extract_info(q, download=False)
                for e in (info or {}).get("entries", []):
                    vid = e.get("id")
                    channel_id = e.get("channel_id") or e.get("uploader_id") or e.get("uploader")
                    if not vid or vid in _SEEN_VIDEOS:
                        continue
                    # Avoid repeating the same creator back-to-back if possible
                    if channel_id and channel_id in _SEEN_CHANNELS and len(_SEEN_CHANNELS) < 50:
                        continue
                    if not _matches_language(e, lang):
                        continue

                    is_live = bool(e.get("live_status") == "is_live" or e.get("is_live"))
                    views = int(e.get("concurrent_view_count") or e.get("view_count") or 0)
                    dur = int(e.get("duration") or 0)

                    if is_live or (views >= 5000 and 120 <= dur <= 3600):
                        candidates[vid] = {
                            "id": vid,
                            "url": f"https://www.youtube.com/watch?v={vid}",
                            "title": e.get("title") or vid,
                            "channel": channel_id or "unknown",
                            "is_live": is_live,
                            "views": views,
                            "heat": (views * 5 if is_live else views),
                        }
            except Exception:
                continue

    ranked = list(candidates.values())
    ranked.sort(key=lambda x: -x["heat"])
    return ranked


def _run_cycle():
    for uid in db.all_user_ids():
        with ctx.as_user(uid):
            for c in creators.all_creators():
                if c.get("mode", "auto") != "auto" or not c.get("auto_watch"):
                    continue
                cid = c["id"]
                daily_cap = int(c.get("daily_cap") or 10)
                if _today_clips_count(cid) >= daily_cap:
                    continue
                if live.active_for(cid) or jobs.busy_count() >= 3:
                    continue

                niche = (c.get("niche") or "GTA 6").strip()
                keywords = (c.get("keywords") or "").strip()
                lang = (c.get("language") or "en").strip()

                ranked = _hunt_global_trending(niche, keywords, lang)
                for item in ranked:
                    _SEEN_VIDEOS.add(item["id"])
                    if item["channel"] != "unknown":
                        _SEEN_CHANNELS.add(item["channel"])
                    try:
                        if item["is_live"]:
                            live.start_session(c, item["url"])
                        else:
                            opts = {
                                "layout": c.get("layout") or "square",
                                "style": c.get("style") or "bounce",
                                "music": True,
                                "max_clips": min(3, int(c.get("max_clips") or 5)),
                                "min_len": int(c.get("min_len") or 15),
                                "max_len": int(c.get("max_len") or 60),
                            }
                            jobs.create(
                                {"type": "url", "url": item["url"], "name": item["title"]},
                                opts,
                                creator=cid,
                            )
                        creators.log(
                            cid,
                            f"Global Hunter [{lang.upper()}] found new creator video ({item['views']:,} views): {item['title'][:55]}",
                        )
                        break
                    except Exception as e:
                        creators.log(cid, f"Global Hunter skip: {e}")


def _loop():
    time.sleep(10)
    while not _STOP.is_set():
        try:
            _run_cycle()
        except Exception as e:
            print(f"[Autopilot] {e}", flush=True)
        _STOP.wait(400)  # Scan every ~6.5 minutes


def start():
    threading.Thread(target=_loop, daemon=True, name="peakcut-autopilot").start()