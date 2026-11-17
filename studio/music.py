"""Fetches and caches royalty-free / no-copyright background beats directly from YouTube."""
import random
import subprocess
from pathlib import Path
from . import config

QUERIES = [
    "no copyright synthwave gaming background music instrumental",
    "royalty free GTA Vice City synthwave beat no copyright",
    "no copyright phonk gaming instrumental stream safe",
]


def _bgm_dir() -> Path:
    d = config.DATA_DIR / "bgm"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_tracks(target_count: int = 3):
    d = _bgm_dir()
    existing = list(d.glob("*.m4a")) + list(d.glob("*.mp3"))
    if len(existing) >= target_count:
        return existing
    q = random.choice(QUERIES)
    out_tpl = str(d / "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        f"ytsearch3:{q}",
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--match-filter", "duration >= 60 & duration <= 300",
        "--no-playlist",
        "--quiet",
        "-o", out_tpl,
    ]
    subprocess.run(cmd, timeout=90, check=False)
    return list(d.glob("*.m4a")) + list(d.glob("*.mp3"))


def pick_track() -> Path | None:
    d = _bgm_dir()
    tracks = list(d.glob("*.m4a")) + list(d.glob("*.mp3"))
    if not tracks:
        tracks = ensure_tracks(2)
    return random.choice(tracks) if tracks else None
