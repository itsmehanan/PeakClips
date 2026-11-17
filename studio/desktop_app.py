"""Launches PeakCut as a native Windows Desktop App (Automation + Manual Studio)."""
import os
import threading
import time
import urllib.request
import webview

os.environ.setdefault("PEAKCUT_ACCOUNTS", "0")
os.environ.setdefault("PEAKCUT_FETCH_MODEL", "0")

from studio import server  # noqa: E402


def _wait_ready(url: str, timeout: float = 20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/healthz", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.3)


if __name__ == "__main__":
    threading.Thread(target=server.main, daemon=True).start()
    base_url = "http://127.0.0.1:8765"
    _wait_ready(base_url)
    webview.create_window(
        "PeakCut AI — GTA VI Automation & Manual Clip Studio",
        url=base_url + "/app",
        width=1360,
        height=860,
        min_size=(1024, 700),
    )
    webview.start()