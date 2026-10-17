from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import webbrowser
from urllib.parse import quote_plus

YOUTUBE = "https://www.youtube.com"
YOUTUBE_RESULTS = YOUTUBE + "/results?search_query="
GOOGLE_SEARCH = "https://www.google.com/search?q="
ORGANIC_VIDEO = re.compile(r'"videoRenderer":\{"videoId":"([\w-]{11})"')
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def brave_candidates() -> list[str]:
    if sys.platform == "win32":
        roots = [os.environ.get(k) for k in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")]
        return [
            os.path.join(root, "BraveSoftware", "Brave-Browser", "Application", "brave.exe")
            for root in roots if root
        ]
    if sys.platform == "darwin":
        return ["/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"]
    return [p for p in (shutil.which("brave-browser"), shutil.which("brave")) if p]


def find_brave() -> str | None:
    return next((p for p in brave_candidates() if os.path.exists(p)), None)


def normalize_url(text: str) -> str:
    text = text.strip()
    if re.match(r"^[a-z][a-z0-9+.-]*://", text, re.IGNORECASE):
        return text
    return "https://" + text


def open_url(url: str, private: bool = False, find=find_brave, popen=subprocess.Popen, fallback=webbrowser.open) -> str:
    brave = find()
    if brave is None:
        if not url:
            raise RuntimeError("Brave isn't installed where Ifa looks for it.")
        fallback(url)
        return "your default browser"
    popen([brave, *(["--incognito"] if private else []), *([url] if url else [])],
          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return "a private Brave window" if private else "Brave"


def youtube_search_url(query: str) -> str:
    return YOUTUBE_RESULTS + quote_plus(query)


def google_search_url(query: str) -> str:
    return GOOGLE_SEARCH + quote_plus(query)


def _fetch(url: str) -> str:
    import httpx

    response = httpx.get(url, headers=BROWSER_HEADERS, cookies={"CONSENT": "YES+"}, timeout=6, follow_redirects=True)
    response.raise_for_status()
    return response.text


def first_youtube_video(query: str, fetch=_fetch) -> str | None:
    try:
        match = ORGANIC_VIDEO.search(fetch(youtube_search_url(query)))
    except Exception as exc:
        print(f"[browser] YouTube lookup failed: {exc}")
        return None
    return f"{YOUTUBE}/watch?v={match.group(1)}" if match else None
