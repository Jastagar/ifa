"""Web search: real web results through the free ddgs package (no API key), Wikipedia as a fallback.

ddgs scrapes several search engines, so it is slow (~4 s) and sometimes comes back empty.
Wikipedia's search API is fast and reliable but only covers encyclopedia topics.
"""
from __future__ import annotations

import html
import re
import time
from urllib.parse import urlparse

TIMEOUT_SEC = 6
MAX_RESULTS = 5
SNIPPET_CHARS = 300
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
TAG = re.compile(r"<[^>]+>")


class WebSearchError(Exception):
    pass


def log(message: str) -> None:
    print(f"[websearch] {message}", flush=True)


def _clean(text: str) -> str:
    return " ".join(html.unescape(TAG.sub("", text or "")).split())


def _site(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _from_ddgs(query: str, max_results: int) -> list[dict]:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException

    try:
        rows = DDGS(timeout=TIMEOUT_SEC).text(query, max_results=max_results)
    except DDGSException as exc:
        if "no results" in str(exc).lower():
            return []
        raise
    return [{"title": _clean(r.get("title")), "snippet": _clean(r.get("body")), "site": _site(r.get("href"))}
            for r in rows]


def _from_wikipedia(query: str, max_results: int) -> list[dict]:
    import requests

    response = requests.get(
        WIKIPEDIA_API,
        params={"action": "query", "list": "search", "srsearch": query, "srlimit": max_results, "format": "json"},
        headers={"User-Agent": "ifa-assistant/1.0"},
        timeout=TIMEOUT_SEC,
    )
    response.raise_for_status()
    rows = response.json().get("query", {}).get("search", [])
    return [{"title": _clean(r.get("title")), "snippet": _clean(r.get("snippet")), "site": "wikipedia.org"}
            for r in rows]


BACKENDS = (("web", _from_ddgs), ("wikipedia", _from_wikipedia))
FALLBACK_NOTE = {
    "wikipedia": "Live web search found nothing, so these are encyclopedia results: fine for background facts, "
                 "but often out of date for anything recent or 'latest'. Say so if the question is about something recent.",
}


def search(query: str, max_results: int = MAX_RESULTS, backends=BACKENDS) -> list[dict]:
    """Results as [{title, snippet, site}]. Tries each backend in turn until one finds something."""
    errors = []
    for name, backend in backends:
        started = time.monotonic()
        try:
            results = [r for r in backend(query, max_results) if r["title"] or r["snippet"]]
        except Exception as exc:
            log(f"{name} failed after {time.monotonic() - started:.1f}s: {type(exc).__name__}: {exc}")
            errors.append(f"{name}: {exc}")
            continue
        log(f"{name}: {len(results)} result(s) for {query!r} in {time.monotonic() - started:.1f}s")
        if results:
            note = FALLBACK_NOTE.get(name)
            return [{**r, "note": note} for r in results] if note else results
    if len(errors) == len(backends):
        raise WebSearchError("; ".join(errors))
    return []
