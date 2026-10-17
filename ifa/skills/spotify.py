import os
import re
import sys
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import requests
import spotipy
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import CacheFileHandler, SpotifyOAuth, SpotifyOauthError

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN_CACHE = REPO_ROOT / ".spotify_token"
SCOPE = "user-modify-playback-state user-read-playback-state"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"
DEVICE_WAIT_SEC = 8
SEARCH_LIMIT = 10
MAX_CANDIDATES = 6
VARIANTS = ("slowed", "reverb", "sped up", "lofi", "lo-fi", "lo fi", "remix", "cover", "karaoke",
            "instrumental", "8d", "mashup", "unplugged", "reprise", "acoustic")
LOGIN_HINT = (
    "Spotify isn't logged in yet. Run 'Log in to Spotify' in the launcher's Setup list, "
    "or python -m ifa.skills.spotify"
)


class SpotifyError(Exception):
    pass


def make_auth(open_browser: bool = False) -> SpotifyOAuth:
    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise SpotifyError("Spotify keys are not set. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET.")
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", "").strip() or DEFAULT_REDIRECT_URI,
        scope=SCOPE,
        cache_handler=CacheFileHandler(cache_path=str(TOKEN_CACHE)),
        open_browser=open_browser,
    )


def explain(exc: Exception) -> str:
    if isinstance(exc, SpotifyError):
        return str(exc)
    if isinstance(exc, SpotifyOauthError):
        text = str(exc)
        if "invalid_client" in text:
            return "Spotify rejected the app keys. Check SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
        if "invalid_grant" in text:
            return "The Spotify login has expired. " + LOGIN_HINT
        return f"Spotify login failed: {text}"
    if isinstance(exc, SpotifyException):
        reason = getattr(exc, "reason", None) or ""
        if exc.http_status == 401:
            return "The Spotify login has expired. " + LOGIN_HINT
        if reason == "PREMIUM_REQUIRED" or exc.http_status == 403:
            return "Spotify only allows playback control on Premium accounts."
        if reason == "NO_ACTIVE_DEVICE" or exc.http_status == 404:
            return "Spotify isn't open on any device. Open the Spotify app and try again."
        if exc.http_status == 429:
            return "Spotify is rate limiting requests. Try again in a minute."
        return f"Spotify returned an error ({exc.http_status}): {exc.msg}"
    if isinstance(exc, requests.RequestException):
        return "Couldn't reach Spotify. Check the internet connection."
    return f"Spotify failed: {exc}"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def variant_of(track: dict) -> str | None:
    name = _normalize(track["name"] + " " + track.get("album", {}).get("name", ""))
    return next((v for v in VARIANTS if re.search(rf"\b{re.escape(v)}\b", name)), None)


def match_score(track: dict, query: str, artist: str | None, rank: int) -> float:
    title = _normalize(re.split(r"\s[(\[-]", track["name"])[0])
    wanted = _normalize(query)
    score = SequenceMatcher(None, title, wanted).ratio()
    if title and (title in wanted or wanted in title):
        score += 0.3
    artists = _normalize(" ".join(a["name"] for a in track.get("artists", [])))
    if artist:
        score += 0.4 * SequenceMatcher(None, artists, _normalize(artist)).ratio()
    elif any(w in artists for w in wanted.split() if len(w) > 3):
        score += 0.15
    variant = variant_of(track)
    if variant:
        score += 0.4 if variant in wanted else -0.6
    score += 0.3 * track.get("popularity", 0) / 100
    return score - 0.02 * rank


def song_name(track: dict) -> str:
    artists = ", ".join(a["name"] for a in track.get("artists", [])[:2])
    return f"{track['name']} by {artists}" if artists else track["name"]


def describe(track: dict) -> str:
    artists = ", ".join(a["name"] for a in track.get("artists", [])[:2])
    album = track.get("album", {})
    year = (album.get("release_date") or "")[:4]
    extra = ", ".join(x for x in (album.get("name") and f"album {album['name']}", year) if x)
    variant = variant_of(track)
    return f"{track['name']} by {artists}" + (f" ({extra})" if extra else "") + (f" [{variant} version]" if variant else "")


class SpotifySkill:
    def __init__(self, tts=None, auth: SpotifyOAuth | None = None, client=None, sleep=time.sleep) -> None:
        self.tts = tts
        self.auth = auth or make_auth()
        self.sp = client or spotipy.Spotify(auth_manager=self.auth, requests_timeout=10, retries=1)
        self._sleep = sleep
        self.candidates: list[dict] = []

    def search(self, query: str, artist: str | None = None) -> str:
        return self._run(self._search_text, query, artist)

    def play(self, query: str | None = None, choice: int | None = None, artist: str | None = None) -> str:
        return self._run(self._play, query, choice, artist)

    def pause(self) -> str:
        return self._run(self._pause)

    def now_playing(self) -> str:
        return self._run(self._now_playing)

    def next(self) -> str:
        return self._run(lambda: (self.sp.next_track(), "Skipped to the next track.")[1])

    def _run(self, action, *args) -> str:
        try:
            if not self.auth.validate_token(self.auth.cache_handler.get_cached_token()):
                raise SpotifyError(LOGIN_HINT)
            return action(*args)
        except Exception as exc:
            raise SpotifyError(explain(exc)) from exc

    def _search(self, query: str, artist: str | None) -> list[dict]:
        query = " ".join(query.split())
        queries = [f"track:{query} artist:{artist}", f"{query} {artist}"] if artist else [query]
        items: list[dict] = []
        for q in queries:
            items = self.sp.search(q=q, limit=SEARCH_LIMIT, type="track", market="from_token")["tracks"]["items"]
            if items:
                break
        seen, unique = set(), []
        for rank, track in enumerate(t for t in items if t):
            key = (_normalize(track["name"]), _normalize(track["artists"][0]["name"]) if track.get("artists") else "")
            if key not in seen:
                seen.add(key)
                unique.append((match_score(track, query, artist, rank), track))
        unique.sort(key=lambda pair: pair[0], reverse=True)
        self.candidates = [track for _, track in unique[:MAX_CANDIDATES]]
        return self.candidates

    def _search_text(self, query: str, artist: str | None) -> str:
        if not self._search(query, artist):
            return f"No songs found on Spotify for {query}."
        lines = [f"{i}. {describe(t)}" for i, t in enumerate(self.candidates, 1)]
        return ("Spotify results, best guess first:\n" + "\n".join(lines) +
                "\nNow call spotify_control with action play and choice set to the number that matches the song "
                "Sir asked for. Prefer the original over remix, slowed, lofi or cover versions unless he asked for one.")

    def _play(self, query: str | None, choice: int | None, artist: str | None) -> str:
        if choice is not None:
            if not 1 <= choice <= len(self.candidates):
                return f"There is no result number {choice}. Search again first."
            track = self.candidates[choice - 1]
        elif query:
            if not self._search(query, artist):
                return f"Couldn't find {query} on Spotify."
            track = self.candidates[0]
        else:
            self.sp.start_playback(device_id=self._device())
            return "Resumed playback."
        self.sp.start_playback(device_id=self._device(), uris=[track["uri"]])
        artists = ", ".join(a["name"] for a in track.get("artists", [])[:2])
        return f"Playing {track['name']}" + (f" by {artists}." if artists else ".")

    def _current_track(self) -> dict | None:
        playback = self.sp.current_playback() or {}
        item = playback.get("item")
        return item if isinstance(item, dict) and item.get("name") else None

    def _pause(self) -> str:
        track = self._current_track()
        self.sp.pause_playback()
        return f"Paused {song_name(track)}." if track else "Paused."

    def _now_playing(self) -> str:
        playback = self.sp.current_playback() or {}
        track = self._current_track()
        if track is None:
            return "Nothing is playing on Spotify."
        state = "Playing" if playback.get("is_playing") else "Paused"
        return f"{state} on Spotify: {song_name(track)}."

    def _device(self) -> str:
        for attempt in range(DEVICE_WAIT_SEC + 1):
            devices = [d for d in self.sp.devices()["devices"] if not d.get("is_restricted")]
            if devices:
                return next((d for d in devices if d.get("is_active")), devices[0])["id"]
            if attempt < DEVICE_WAIT_SEC:
                self._sleep(1)
        raise SpotifyError("Spotify isn't open on any device. Open the Spotify app and try again.")


def login() -> int:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
    try:
        auth = make_auth(open_browser=True)
        print(f"Opening Spotify in your browser. After you approve, it redirects to {auth.redirect_uri}.", flush=True)
        auth.get_access_token(as_dict=False)
        user = spotipy.Spotify(auth_manager=auth, requests_timeout=10).current_user()
    except Exception as exc:
        print(explain(exc), file=sys.stderr)
        return 1
    print(f"Logged in to Spotify as {user.get('display_name') or user.get('id')}. Token saved to {TOKEN_CACHE.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(login())
