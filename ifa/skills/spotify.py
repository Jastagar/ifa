import os
import sys
import time
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


class SpotifySkill:
    def __init__(self, tts=None, auth: SpotifyOAuth | None = None, client=None, sleep=time.sleep) -> None:
        self.tts = tts
        self.auth = auth or make_auth()
        self.sp = client or spotipy.Spotify(auth_manager=self.auth, requests_timeout=10, retries=1)
        self._sleep = sleep

    def play(self, query: str | None = None) -> str:
        return self._run(self._play, query)

    def pause(self) -> str:
        return self._run(lambda: (self.sp.pause_playback(), "Paused.")[1])

    def next(self) -> str:
        return self._run(lambda: (self.sp.next_track(), "Skipped to the next track.")[1])

    def _run(self, action, *args) -> str:
        try:
            if not self.auth.validate_token(self.auth.cache_handler.get_cached_token()):
                raise SpotifyError(LOGIN_HINT)
            return action(*args)
        except Exception as exc:
            raise SpotifyError(explain(exc)) from exc

    def _play(self, query: str | None) -> str:
        if not query:
            self.sp.start_playback(device_id=self._device())
            return "Resumed playback."
        items = self.sp.search(q=query, limit=1, type="track")["tracks"]["items"]
        if not items:
            return f"Couldn't find {query} on Spotify."
        track = items[0]
        self.sp.start_playback(device_id=self._device(), uris=[track["uri"]])
        artists = ", ".join(a["name"] for a in track.get("artists", [])[:2])
        return f"Playing {track['name']}" + (f" by {artists}." if artists else ".")

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
