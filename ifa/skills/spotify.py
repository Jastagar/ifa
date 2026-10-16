import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth


class SpotifySkill:
    def __init__(self, tts):
        self.tts = tts

        client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
        client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()
        if not client_id or not client_secret:
            raise RuntimeError("Spotify keys are not set (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET)")

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=os.environ.get("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
            scope="user-modify-playback-state user-read-playback-state"
        ))

    def play(self, query: str):
        results = self.sp.search(q=query, limit=5, type="track")

        print("PLAYING IN SPOTIFY")

        if not results["tracks"]["items"]:
            return "Couldn't find that song"

        track = results["tracks"]["items"][0]
        uri = track["uri"]

        devices = self.sp.devices()
        if not devices["devices"]:
            return "No active Spotify device"

        device_id = devices["devices"][0]["id"]

        self.sp.start_playback(device_id=device_id, uris=[uri])
        return f"Playing {track['name']}"

    def pause(self):
        self.sp.pause_playback()
        return "Paused playback"

    def next(self):
        self.sp.next_track()
        return "Skipping track"