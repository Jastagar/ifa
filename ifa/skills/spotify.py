# ifa/skills/spotify.py

from ifa.skills.base import Skill
import spotipy
from spotipy.oauth2 import SpotifyOAuth


class SpotifySkill(Skill):
    def __init__(self, tts):
        self.tts = tts

        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id="697e9ce952f04a3fb027fc65d675f859",
            client_secret="fe03d76048374d04993038139dd405f5",
            redirect_uri="http://127.0.0.1:8888/callback",
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