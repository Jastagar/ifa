import unittest
from unittest.mock import MagicMock

import requests
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOauthError

from ifa.skills.spotify import DEVICE_WAIT_SEC, LOGIN_HINT, SpotifyError, SpotifySkill, explain

TRACK = {"name": "Tum Hi Ho", "uri": "spotify:track:1", "artists": [{"name": "Arijit Singh"}]}


def device(id, active=False, restricted=False):
    return {"id": id, "is_active": active, "is_restricted": restricted}


class SpotifySkillTests(unittest.TestCase):
    def setUp(self):
        self.auth = MagicMock()
        self.auth.validate_token.return_value = {"access_token": "x"}
        self.sp = MagicMock()
        self.sp.search.return_value = {"tracks": {"items": [TRACK]}}
        self.sp.devices.return_value = {"devices": [device("phone"), device("pc", active=True)]}
        self.sleeps = []
        self.skill = SpotifySkill(auth=self.auth, client=self.sp, sleep=self.sleeps.append)

    def test_plays_the_top_result_on_the_active_device(self):
        self.assertEqual(self.skill.play("tum hi ho"), "Playing Tum Hi Ho by Arijit Singh.")
        self.sp.start_playback.assert_called_once_with(device_id="pc", uris=["spotify:track:1"])

    def test_play_without_a_query_resumes(self):
        self.assertEqual(self.skill.play(None), "Resumed playback.")
        self.sp.start_playback.assert_called_once_with(device_id="pc")

    def test_no_search_results(self):
        self.sp.search.return_value = {"tracks": {"items": []}}
        self.assertIn("Couldn't find", self.skill.play("zzzz"))
        self.sp.start_playback.assert_not_called()

    def test_not_logged_in_never_starts_a_login_flow(self):
        self.auth.validate_token.return_value = None
        with self.assertRaises(SpotifyError) as cm:
            self.skill.play("x")
        self.assertEqual(str(cm.exception), LOGIN_HINT)
        self.sp.search.assert_not_called()

    def test_waits_for_a_device_that_is_still_starting(self):
        self.sp.devices.side_effect = [{"devices": []}, {"devices": []}, {"devices": [device("pc")]}]
        self.skill.play("x")
        self.assertEqual(len(self.sleeps), 2)
        self.sp.start_playback.assert_called_once_with(device_id="pc", uris=["spotify:track:1"])

    def test_gives_up_when_no_device_appears(self):
        self.sp.devices.return_value = {"devices": [device("tv", restricted=True)]}
        with self.assertRaises(SpotifyError) as cm:
            self.skill.play("x")
        self.assertIn("isn't open on any device", str(cm.exception))
        self.assertEqual(len(self.sleeps), DEVICE_WAIT_SEC)

    def test_api_errors_become_sentences(self):
        self.sp.pause_playback.side_effect = SpotifyException(403, -1, "Player command failed", reason="PREMIUM_REQUIRED")
        with self.assertRaises(SpotifyError) as cm:
            self.skill.pause()
        self.assertIn("Premium", str(cm.exception))


class ExplainTests(unittest.TestCase):
    def test_messages(self):
        cases = {
            SpotifyOauthError("error: invalid_client"): "rejected the app keys",
            SpotifyOauthError("error: invalid_grant"): "expired",
            SpotifyException(401, -1, "token expired"): "expired",
            SpotifyException(404, -1, "no device", reason="NO_ACTIVE_DEVICE"): "isn't open",
            SpotifyException(429, -1, "slow down"): "rate limiting",
            requests.ConnectionError("down"): "Couldn't reach Spotify",
        }
        for exc, expected in cases.items():
            self.assertIn(expected, explain(exc), exc)


if __name__ == "__main__":
    unittest.main()
