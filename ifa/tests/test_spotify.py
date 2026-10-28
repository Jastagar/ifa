import unittest
from unittest.mock import MagicMock

import requests
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyOauthError

from ifa.skills.spotify import DEVICE_WAIT_SEC, LOGIN_HINT, SpotifyError, SpotifySkill, explain

TRACK = {"name": "Tum Hi Ho", "uri": "spotify:track:1", "artists": [{"name": "Arijit Singh"}]}


def device(id, active=False, restricted=False):
    return {"id": id, "is_active": active, "is_restricted": restricted}


def report_what_was_started(sp):
    def current():
        explicit = sp.current_playback.return_value
        if explicit is None or isinstance(explicit, dict):
            return explicit
        uris = (sp.start_playback.call_args.kwargs.get("uris") if sp.start_playback.call_args else None) or ["x"]
        return {"is_playing": True, "item": {"uri": uris[0], "name": "x"}, "device": {"name": "PC"}}
    sp.current_playback.side_effect = current


class SpotifySkillTests(unittest.TestCase):
    def setUp(self):
        self.auth = MagicMock()
        self.auth.validate_token.return_value = {"access_token": "x"}
        self.sp = MagicMock()
        self.sp.search.return_value = {"tracks": {"items": [TRACK]}}
        self.sp.devices.return_value = {"devices": [device("phone"), device("pc", active=True)]}
        self.sleeps = []
        self.skill = SpotifySkill(auth=self.auth, client=self.sp, sleep=self.sleeps.append)
        report_what_was_started(self.sp)

    def test_plays_the_top_result_on_the_active_device(self):
        self.assertEqual(self.skill.play("tum hi ho"), "Playing Tum Hi Ho by Arijit Singh on PC.")
        self.sp.start_playback.assert_called_once_with(device_id="pc", uris=["spotify:track:1"])

    def test_pause_says_what_it_paused(self):
        self.sp.current_playback.return_value = {"is_playing": True, "item": TRACK}
        self.assertEqual(self.skill.pause(), "Paused Tum Hi Ho by Arijit Singh.")
        self.sp.pause_playback.assert_called_once()

    def test_pause_with_nothing_playing(self):
        self.sp.current_playback.return_value = None
        self.assertEqual(self.skill.pause(), "Paused.")

    def test_now_playing(self):
        self.sp.current_playback.return_value = {"is_playing": False, "item": TRACK}
        self.assertEqual(self.skill.now_playing(), "Paused on Spotify: Tum Hi Ho by Arijit Singh.")
        self.sp.current_playback.return_value = None
        self.assertEqual(self.skill.now_playing(), "Nothing is playing on Spotify.")

    def test_play_without_a_query_resumes(self):
        self.assertEqual(self.skill.play(None), "Resumed playback on PC.")
        self.sp.start_playback.assert_called_once_with(device_id="pc", uris=None)

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
        from unittest.mock import patch
        with patch.object(SpotifySkill, "_launch_app") as launch:
            self.skill.play("x")
        launch.assert_called_once()
        self.assertEqual(self.sleeps[:2], [1, 1])
        self.sp.start_playback.assert_called_once_with(device_id="pc", uris=["spotify:track:1"])

    def test_gives_up_when_no_device_appears(self):
        self.sp.devices.return_value = {"devices": [device("tv", restricted=True)]}
        with self.assertRaises(SpotifyError) as cm:
            self.skill.play("x")
        self.assertIn("isn't open on any device", str(cm.exception))
        self.assertEqual(len(self.sleeps), DEVICE_WAIT_SEC)

    def test_accepted_but_not_playing_says_so_without_retry_storms(self):
        self.sp.current_playback.return_value = {"is_playing": False, "item": None, "device": {"name": "PC"}}
        with self.assertRaises(SpotifyError) as cm:
            self.skill.play("tum hi ho")
        self.assertIn("nothing is playing", str(cm.exception))
        self.sp.transfer_playback.assert_not_called()
        self.assertEqual(self.sp.start_playback.call_count, 1)

    def test_on_this_pc_it_starts_the_app_and_plays_again(self):
        from unittest.mock import patch
        import socket
        host = socket.gethostname().split(".")[0]
        pc = {"id": "me", "name": host, "type": "Computer", "is_active": True}
        self.sp.devices.return_value = {"devices": [pc]}
        states = iter([{"is_playing": False}] * 5 + [{"is_playing": True, "item": TRACK, "device": {"name": host}}] * 20)
        self.sp.current_playback.side_effect = lambda: next(states)
        with patch.object(SpotifySkill, "_launch_app") as launch:
            self.assertEqual(self.skill.play("tum hi ho"), f"Playing Tum Hi Ho by Arijit Singh on {host}.")
        launch.assert_called_once()
        self.assertEqual(self.sp.start_playback.call_count, 2)
        self.sp.transfer_playback.assert_not_called()

    def test_prefers_this_computer_when_nothing_is_active(self):
        self.sp.devices.return_value = {"devices": [{"id": "ph", "type": "Smartphone", "is_active": False},
                                                    {"id": "pc", "type": "Computer", "is_active": False}]}
        self.skill.play("tum hi ho")
        self.assertEqual(self.sp.start_playback.call_args.kwargs["device_id"], "pc")
        self.sp.transfer_playback.assert_not_called()

    def test_api_errors_become_sentences(self):
        self.sp.pause_playback.side_effect = SpotifyException(403, -1, "Player command failed", reason="PREMIUM_REQUIRED")
        with self.assertRaises(SpotifyError) as cm:
            self.skill.pause()
        self.assertIn("Premium", str(cm.exception))


def track(name, artist, uri, popularity=50, album="Album"):
    return {"name": name, "uri": uri, "popularity": popularity,
            "artists": [{"name": artist}], "album": {"name": album, "release_date": "2013-04-01"}}


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.auth = MagicMock()
        self.auth.validate_token.return_value = {"access_token": "x"}
        self.sp = MagicMock()
        self.sp.devices.return_value = {"devices": [device("pc", active=True)]}
        self.skill = SpotifySkill(auth=self.auth, client=self.sp, sleep=lambda s: None)
        report_what_was_started(self.sp)

    def results(self, *tracks):
        self.sp.search.return_value = {"tracks": {"items": list(tracks)}}

    def test_original_beats_popular_slowed_and_lofi_versions(self):
        self.results(
            track("Tum Hi Ho (Slowed + Reverb)", "Lofi Guy", "u:slowed", popularity=70),
            track("Tum Hi Ho - Lofi Flip", "Someone", "u:lofi", popularity=65),
            track("Tum Hi Ho", "Arijit Singh", "u:original", popularity=60, album="Aashiqui 2"),
        )
        text = self.skill.search("tum hi ho")
        self.assertTrue(text.splitlines()[1].startswith("1. Tum Hi Ho by Arijit Singh (album Aashiqui 2, 2013)"))
        self.assertIn("[slowed version]", text)
        self.assertEqual(self.skill.candidates[0]["uri"], "u:original")

    def test_asking_for_a_variant_does_not_penalise_it(self):
        self.results(track("Tum Hi Ho", "Arijit Singh", "u:original"),
                     track("Tum Hi Ho (Slowed + Reverb)", "Lofi Guy", "u:slowed"))
        self.skill.search("tum hi ho slowed reverb")
        self.assertEqual(self.skill.candidates[0]["uri"], "u:slowed")

    def test_artist_is_a_filter_with_a_plain_fallback(self):
        self.sp.search.side_effect = [{"tracks": {"items": []}},
                                      {"tracks": {"items": [track("Lover", "Diljit Dosanjh", "u:lover")]}}]
        self.skill.search("lover", artist="diljit")
        queries = [c.kwargs["q"] for c in self.sp.search.call_args_list]
        self.assertEqual(queries, ["track:lover artist:diljit", "lover diljit"])
        self.assertEqual(self.sp.search.call_args.kwargs["market"], "from_token")

    def test_unplayable_results_are_skipped(self):
        self.results({**track("Alpha", "Avi Gill", "u:blocked"), "is_playable": False}, track("Alpha", "Ravi Basrur", "u:ok"))
        self.skill.search("alpha")
        self.assertEqual([t["uri"] for t in self.skill.candidates], ["u:ok"])

    def test_duplicates_are_listed_once(self):
        self.results(track("Kesariya", "Arijit Singh", "u:1"), track("Kesariya", "Arijit Singh", "u:2"),
                     track("Kesariya", "Pritam", "u:3"))
        self.skill.search("kesariya")
        self.assertEqual(len(self.skill.candidates), 2)

    def test_play_by_choice_and_bad_choice(self):
        self.results(track("Brown Munde", "AP Dhillon", "u:1", 80), track("Brown Munde (Remix)", "DJ", "u:2", 40))
        self.skill.search("brown munde")
        self.assertEqual(self.skill.play(choice=2), "Playing Brown Munde (Remix) by DJ on PC.")
        self.sp.start_playback.assert_called_once_with(device_id="pc", uris=["u:2"])
        self.assertIn("no result number 9", self.skill.play(choice=9))

    def test_no_results(self):
        self.results()
        self.assertIn("No songs found", self.skill.search("zzqx"))
        self.assertIn("There is no result number 1", self.skill.play(choice=1))


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
