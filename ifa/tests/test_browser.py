import unittest
from unittest.mock import patch

from ifa.skills import browser
from ifa.tools.browser import TOOL

SEARCH_PAGE = (
    '{"adSlotRenderer":{"videoId":"AAAAAAAAAAA"}}'
    '{"videoRenderer":{"videoId":"Umqb9KENgmk","title":{"runs":[{"text":"Tum Hi Ho"}]}}}'
    '{"videoRenderer":{"videoId":"BjL7AuPsmEk"}}'
)


class BrowserSkillTests(unittest.TestCase):
    def test_urls_encode_the_query(self):
        self.assertEqual(browser.youtube_search_url("tum hi ho & more"),
                         "https://www.youtube.com/results?search_query=tum+hi+ho+%26+more")
        self.assertEqual(browser.google_search_url("weather today"), "https://www.google.com/search?q=weather+today")

    def test_normalize_url(self):
        self.assertEqual(browser.normalize_url("www.youtube.com"), "https://www.youtube.com")
        self.assertEqual(browser.normalize_url("http://localhost:8080"), "http://localhost:8080")

    def test_first_video_skips_ads(self):
        url = browser.first_youtube_video("tum hi ho", fetch=lambda _: SEARCH_PAGE)
        self.assertEqual(url, "https://www.youtube.com/watch?v=Umqb9KENgmk")

    def test_first_video_returns_none_when_lookup_fails(self):
        def broken(_):
            raise OSError("offline")
        self.assertIsNone(browser.first_youtube_video("x", fetch=broken))
        self.assertIsNone(browser.first_youtube_video("x", fetch=lambda _: "<html></html>"))

    def test_open_url_passes_the_url_to_brave(self):
        calls = []
        where = browser.open_url("https://youtube.com", find=lambda: "brave.exe", popen=lambda cmd, **kw: calls.append(cmd))
        self.assertEqual(calls, [["brave.exe", "https://youtube.com"]])
        self.assertEqual(where, "Brave")
        browser.open_url("https://x.com", private=True, find=lambda: "brave.exe", popen=lambda cmd, **kw: calls.append(cmd))
        self.assertEqual(calls[-1], ["brave.exe", "--incognito", "https://x.com"])

    def test_open_url_falls_back_to_the_default_browser(self):
        opened = []
        where = browser.open_url("https://x.com", find=lambda: None, fallback=opened.append)
        self.assertEqual(opened, ["https://x.com"])
        self.assertEqual(where, "your default browser")


class OpenWebsiteToolTests(unittest.TestCase):
    def run_tool(self, args, video="https://www.youtube.com/watch?v=Umqb9KENgmk"):
        with patch.object(browser, "open_url", return_value="Brave") as open_url, \
             patch.object(browser, "first_youtube_video", return_value=video):
            result = TOOL.handler(args, None)
        return result, open_url.call_args

    def test_play_on_youtube_opens_the_top_video(self):
        result, call = self.run_tool({"site": "youtube", "query": "Tum Hi Ho by Arijit Singh"})
        self.assertEqual(call.args[0], "https://www.youtube.com/watch?v=Umqb9KENgmk")
        self.assertIn("playing", result)

    def test_youtube_falls_back_to_results(self):
        result, call = self.run_tool({"site": "youtube", "query": "tum hi ho"}, video=None)
        self.assertEqual(call.args[0], "https://www.youtube.com/results?search_query=tum+hi+ho")

    def test_youtube_search_without_play(self):
        _, call = self.run_tool({"site": "youtube", "query": "lofi", "play": False})
        self.assertEqual(call.args[0], "https://www.youtube.com/results?search_query=lofi")

    def test_google_and_url_and_private(self):
        _, call = self.run_tool({"site": "google", "query": "ifa"})
        self.assertEqual(call.args[0], "https://www.google.com/search?q=ifa")
        _, call = self.run_tool({"site": "url", "url": "github.com", "private": True})
        self.assertEqual(call.args[0], "https://github.com")
        self.assertTrue(call.kwargs["private"])

    def test_missing_inputs_are_errors(self):
        self.assertTrue(self.run_tool({"site": "google"})[0].startswith("ERROR"))
        self.assertTrue(self.run_tool({"site": "url"})[0].startswith("ERROR"))


if __name__ == "__main__":
    unittest.main()
