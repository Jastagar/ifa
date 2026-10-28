import unittest
from unittest.mock import patch

from ifa.skills import websearch
from ifa.tools import websearch as tool


def failing(query, n):
    raise OSError("offline")


class WebSearchTests(unittest.TestCase):
    def test_falls_back_when_the_first_backend_is_empty_or_fails(self):
        hit = [{"title": "Prime Minister of Japan", "snippet": "The head of government", "site": "wikipedia.org"}]
        for first in (lambda q, n: [], failing):
            self.assertEqual(websearch.search("pm of japan", backends=(("web", first), ("wiki", lambda q, n: hit))), hit)
        with self.assertRaises(websearch.WebSearchError):
            websearch.search("x", backends=(("web", failing), ("wiki", failing)))

    def test_tool_result_names_the_site_and_reports_failures(self):
        hit = [{"title": "IPL 2026 final", "snippet": "RCB won.", "site": "espncricinfo.com"}]
        with patch.object(websearch, "search", return_value=hit):
            self.assertIn("1. IPL 2026 final (espncricinfo.com): RCB won.", tool.TOOL.handler({"query": "ipl"}, None))
        with patch.object(websearch, "search", side_effect=websearch.WebSearchError("offline")):
            self.assertTrue(tool.TOOL.handler({"query": "ipl"}, None).startswith("ERROR"))
