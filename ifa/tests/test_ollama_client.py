"""Tests for ifa.services.ollama_client.

Run: python -m unittest ifa.tests.test_ollama_client -v
"""
import unittest
from unittest.mock import MagicMock, patch

import httpx

from ifa.services.ollama_client import (
    build_tool_result_message,
    chat,
    check_health,
)


def _mock_response(status_code: int = 200, json_data: dict | None = None) -> MagicMock:
    """Fabricate an httpx.Response with json() + raise_for_status()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "boom", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class BuildToolResultMessageTests(unittest.TestCase):
    """Ollama's spec uses `tool_name`, not `name`. Lock this in."""

    def test_returns_expected_shape(self):
        msg = build_tool_result_message("get_time", "12:00pm")
        self.assertEqual(msg, {"role": "tool", "tool_name": "get_time", "content": "12:00pm"})

    def test_field_is_tool_name_not_name(self):
        msg = build_tool_result_message("get_time", "noon")
        self.assertIn("tool_name", msg)
        self.assertNotIn("name", msg)


class ChatTests(unittest.TestCase):
    def test_posts_to_api_chat(self):
        with patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(200, {"message": {"content": "hi"}})) as post:
            result = chat("qwen2.5:7b-instruct", [{"role": "user", "content": "hello"}])

        self.assertEqual(result, {"message": {"content": "hi"}})
        post.assert_called_once()
        url = post.call_args[0][0]
        self.assertTrue(url.endswith("/api/chat"))

    def test_includes_tools_when_provided(self):
        tools = [{"type": "function", "function": {"name": "x", "description": "y", "parameters": {}}}]
        with patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(200, {"message": {"content": "ok"}})) as post:
            chat("qwen2.5:7b-instruct", [{"role": "user", "content": "hi"}], tools=tools)

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["tools"], tools)

    def test_omits_tools_when_none(self):
        with patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(200, {"message": {"content": "ok"}})) as post:
            chat("qwen2.5:7b-instruct", [{"role": "user", "content": "hi"}])

        payload = post.call_args.kwargs["json"]
        self.assertNotIn("tools", payload)

    def test_sends_stream_false(self):
        with patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(200, {"message": {}})) as post:
            chat("qwen2.5:7b-instruct", [{"role": "user", "content": "hi"}])
        self.assertFalse(post.call_args.kwargs["json"]["stream"])

    def test_raises_on_http_error(self):
        with patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(500)):
            with self.assertRaises(httpx.HTTPStatusError):
                chat("qwen2.5:7b-instruct", [{"role": "user", "content": "hi"}])


class CheckHealthTests(unittest.TestCase):
    MODEL = "qwen2.5:7b-instruct"

    def _mock_tags(self, model_names: list[str]) -> MagicMock:
        return _mock_response(200, {"models": [{"name": n} for n in model_names]})

    def test_happy_path_passes_silently(self):
        with patch("ifa.services.ollama_client.httpx.get",
                   return_value=self._mock_tags(["qwen2.5:7b-instruct-q4_K_M"])), \
             patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(200, {"message": {"content": "ok"}})):
            check_health(self.MODEL)  # must not raise

    def test_ollama_not_running(self):
        with patch("ifa.services.ollama_client.httpx.get",
                   side_effect=httpx.ConnectError("connection refused")):
            with self.assertRaises(RuntimeError) as cm:
                check_health(self.MODEL)
        self.assertIn("not running", str(cm.exception).lower())
        self.assertIn("ollama serve", str(cm.exception))

    def test_model_not_pulled(self):
        with patch("ifa.services.ollama_client.httpx.get",
                   return_value=self._mock_tags(["llama3.1:8b"])):
            with self.assertRaises(RuntimeError) as cm:
                check_health(self.MODEL)
        self.assertIn("ollama pull qwen2.5:7b-instruct", str(cm.exception))

    def test_tool_calling_not_supported(self):
        """Ollama responds to /api/tags but rejects /api/chat with tools."""
        def post_fails_on_chat(url, **kwargs):
            if url.endswith("/api/chat"):
                return _mock_response(400, {"error": "tools not supported"})
            return _mock_response(200)

        with patch("ifa.services.ollama_client.httpx.get",
                   return_value=self._mock_tags(["qwen2.5:7b-instruct"])), \
             patch("ifa.services.ollama_client.httpx.post",
                   side_effect=post_fails_on_chat):
            with self.assertRaises(RuntimeError) as cm:
                check_health(self.MODEL)
        self.assertIn("tool-calling", str(cm.exception).lower())

    def test_model_prefix_match(self):
        """check_health accepts any tag starting with the required prefix."""
        with patch("ifa.services.ollama_client.httpx.get",
                   return_value=self._mock_tags(["qwen2.5:7b-instruct-q8_0"])), \
             patch("ifa.services.ollama_client.httpx.post",
                   return_value=_mock_response(200, {"message": {"content": "ok"}})):
            check_health(self.MODEL)  # must not raise

    def test_empty_model_list(self):
        with patch("ifa.services.ollama_client.httpx.get",
                   return_value=self._mock_tags([])):
            with self.assertRaises(RuntimeError) as cm:
                check_health(self.MODEL)
        self.assertIn("ollama pull", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
