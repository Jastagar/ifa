"""Tests for ifa.tools.n8n (call_n8n_workflow tool + load_n8n_config).

Run: python -m unittest ifa.tests.test_tools_n8n -v
"""
import os
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import httpx

from ifa.core.context import AgentContext
from ifa.tools import registry


def _ctx(n8n_config: dict) -> AgentContext:
    return AgentContext(tts=MagicMock(), db_path=":memory:", n8n_config=n8n_config)


def _mock_response(status_code: int = 200, text: str = "ok") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    return resp


# ==================== load_n8n_config ====================

class LoadConfigTests(unittest.TestCase):
    def test_missing_file_returns_empty(self):
        from ifa.tools.n8n import load_n8n_config
        self.assertEqual(load_n8n_config("/nonexistent/path.yaml"), {})

    def test_valid_minimal_config(self):
        from ifa.tools.n8n import load_n8n_config
        yaml_content = """
workflows:
  ping:
    url: https://example.com/webhook/abc
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config = load_n8n_config(path)
            self.assertIn("ping", config)
            self.assertEqual(config["ping"]["url"], "https://example.com/webhook/abc")
        finally:
            os.unlink(path)

    def test_full_config_with_auth_and_schema(self):
        from ifa.tools.n8n import load_n8n_config
        yaml_content = """
workflows:
  full:
    url: https://n8n.local/webhook/xyz
    auth:
      type: header
      name: X-API-Key
      env: MY_KEY
    timeout: 45
    payload_schema:
      type: object
      additionalProperties: false
      properties:
        message: {type: string}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name
        try:
            config = load_n8n_config(path)
            self.assertEqual(config["full"]["auth"]["env"], "MY_KEY")
            self.assertEqual(config["full"]["timeout"], 45)
            self.assertIn("payload_schema", config["full"])
        finally:
            os.unlink(path)

    def test_yaml_syntax_error_raises(self):
        from ifa.tools.n8n import load_n8n_config, N8nConfigError
        # Tab indentation inside a block is a real YAML syntax error
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("workflows:\n\tbad: {]\n")  # tab + mismatched braces
            path = f.name
        try:
            with self.assertRaises(N8nConfigError) as cm:
                load_n8n_config(path)
            self.assertIn("YAML syntax error", str(cm.exception))
        finally:
            os.unlink(path)

    def test_missing_workflows_key_raises(self):
        from ifa.tools.n8n import load_n8n_config, N8nConfigError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("not_workflows: {}\n")
            path = f.name
        try:
            with self.assertRaises(N8nConfigError):
                load_n8n_config(path)
        finally:
            os.unlink(path)

    def test_workflow_missing_url_raises(self):
        from ifa.tools.n8n import load_n8n_config, N8nConfigError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("workflows:\n  broken:\n    timeout: 30\n")
            path = f.name
        try:
            with self.assertRaises(N8nConfigError) as cm:
                load_n8n_config(path)
            self.assertIn("url", str(cm.exception).lower())
        finally:
            os.unlink(path)

    def test_empty_file_returns_empty(self):
        from ifa.tools.n8n import load_n8n_config, N8nConfigError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # empty file — yaml.safe_load returns None
            path = f.name
        try:
            # Empty file yields None → our handler treats missing workflows as error
            with self.assertRaises(N8nConfigError):
                load_n8n_config(path)
        finally:
            os.unlink(path)


# ==================== call_n8n_workflow tool ====================

class CallWorkflowToolTests(unittest.TestCase):
    def setUp(self):
        registry.clear()
        from ifa.tools.n8n import TOOL
        registry.register(TOOL)

    def tearDown(self):
        registry.clear()

    def test_registered(self):
        self.assertIsNotNone(registry.get("call_n8n_workflow"))

    def test_happy_path_posts_and_returns_body(self):
        ctx = _ctx({"ping": {"url": "https://n8n.local/webhook/abc"}})
        with patch("ifa.tools.n8n.httpx.post", return_value=_mock_response(200, "success")) as post:
            result = registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "ping", "payload": {}},
                ctx,
            )
        self.assertIn("success", result)
        self.assertIn("200", result)
        self.assertEqual(post.call_args.args[0], "https://n8n.local/webhook/abc")
        self.assertEqual(post.call_args.kwargs["json"], {})

    def test_unknown_workflow_returns_error(self):
        ctx = _ctx({"ping": {"url": "https://example.com"}})
        result = registry.dispatch(
            "call_n8n_workflow",
            {"workflow_name": "nonexistent", "payload": {}},
            ctx,
        )
        self.assertIn("No workflow named", result)
        self.assertIn("ping", result)  # lists what IS available

    def test_auth_header_resolved_from_env(self):
        ctx = _ctx({
            "secured": {
                "url": "https://example.com/webhook",
                "auth": {"type": "header", "name": "X-API-Key", "env": "TEST_IFA_KEY"},
            }
        })
        with patch.dict(os.environ, {"TEST_IFA_KEY": "s3cret"}), \
             patch("ifa.tools.n8n.httpx.post", return_value=_mock_response(200)) as post:
            registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "secured", "payload": {}},
                ctx,
            )
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["X-API-Key"], "s3cret")

    def test_missing_env_var_returns_error(self):
        ctx = _ctx({
            "secured": {
                "url": "https://example.com",
                "auth": {"type": "header", "name": "X-API-Key", "env": "NONEXISTENT_VAR_XYZ"},
            }
        })
        # Ensure the env var isn't accidentally set
        env = {k: v for k, v in os.environ.items() if k != "NONEXISTENT_VAR_XYZ"}
        with patch.dict(os.environ, env, clear=True):
            result = registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "secured", "payload": {}},
                ctx,
            )
        self.assertIn("Auth misconfigured", result)
        self.assertIn("not set", result)

    def test_timeout_returns_error(self):
        ctx = _ctx({"slow": {"url": "https://example.com", "timeout": 5}})
        with patch("ifa.tools.n8n.httpx.post", side_effect=httpx.TimeoutException("slow")):
            result = registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "slow", "payload": {}},
                ctx,
            )
        self.assertIn("timed out", result.lower())

    def test_connect_error_returns_error(self):
        ctx = _ctx({"down": {"url": "https://unreachable.example.com"}})
        with patch("ifa.tools.n8n.httpx.post", side_effect=httpx.ConnectError("no route")):
            result = registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "down", "payload": {}},
                ctx,
            )
        self.assertIn("unreachable", result.lower())

    def test_large_response_truncated(self):
        ctx = _ctx({"big": {"url": "https://example.com"}})
        huge_body = "x" * 5000
        with patch("ifa.tools.n8n.httpx.post", return_value=_mock_response(200, huge_body)):
            result = registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "big", "payload": {}},
                ctx,
            )
        self.assertIn("truncated", result)
        # Result is ~2KB of x's plus status line + truncation note, not 5KB
        self.assertLess(len(result), 3000)

    def test_payload_schema_default_is_restrictive(self):
        """Workflow with no payload_schema → empty-props + additionalProperties:false
        → LLM can only send empty payload."""
        ctx = _ctx({"strict": {"url": "https://example.com"}})
        # Sending extra fields should be rejected by the default schema
        result = registry.dispatch(
            "call_n8n_workflow",
            {"workflow_name": "strict", "payload": {"sneaky": "field"}},
            ctx,
        )
        self.assertIn("rejected by schema", result)

    def test_payload_schema_enforced(self):
        ctx = _ctx({
            "typed": {
                "url": "https://example.com",
                "payload_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            }
        })
        # Missing required field
        result = registry.dispatch(
            "call_n8n_workflow",
            {"workflow_name": "typed", "payload": {}},
            ctx,
        )
        self.assertIn("rejected", result)

    def test_payload_schema_passes_valid(self):
        ctx = _ctx({
            "typed": {
                "url": "https://example.com",
                "payload_schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"message": {"type": "string"}},
                },
            }
        })
        with patch("ifa.tools.n8n.httpx.post", return_value=_mock_response(200, "ok")):
            result = registry.dispatch(
                "call_n8n_workflow",
                {"workflow_name": "typed", "payload": {"message": "hi"}},
                ctx,
            )
        self.assertIn("ok", result)
        self.assertNotIn("rejected", result)

    def test_tool_level_schema_rejects_missing_fields(self):
        ctx = _ctx({})
        result = registry.dispatch(
            "call_n8n_workflow",
            {"workflow_name": "x"},  # missing `payload`
            ctx,
        )
        self.assertTrue(result.startswith("ERROR"))


# ==================== gitignore + example file ====================

class ConfigFilePresenceTests(unittest.TestCase):
    def test_example_template_is_committed(self):
        example_path = pathlib.Path(__file__).parent.parent / "config" / "n8n_workflows.yaml.example"
        self.assertTrue(example_path.exists(), "n8n_workflows.yaml.example must be committed")

    def test_real_config_is_gitignored(self):
        """git check-ignore exits 0 if the path is ignored."""
        repo_root = pathlib.Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["git", "check-ignore", "ifa/config/n8n_workflows.yaml"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        # exit code 0 = ignored
        self.assertEqual(
            result.returncode, 0,
            f"ifa/config/n8n_workflows.yaml must be in .gitignore. Got: {result.stdout} {result.stderr}",
        )


if __name__ == "__main__":
    unittest.main()
