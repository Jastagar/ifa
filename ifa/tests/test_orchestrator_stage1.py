"""Startup tests for the orchestrator: fail fast on Ollama / n8n config errors.

Run: python -m unittest ifa.tests.test_orchestrator_stage1 -v
"""
import unittest
from unittest.mock import patch

from ifa.tools import registry


class StartupIntegrationTests(unittest.TestCase):
    """Verify orchestrator startup wires Ollama check, n8n config, DB, tools."""

    def setUp(self):
        registry.clear()

    def tearDown(self):
        registry.clear()

    def test_health_check_failure_exits(self):
        from ifa.core import orchestrator

        with patch("ifa.core.orchestrator.check_health",
                   side_effect=RuntimeError("Ollama not running")):
            with self.assertRaises(SystemExit) as cm:
                orchestrator.run()
        self.assertEqual(cm.exception.code, 1)

    def test_n8n_config_parse_error_exits(self):
        from ifa.core import orchestrator
        from ifa.tools.n8n import N8nConfigError

        with patch("ifa.core.orchestrator.check_health"), \
             patch("ifa.core.orchestrator.load_n8n_config",
                   side_effect=N8nConfigError("YAML syntax error on line 3")):
            with self.assertRaises(SystemExit) as cm:
                orchestrator.run()
        self.assertEqual(cm.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
