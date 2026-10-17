import os
import pathlib
import tempfile
import time
import unittest

from ifa.services import prompts


class PromptsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "prompts.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_defaults_without_a_file(self):
        self.assertEqual(prompts.get("urgency", self.path), prompts.URGENCY)

    def test_saved_override_is_picked_up_without_restart(self):
        prompts.save({"urgency": "Only fire alarms are urgent."}, self.path)
        self.assertEqual(prompts.get("urgency", self.path), "Only fire alarms are urgent.")
        self.assertEqual(prompts.get("summary", self.path), prompts.SUMMARY)
        time.sleep(0.01)
        prompts.save({"urgency": "Nothing is urgent."}, self.path)
        os.utime(self.path, (time.time() + 5, time.time() + 5))
        self.assertEqual(prompts.get("urgency", self.path), "Nothing is urgent.")

    def test_validate_drops_defaults_and_blanks(self):
        values, errors = prompts.validate({"persona": prompts.PERSONA, "urgency": "  ", "summary": "Be brief."})
        self.assertEqual(values, {"summary": "Be brief."})
        self.assertEqual(errors, {})

    def test_validate_rejects_bad_input(self):
        _, errors = prompts.validate({"nope": "x", "summary": 3, "urgency": "x" * (prompts.MAX_CHARS + 1)})
        self.assertEqual(set(errors), {"nope", "summary", "urgency"})

    def test_broken_file_falls_back_to_defaults(self):
        self.path.write_text("{not json", encoding="utf-8")
        self.assertEqual(prompts.get("persona", self.path), prompts.PERSONA)


if __name__ == "__main__":
    unittest.main()
