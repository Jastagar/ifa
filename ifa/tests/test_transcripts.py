import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ifa.services.transcripts import TranscriptStore


class TranscriptStoreTests(unittest.TestCase):
    def test_review_then_freeze_into_train_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TranscriptStore(str(Path(tmp) / "ifa.db"), Path(tmp) / "stt_data", clock=lambda: 1000.0)
            ids = [store.record(np.full(8000, 0.1, np.float32), f"heard {i}", "phone") for i in range(10)]
            self.assertEqual(store.review(ids[0], "Spotify pe Humdard song chalao")["status"], "fixed")
            self.assertEqual(store.review(ids[1], "heard 1")["status"], "ok")
            for i in ids[2:9]:
                store.review(i, "heard")
            store.delete(ids[9])
            self.assertEqual(store.list("all")["counts"], {"new": 0, "ok": 1, "fixed": 8})

            out = Path(store.export()["path"])
            rows = {split: list(csv.DictReader(open(out / split / "metadata.csv", encoding="utf-8")))
                    for split in ("train", "test")}
            self.assertEqual((len(rows["train"]), len(rows["test"])), (8, 1))
            self.assertEqual(rows["train"][0]["transcription"], "Spotify pe Humdard song chalao")
            self.assertTrue(all((out / s / r["file_name"]).exists() for s in rows for r in rows[s]))
