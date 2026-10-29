"""Every transcription Ifa makes, with its audio, so Sir can correct them and train speech-to-text on his voice.

Clips live in stt_data/clips (git-ignored: it is his voice). Each row starts as "new"; reviewing it in the
launcher's Transcripts tab marks it "ok" (heard right) or "fixed" (text corrected). Export writes the reviewed
clips as a Hugging Face audiofolder dataset: train/ and test/, each with WAVs and metadata.csv.
"""
from __future__ import annotations

import csv
import os
import shutil
import sqlite3
import threading
import time
import uuid
import wave
from contextlib import closing
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "stt_data"
SAMPLE_RATE = 16_000
TEST_EVERY = 5  # every 5th reviewed clip goes to the held-out test split
STATUSES = ("new", "ok", "fixed")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    source TEXT NOT NULL,
    audio TEXT NOT NULL,
    duration REAL NOT NULL,
    heard TEXT NOT NULL,
    text TEXT,
    status TEXT NOT NULL DEFAULT 'new'
)
"""


def collecting() -> bool:
    return os.environ.get("IFA_STT_COLLECT", "1").strip().lower() not in {"0", "false", "no", "off"}


class TranscriptStore:
    def __init__(self, db_path: str, data_dir: Path = DATA_DIR, clock=time.time) -> None:
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.clips_dir = self.data_dir / "clips"
        self._clock = clock
        self._lock = threading.Lock()
        with closing(self._connect()) as conn, conn:
            conn.execute(SCHEMA)
            conn.execute("CREATE INDEX IF NOT EXISTS transcripts_status ON transcripts (status, id)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def record(self, audio: np.ndarray, heard: str, source: str) -> int | None:
        heard = " ".join((heard or "").split())
        if not heard or audio is None or not len(audio):
            return None
        now = self._clock()
        name = f"{source}_{int(now * 1000)}_{uuid.uuid4().hex[:6]}.wav"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(self.clips_dir / name), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        with self._lock, closing(self._connect()) as conn, conn:
            cur = conn.execute(
                "INSERT INTO transcripts (created_at, source, audio, duration, heard) VALUES (?, ?, ?, ?, ?)",
                (now, source, name, len(pcm) / SAMPLE_RATE, heard),
            )
            return cur.lastrowid

    def list(self, status: str = "new", limit: int = 100, offset: int = 0) -> dict:
        where = {"new": "status = 'new'", "reviewed": "status != 'new'", "all": "1"}.get(status)
        if where is None:
            raise ValueError(f"status must be new, reviewed or all, not {status!r}")
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT id, created_at, source, duration, heard, text, status FROM transcripts WHERE {where} "
                "ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset),
            ).fetchall()
            counts = dict(conn.execute("SELECT status, COUNT(*) FROM transcripts GROUP BY status").fetchall())
        return {"items": [dict(r) for r in rows], "counts": {s: counts.get(s, 0) for s in STATUSES}}

    def review(self, item_id: int, text: str) -> dict:
        text = " ".join((text or "").split())
        if not text:
            raise ValueError("the text can't be empty; delete the clip instead")
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT heard FROM transcripts WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise ValueError(f"no transcript {item_id}")
            status = "ok" if text == row["heard"] else "fixed"
            conn.execute("UPDATE transcripts SET text = ?, status = ? WHERE id = ?", (text, status, item_id))
        return {"id": item_id, "text": text, "status": status}

    def delete(self, item_id: int) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            row = conn.execute("SELECT audio FROM transcripts WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                raise ValueError(f"no transcript {item_id}")
            conn.execute("DELETE FROM transcripts WHERE id = ?", (item_id,))
        (self.clips_dir / row["audio"]).unlink(missing_ok=True)

    def audio_path(self, item_id: int) -> Path:
        with self._lock, closing(self._connect()) as conn:
            row = conn.execute("SELECT audio FROM transcripts WHERE id = ?", (item_id,)).fetchone()
        if row is None:
            raise ValueError(f"no transcript {item_id}")
        return self.clips_dir / Path(row["audio"]).name

    def export(self) -> dict:
        """Freeze every reviewed clip into stt_data/datasets/<time>/ for training."""
        with self._lock, closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT id, audio, text FROM transcripts WHERE status != 'new' ORDER BY id").fetchall()
        if not rows:
            raise ValueError("nothing reviewed yet: fix or tick some transcripts first")
        out = self.data_dir / "datasets" / time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        splits: dict[str, list[tuple[str, str]]] = {"train": [], "test": []}
        for i, row in enumerate(rows, 1):
            src = self.clips_dir / row["audio"]
            if not src.exists():
                continue
            split = "test" if i % TEST_EVERY == 0 else "train"
            (out / split).mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / split / src.name)
            splits[split].append((src.name, row["text"]))
        for split, items in splits.items():
            if items:
                with open(out / split / "metadata.csv", "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["file_name", "transcription"])
                    writer.writerows(items)
        (out / "README.txt").write_text(
            f"Ifa speech-to-text dataset, frozen {time.strftime('%Y-%m-%d %H:%M')}.\n"
            f"train: {len(splits['train'])} clips, test: {len(splits['test'])} clips (16 kHz mono WAV).\n\n"
            "Load it with:\n"
            "    from datasets import load_dataset\n"
            f"    ds = load_dataset('audiofolder', data_dir=r'{out}')\n",
            encoding="utf-8",
        )
        return {"path": str(out), "train": len(splits["train"]), "test": len(splits["test"])}


_store: TranscriptStore | None = None
_store_lock = threading.Lock()


def record(audio: np.ndarray, heard: str, source: str) -> None:
    """Save one transcription for review. Never lets a storage problem break the turn."""
    global _store
    if not collecting():
        return
    try:
        with _store_lock:
            if _store is None:
                from ifa.services.db import DB_PATH
                _store = TranscriptStore(DB_PATH)
        _store.record(audio, heard, source)
    except Exception as exc:
        print(f"[stt-data] couldn't save the clip for review: {exc}")
