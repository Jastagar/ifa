from __future__ import annotations

import re

# Words whose trailing period doesn't end a sentence ("Dr. Rao", "e.g. this").
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg", "ie",
    "no", "approx", "dept", "est", "fig", "inc", "ltd", "co", "jan", "feb", "mar", "apr",
    "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_BOUNDARY = re.compile(r"""([.!?…]+["')\]]*)\s+(?=["'(\[]?[A-Z0-9])""")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for speech. Plain Python, so it runs on every platform."""
    sentences: list[str] = []
    for paragraph in re.split(r"\n+", text):
        start = 0
        for match in _BOUNDARY.finditer(paragraph):
            end = match.end(1)
            before = paragraph[start:match.start(1)].rsplit(None, 1)
            last = before[-1] if before else ""
            word = last.lower().replace(".", "")
            if match.group(1) == "." and (word in ABBREVIATIONS or len(word) == 1 or "." in last):
                continue  # "Dr. Rao", "J. Smith", "U.S. Army"
            sentences.append(paragraph[start:end].strip())
            start = match.end()
        sentences.append(paragraph[start:].strip())
    return [s for s in sentences if s]
