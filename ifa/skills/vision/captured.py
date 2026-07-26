from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(slots=True)
class Screenshot:
    image: Image.Image
    width: int
    height: int
    monitor: int

    def save(self, path: str | Path):
        self.image.save(path)