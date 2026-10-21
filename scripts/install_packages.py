"""Install every package Ifa needs, at the exact versions in ifa/constraints.txt.

Run with the venv's Python: venv/bin/python -m scripts.install_packages
(Windows: venv\\Scripts\\python -m scripts.install_packages). Safe to re-run: it
also puts back any package whose version drifted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUPPORTED = ((3, 10), (3, 11), (3, 12))
IMPORT_CHECK = """
import torch, transformers, sherpa_onnx, openwakeword, faster_whisper, perth
if perth.PerthImplicitWatermarker is None:
    from perth.perth_net.perth_net_implicit.perth_watermarker import PerthImplicitWatermarker
import chatterbox.tts_turbo
print(f"[install] imports OK: torch {torch.__version__}, transformers {transformers.__version__}, "
      f"CUDA {'yes' if torch.cuda.is_available() else 'no'}")
"""


def pip(*args: str) -> int:
    print(f"$ python -m pip {' '.join(args)}", flush=True)
    return subprocess.call([sys.executable, "-m", "pip", *args], cwd=ROOT)


def main() -> int:
    v = sys.version_info
    print(f"[install] Python {v.major}.{v.minor}.{v.micro} ({sys.executable})", flush=True)
    if (v.major, v.minor) not in SUPPORTED:
        print(f"[install] Ifa needs Python 3.10, 3.11 or 3.12, but this venv runs {v.major}.{v.minor}. "
              "Delete the venv folder and create it again with Python 3.12.", flush=True)
        return 1
    if pip("install", "--upgrade", "pip") != 0:
        print("[install] couldn't upgrade pip; carrying on with the one in the venv", flush=True)
    code = pip("install", "-r", "ifa/requirements.txt", "-c", "ifa/constraints.txt")
    if code != 0:
        print("[install] pip couldn't install the packages. The lines above starting with "
              "'The conflict is caused by' say exactly why.", flush=True)
        return code
    if pip("check") != 0:
        print("[install] installed, but pip check found the conflicts listed above.", flush=True)
        return 1
    print("[install] checking that the key libraries import...", flush=True)
    if subprocess.call([sys.executable, "-c", IMPORT_CHECK], cwd=ROOT) != 0:
        print("[install] a library is installed but won't import (see the error above).", flush=True)
        return 1
    print("[install] all packages installed at the locked versions and consistent.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
