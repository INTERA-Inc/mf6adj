from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "source"
BUILD_DIR = ROOT / "docs" / "_build" / "html"


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sphinx_apidoc.py")], check=True
    )
    command = [
        sys.executable,
        "-m",
        "sphinx",
        "-b",
        "html",
        str(SOURCE_DIR),
        str(BUILD_DIR),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
