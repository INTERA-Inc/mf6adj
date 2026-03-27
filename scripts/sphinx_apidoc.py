from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "source"
API_DIR = SOURCE_DIR / "_api"
PACKAGE_DIR = ROOT / "mf6adj"


def main() -> None:
    API_DIR.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "sphinx.ext.apidoc",
        "-f",
        "-e",
        "-M",
        "-o",
        str(API_DIR),
        str(PACKAGE_DIR),
        str(PACKAGE_DIR / "build"),
        str(PACKAGE_DIR / "__pycache__"),
        str(PACKAGE_DIR / "version.py"),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
