from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "docs" / "source"
BUILD_DIR = ROOT / "docs" / "_build" / "html"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute notebooks before adding them to docs.",
    )
    args = parser.parse_args()

    prepare_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sphinx_prepare_notebooks.py"),
    ]
    if args.execute:
        prepare_cmd.append("--execute")

    subprocess.run(prepare_cmd, check=True)
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
