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
        "-T",
        "-o",
        str(API_DIR),
        str(PACKAGE_DIR),
        str(PACKAGE_DIR / "build"),
        str(PACKAGE_DIR / "__pycache__"),
        str(PACKAGE_DIR / "version.py"),
    ]
    subprocess.run(command, check=True)
    _postprocess(API_DIR)


def _postprocess(api_dir: Path) -> None:
    """Fix up auto-generated RST files after sphinx-apidoc runs.

    - Updates the mf6adj package heading.
    - Removes the Submodules toctree so only the top-level public API is shown.
    """
    pkg_rst = api_dir / "mf6adj.rst"
    if not pkg_rst.exists():
        return

    text = pkg_rst.read_text()

    # Replace the auto-generated title with a friendlier one.
    text = text.replace(
        "mf6adj package\n==============\n",
        "mf6adj classes and helper functions\n====================================\n",
    )

    # Strip the Submodules section (everything from that heading onward).
    idx = text.find("\nSubmodules\n")
    if idx != -1:
        text = text[:idx] + "\n"

    pkg_rst.write_text(text)


if __name__ == "__main__":
    main()
