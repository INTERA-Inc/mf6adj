#!/usr/bin/env python
"""Build the supplemental technical information document for the docs build.

Runs the LaTeX build in docs/SuppInfo and copies the result into the Sphinx
static directory, where docs/source/supplemental.rst offers it for download.

The document is supplementary, so a missing LaTeX distribution is reported and
skipped rather than failing the documentation build.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "SuppInfo"
STATIC = ROOT / "docs" / "source" / "_static"
PDF = "mf6adjsuppinfo.pdf"


def main() -> int:
    if shutil.which("pdflatex") is None:
        print("[suppinfo] pdflatex not found, skipping the supplemental document")
        return 0

    print(f"[suppinfo] building {PDF}", flush=True)
    result = subprocess.run(["make"], cwd=SOURCE, capture_output=True, text=True)
    built = SOURCE / PDF
    if result.returncode != 0 or not built.is_file():
        print("[suppinfo] the build failed, skipping the supplemental document")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return 0

    STATIC.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, STATIC / PDF)
    print(f"[suppinfo] copied {PDF} to {STATIC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
