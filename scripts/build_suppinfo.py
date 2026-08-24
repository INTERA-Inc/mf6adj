#!/usr/bin/env python
"""Build the supplemental technical information document for the docs build.

Runs the LaTeX build in docs/SuppInfo and copies the result in beside the
rendered notebooks, so it is picked up by the artifact the documentation
workflow uploads and the rtds_action extension downloads on Read the Docs.

A missing LaTeX distribution is reported and skipped, so the documentation can
be built without one. A LaTeX distribution that cannot build the document is a
failure: skipping that case hides a broken document behind a passing build.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "SuppInfo"
# the directory the rtds_action extension downloads and extracts into,
# so the document reaches Read the Docs with the rendered notebooks
DEST = ROOT / "docs" / "source" / "examples"
PDF = "mf6adjsuppinfo.pdf"


def main() -> int:
    if shutil.which("pdflatex") is None:
        print("[suppinfo] pdflatex not found, skipping the supplemental document")
        return 0

    print(f"[suppinfo] building {PDF}", flush=True)
    result = subprocess.run(["make"], cwd=SOURCE, capture_output=True, text=True)
    built = SOURCE / PDF
    if result.returncode != 0 or not built.is_file():
        print("[suppinfo] the build failed")
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        return 1

    DEST.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built, DEST / PDF)
    print(f"[suppinfo] copied {PDF} to {DEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
