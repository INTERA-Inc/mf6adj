from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
DOCS_NOTEBOOKS_DIR = ROOT / "docs" / "source" / "examples"


def _copy_or_execute_notebooks(
    execute: bool = False, skip_if_exist: bool = False
) -> list[Path]:
    DOCS_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    # If skip_if_exist, only process notebooks that don't exist in docs yet
    if skip_if_exist:
        existing = {f.name for f in DOCS_NOTEBOOKS_DIR.glob("*.ipynb")}
        notebooks = sorted(
            f for f in EXAMPLES_DIR.glob("*.ipynb") if f.name not in existing
        )
        if not notebooks:
            # All notebooks already exist, just list what's there
            notebooks = sorted(DOCS_NOTEBOOKS_DIR.glob("*.ipynb"))
    else:
        # Clean up existing notebooks
        for old_file in DOCS_NOTEBOOKS_DIR.glob("*.ipynb"):
            old_file.unlink()

        notebooks = sorted(EXAMPLES_DIR.glob("*.ipynb"))

    for notebook in notebooks:
        doc_notebook = DOCS_NOTEBOOKS_DIR / notebook.name

        # Skip if already exists and skip_if_exist is True
        if skip_if_exist and doc_notebook.exists():
            continue

        if execute:
            command = [
                sys.executable,
                "-m",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=3000",
                "--output-dir",
                str(DOCS_NOTEBOOKS_DIR),
                "--output",
                notebook.name,
                notebook.name,
            ]
            subprocess.run(command, check=True, cwd=notebook.parent)
        else:
            shutil.copy2(notebook, DOCS_NOTEBOOKS_DIR / notebook.name)

    return sorted(DOCS_NOTEBOOKS_DIR.glob("*.ipynb"))


def _write_index(notebooks: list[Path]) -> None:
    lines = [
        "Example Notebooks",
        "=================",
        "",
        (
            "The following notebooks from the `examples/` directory are "
            + "rendered in the documentation."
        ),
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
    ]

    for notebook in notebooks:
        lines.append(f"   {notebook.stem}")

    lines.append("")
    (DOCS_NOTEBOOKS_DIR / "index.rst").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute notebooks before copying them into docs source.",
    )
    parser.add_argument(
        "--skip-if-exist",
        action="store_true",
        help="Skip copying notebooks if they already exist in docs source. "
        + "Useful when artifacts have been pre-downloaded.",
    )
    args = parser.parse_args()

    notebooks = _copy_or_execute_notebooks(
        execute=args.execute, skip_if_exist=args.skip_if_exist
    )
    _write_index(notebooks)


if __name__ == "__main__":
    main()
