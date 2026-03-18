from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = ROOT / "examples"
DOCS_NOTEBOOKS_DIR = ROOT / "docs" / "source" / "examples"


def _copy_or_execute_notebooks(execute: bool = False) -> list[Path]:
    DOCS_NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in DOCS_NOTEBOOKS_DIR.glob("*.ipynb"):
        old_file.unlink()

    notebooks = sorted(EXAMPLES_DIR.glob("*.ipynb"))
    for notebook in notebooks:
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

    return notebooks


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
    args = parser.parse_args()

    notebooks = _copy_or_execute_notebooks(execute=args.execute)
    _write_index(notebooks)


if __name__ == "__main__":
    main()
