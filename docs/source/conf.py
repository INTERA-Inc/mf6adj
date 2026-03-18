from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_version() -> str:
    version_file = ROOT / "mf6adj" / "version.py"
    for line in version_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("__version__"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


project = "mf6adj"
author = "INTERA Incorporated"
copyright = f"{datetime.now():%Y}, {author}"
version = _read_version()
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "flopy",
    "h5py",
    "matplotlib",
    "modflowapi",
    "numpy",
    "pandas",
    "pyemu",
    "scipy",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
