"""
Tests the parsing of the simulation and model name files.

A name file may quote any token, and a quoted file name may hold spaces.

Cases:
  - test_models_block      : an unquoted, single-quoted, or double-quoted MODELS
                             line gives the model type, name, and name file.
  - test_packages_block    : the same for a PACKAGES line, which gives the
                             package names.
  - test_quoted_name_files : a quoted name file with a space and upper case in
                             its name is found through mfsim.nam.
"""

import io
import pathlib as pl
import sys

import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.utils.utils_modflow import (
    get_model_names_from_mfsim,
    get_package_names_from_gwfname,
    parse_models_block,
    parse_packages_block,
)

QUOTES = ("", "'", '"')


@pytest.mark.parametrize("q", QUOTES)
def test_models_block(q):
    f = io.StringIO(f"  GWF6 {q}model2.nam{q} {q}MODFLOW{q}\nEND MODELS\n")
    model_dict, namfile_dict = parse_models_block(f)
    assert model_dict == {"modflow": "gwf6"}
    assert namfile_dict == {"modflow": "model2.nam"}


@pytest.mark.parametrize("q", QUOTES)
def test_packages_block(q):
    f = io.StringIO(
        f"  DIS6 {q}model2.dis{q} {q}DIS{q}\n"
        + f"  WEL6 {q}model2.wel{q}\n"
        + "END PACKAGES\n"
    )
    assert parse_packages_block(f) == {"dis6": ["dis"], "wel6": ["wel-1"]}


@pytest.mark.parametrize("q", ("'", '"'))
def test_quoted_name_files(function_tmpdir, q):
    gwf_nam = "My Model.nam"
    (function_tmpdir / "mfsim.nam").write_text(
        f"BEGIN MODELS\n  GWF6 {q}{gwf_nam}{q} {q}MODFLOW{q}\nEND MODELS\n"
    )
    (function_tmpdir / gwf_nam).write_text(
        f"BEGIN PACKAGES\n  DIS6 {q}My Model.dis{q} {q}DIS{q}\nEND PACKAGES\n"
    )

    model_dict, namfile_dict = get_model_names_from_mfsim(function_tmpdir)
    assert model_dict == {"modflow": "gwf6"}
    assert namfile_dict == {"modflow": gwf_nam}

    package_dict = get_package_names_from_gwfname(
        function_tmpdir / namfile_dict["modflow"]
    )
    assert package_dict == {"dis6": ["dis"]}
