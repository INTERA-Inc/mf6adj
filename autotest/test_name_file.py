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
  - test_quoted_solve      : a model whose name files quote every token solves
                             to the sensitivities of the same model unquoted.
"""

import io
import pathlib as pl
import shutil
import sys

import flopy
import h5py
import numpy as np
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

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

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


def _build_model(ws):
    """A steady 3 by 3 grid with a constant head and a well."""
    sim = flopy.mf6.MFSimulation(sim_name="sim", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim)
    flopy.mf6.ModflowIms(sim, outer_dvclose=1.0e-9, inner_dvclose=1.0e-10)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="model")
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=3, ncol=3, top=0.0, botm=-10.0)
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 0), 0.0]])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[(0, 2, 2), -5.0]])
    sim.write_simulation(silent=True)


def _quote_name_files(ws, gwf_nam):
    """Quote every token after the type in the MODELS and PACKAGES blocks."""
    (ws / "model.nam").rename(ws / gwf_nam)
    for path, block in ((ws / "mfsim.nam", "models"), (ws / gwf_nam, "packages")):
        lines = []
        in_block = False
        for line in path.read_text().splitlines():
            key = line.strip().lower()
            if key.startswith("begin") and block in key:
                in_block = True
            elif key.startswith("end") and block in key:
                in_block = False
            elif in_block and key != "":
                raw = line.split()
                if block == "models":
                    raw[1] = gwf_nam
                line = "  " + " ".join([raw[0]] + [f"'{t}'" for t in raw[1:]])
            lines.append(line)
        path.write_text("\n".join(lines) + "\n")


def _solve_adjoint(ws):
    """Return the sensitivities of the head at the center cell."""
    (ws / "name.adj").write_text(
        "begin performance_measure pm\n"
        + "  1 1 1 2 2 head direct 1.0 -1.0e+30\n"
        + "end performance_measure\n"
    )
    adj = mf6adj.Mf6Adj(
        "name.adj", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="forward.hd5")
    adj.solve_adjoint()
    adj.finalize()
    values = {}
    with h5py.File(ws / "adjoint_solution_pm.hd5", "r") as hf:
        for key in hf:
            if key.startswith("solution_"):
                for name in hf[key]:
                    values[f"{key}/{name}"] = hf[key][name][:]
    return values


def test_quoted_solve(function_tmpdir):
    base_ws = function_tmpdir / "base"
    quoted_ws = function_tmpdir / "quoted"
    _build_model(base_ws)
    shutil.copytree(base_ws, quoted_ws)
    _quote_name_files(quoted_ws, "My Model.nam")

    base = _solve_adjoint(base_ws)
    quoted = _solve_adjoint(quoted_ws)
    assert len(base) > 0
    assert quoted.keys() == base.keys()
    for key, value in base.items():
        assert np.allclose(quoted[key], value), key
