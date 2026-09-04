"""
Tests for the sparsity that locates the entries of the assembled matrix.

The adjoint is taken from the matrix MODFLOW assembled, so the row pointer and
column index that locate its entries have to be the solution's. They match the
groundwater flow grid's until a package adds its own equations, which puts a
column inside the row of every cell that package connects to.

Cases:
  - test_assembled_matrix_reproduces_equations : the assembled matrix and the
                                                 solution vector satisfy the
                                                 equations MODFLOW solved, with
                                                 and without a package that adds
                                                 equations.
  - test_grid_connectivity_misplaces_entries   : the grid connectivity does not
                                                 locate those entries once maw6
                                                 adds its own.
  - test_forward_file_carries_solution_ia_ja   : the forward solution file
                                                 carries the solution sparsity,
                                                 and it matches the grid's for a
                                                 model the adjoint supports.
  - test_extra_equations_are_refused           : a solution carrying equations
                                                 beyond the flow grid's is
                                                 refused rather than solved
                                                 short of equations.
"""

import pathlib as pl
import shutil
import sys

import flopy
import h5py
import modflowapi
import numpy as np
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

from mf6adj.utils.utils_modflow import assemble_matrix

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NAME = "slnmat"
NLAY, NROW, NCOL = 3, 5, 5
BOTM = [-10.0, -100.0, -1000.0]
WELL_CELL = (1, 2, 2)


def _build_model(ws, maw):
    """A small unconfined model, optionally pumped through a multi-aquifer well."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="complex", outer_dvclose=1e-9)
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=NAME, newtonoptions="NEWTON", save_flows=True
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=10.0,
        delc=10.0,
        top=0.0,
        botm=BOTM,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1.0, k33=1.0)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1e-5, sy=0.1, transient={0: True})
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), 0.0, 100.0] for i in range(NROW)],
        pname="ghb-1",
    )
    if maw:
        # a well screened across every layer, so it takes a column in three
        # rows of the solution rather than in one
        flopy.mf6.ModflowGwfmaw(
            gwf,
            nmawwells=1,
            packagedata=[[0, 1.0, BOTM[-1], 0.0, "THIEM", NLAY]],
            connectiondata=[
                [0, k, (k, 2, 2), -999, -999, -999.0, -999.0] for k in range(NLAY)
            ],
            perioddata={0: [[0, "RATE", -100.0]]},
            pname="maw-1",
        )
    else:
        flopy.mf6.ModflowGwfwel(
            gwf, stress_period_data=[[WELL_CELL, -100.0]], pname="wel-1"
        )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    return ws


def _solution_arrays(ws):
    """Return the assembled matrix, its two sparsities, and the solved system.

    The values are taken after one time step, where the matrix MODFLOW last
    assembled and the heads it converged on satisfy the same equations.
    """
    mf6 = modflowapi.ModflowApi(lib_name, working_directory=str(ws))
    mf6.initialize()
    mf6.update()

    def value(name, *components):
        return np.asarray(mf6.get_value(mf6.get_var_address(name, *components)))

    arrays = {
        "amat": value("AMAT", "SLN_1"),
        "x": value("X", "SLN_1"),
        "rhs": value("RHS", "SLN_1"),
        "sln_ia": value("IA", "SLN_1") - 1,
        "sln_ja": value("JA", "SLN_1") - 1,
        "grid_ia": value("IA", NAME.upper(), "CON") - 1,
        "grid_ja": value("JA", NAME.upper(), "CON") - 1,
    }
    mf6.finalize()
    return arrays


def _residual(arrays, ia, ja):
    """Return the largest residual of the equations, against their own scale."""
    nrow = len(ia) - 1
    amat = assemble_matrix(arrays["amat"], ia, ja)
    residual = amat @ arrays["x"][:nrow] - arrays["rhs"][:nrow]
    return np.abs(residual).max() / np.abs(arrays["rhs"][:nrow]).max()


@pytest.mark.parametrize("maw", [False, True], ids=["wel", "maw"])
def test_assembled_matrix_reproduces_equations(function_tmpdir, maw):
    """The solution sparsity locates the entries of the matrix MODFLOW assembled."""
    arrays = _solution_arrays(_build_model(function_tmpdir / "m", maw))

    nnodes = len(arrays["grid_ia"]) - 1
    nsolution = len(arrays["sln_ia"]) - 1
    assert nsolution == nnodes + (1 if maw else 0), (
        "maw6 adds one equation per well to the solution"
    )

    assert _residual(arrays, arrays["sln_ia"], arrays["sln_ja"]) < 1.0e-4


def test_grid_connectivity_misplaces_entries(function_tmpdir):
    """The grid connectivity no longer locates those entries once maw6 adds its own."""
    arrays = _solution_arrays(_build_model(function_tmpdir / "m", maw=True))

    # the well's column is inside the row of each cell it connects to, so the
    # grid connectivity, which does not count it, shifts every entry after the
    # first of those rows
    assert _residual(arrays, arrays["grid_ia"], arrays["grid_ja"]) > 1.0


def test_forward_file_carries_solution_ia_ja(function_tmpdir):
    """A supported model's two sparsities agree, and the forward file carries both."""
    ws = _build_model(function_tmpdir / "m", maw=False)
    k, i, j = WELL_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.finalize()

    with h5py.File(ws / "fwd.hd5", "r") as hf:
        info = hf["gwf_info"]
        assert "sln_ia" in info and "sln_ja" in info
        assert np.array_equal(info["sln_ia"][:], info["ia"][:])
        assert np.array_equal(info["sln_ja"][:], info["ja"][:])


def test_extra_equations_are_refused(function_tmpdir):
    """A solution with more equations than the flow grid has is refused.

    Mf6Adj rejects such a model as it reads it, so the forward file of a
    supported model is given one more equation than the grid has to reach the
    adjoint's own guard. This is the tripwire for the day a package that adds
    equations is read: the adjoint has to form their terms, not solve without
    them.
    """
    ws = _build_model(function_tmpdir / "m", maw=False)
    k, i, j = WELL_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.finalize()

    shutil.copy(ws / "fwd.hd5", ws / "extra.hd5")
    with h5py.File(ws / "extra.hd5", "r+") as hf:
        sln_ia = hf["gwf_info"]["sln_ia"][:]
        del hf["gwf_info"]["sln_ia"]
        hf["gwf_info"]["sln_ia"] = np.append(sln_ia, sln_ia[-1])

    adj._hdf5_name = "extra.hd5"
    with pytest.raises(Exception, match="equations beyond"):
        adj.solve_adjoint()
