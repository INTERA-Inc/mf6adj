"""
Tests for the recharge sensitivity.

MODFLOW 6 treats recharge as a rate over the cell area, so the flow a recharge
value produces is the rate times the area and the sensitivity of a performance
measure to it carries that area. A well rate is already a flow and does not.

Cases:
  - test_recharge_sensitivity      : the adjoint matches a finite-difference
                                     derivative with respect to the recharge
                                     rate.
  - test_recharge_scales_with_area : the recharge sensitivity is the well
                                     sensitivity times the cell area.
  - test_specified_flux_measure_rejected : measuring the flow of a well, list
                                     recharge, or array recharge is an error
                                     rather than a silently zero sensitivity.
  - test_head_dependent_measure_allowed : measuring a head-dependent boundary in
                                     the same model is unaffected.
"""

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

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NAME = "rch"
NROW = NCOL = 7
DELRC = 250.0  # a cell area of 62500, far from 1.0
CELL_AREA = DELRC * DELRC
TOP, BOTM = 10.0, -20.0
OBS_CELL = (0, 3, 3)
BASE_RECHARGE = 1.0e-3


def _build_model(ws, recharge, array_recharge=False):
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(
        sim, complexity="simple", outer_dvclose=1e-10, inner_dvclose=1e-11
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=NROW, ncol=NCOL, delr=DELRC, delc=DELRC, top=TOP, botm=BOTM
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), 0.0, 100.0] for i in range(NROW)],
        pname="ghb-1",
    )
    # recharge given as a list of cells, or as an array over the grid; both are
    # applied at a specified rate, so neither flow follows the head
    if array_recharge:
        flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge, pname="rcha-1")
    else:
        flopy.mf6.ModflowGwfrch(
            gwf,
            stress_period_data=[
                [(0, i, j), recharge] for i in range(NROW) for j in range(NCOL)
            ],
            pname="rch-1",
        )
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[OBS_CELL, 0.0]], pname="wel-1")
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{NAME}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        budget_filerecord=f"{NAME}.cbc",
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-15:])
    return ws


def _head(ws):
    return float(flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds").get_data()[OBS_CELL])


def _solve_adjoint(ws):
    ws = pl.Path(ws)
    k, i, j = OBS_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / "adjoint_solution_obs.hd5", "r") as hf:
        return (
            hf["composite"]["rch6_recharge"][:].copy(),
            hf["composite"]["wel6_q"][:].copy(),
        )


def test_recharge_sensitivity(function_tmpdir):
    """The adjoint matches a finite difference in the recharge rate."""
    dr = 1.0e-5

    base_ws = _build_model(function_tmpdir / "base", BASE_RECHARGE)
    pert_ws = _build_model(function_tmpdir / "pert", BASE_RECHARGE + dr)
    # every cell is recharged, so the finite difference is the summed sensitivity
    finite_difference = (_head(pert_ws) - _head(base_ws)) / dr

    recharge_sens, _ = _solve_adjoint(base_ws)
    adjoint = float(np.sum(recharge_sens))

    assert np.isclose(adjoint, finite_difference, rtol=1e-3), (
        f"adjoint {adjoint:.6e} does not match the finite-difference "
        f"derivative {finite_difference:.6e}"
    )


def test_recharge_scales_with_area(function_tmpdir):
    """The recharge sensitivity is the well sensitivity times the cell area."""
    ws = _build_model(function_tmpdir / "run", BASE_RECHARGE)
    recharge_sens, well_sens = _solve_adjoint(ws)

    assert np.allclose(recharge_sens, well_sens * CELL_AREA, rtol=1e-10), (
        "the recharge sensitivity does not carry the cell area"
    )


def _write_measure(ws, package):
    """Write a flux measure over every cell of one package."""
    ws = pl.Path(ws)
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure flux\n")
        for i in range(NROW):
            for j in range(NCOL):
                f.write(f"  1 1 1 {i + 1} {j + 1} {package} direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")


@pytest.mark.parametrize(
    "package, array_recharge",
    [("wel-1", False), ("rch-1", False), ("rcha-1", True)],
    ids=["wel", "rch", "rcha"],
)
def test_specified_flux_measure_rejected(function_tmpdir, package, array_recharge):
    """Measuring the flow of a specified-flux package is an error.

    A well rate and a recharge rate are both applied as specified, so neither
    flow follows the head and the sensitivity of a measure of either is zero
    everywhere. Reporting those zeros looks like an answer.
    """
    ws = _build_model(function_tmpdir / package, BASE_RECHARGE, array_recharge)
    _write_measure(ws, package)

    with pytest.raises(Exception, match="specified rather than calculated"):
        mf6adj.Mf6Adj(
            "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
        )


def test_head_dependent_measure_allowed(function_tmpdir):
    """A head-dependent boundary in the same model is still accepted."""
    ws = _build_model(function_tmpdir / "allowed", BASE_RECHARGE)
    with open(pl.Path(ws) / "pm.dat", "w") as f:
        f.write("begin performance_measure flux\n")
        for i in range(NROW):
            f.write(f"  1 1 1 {i + 1} 1 ghb-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")

    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(pl.Path(ws) / "adjoint_solution_flux.hd5", "r") as hf:
        sensitivity = hf["composite"]["wel6_q"][:]
    assert np.abs(sensitivity).max() > 0.0, (
        "a head-dependent measure should have a non-zero sensitivity"
    )
