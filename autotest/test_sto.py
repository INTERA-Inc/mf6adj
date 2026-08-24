"""
Tests for the storage terms that couple one time step to the next.

MODFLOW 6 selects between the specific-storage and specific-yield terms on the
cell saturation, and under SS_CONFINED_ONLY it drops the specific-storage term
entirely for a cell that is not full rather than scaling it. The adjoint has to
make the same selection or the sensitivity carries a storage term the forward
model does not have.

Cases:
  - test_storage_sensitivity[confined_only] : SS_CONFINED_ONLY, partially
                                              saturated cells, adjoint matches a
                                              finite-difference derivative.
  - test_storage_sensitivity[default]       : the same for the default specific
                                              storage formulation.
  - test_confined_only_drops_partial_cells  : the specific-storage term is
                                              dropped where the cell was not
                                              full, and kept where it was.
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

NAME = "sto"
NROW = NCOL = 7
TOP, BOTM = 10.0, 0.0
WELL_CELL = (0, 3, 3)
OBS_CELL = (0, 1, 1)
WELL_RATE = -60.0
NSTP = 4


def _build_model(ws, rate, confined_only):
    """A partially saturated transient model driven by one well."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    # several steps, so the saturation carried between them actually changes
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(400.0, NSTP, 1.0)])
    flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        outer_dvclose=1e-9,
        inner_dvclose=1e-10,
        outer_maximum=200,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=NAME, newtonoptions="NEWTON", save_flows=True
    )
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=NROW, ncol=NCOL, delr=100.0, delc=100.0, top=TOP, botm=BOTM
    )
    flopy.mf6.ModflowGwfic(gwf, strt=8.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1.0)
    kwargs = {"iconvert": 1, "ss": 1.0e-3, "sy": 0.2, "transient": {0: True}}
    if confined_only:
        kwargs["ss_confined_only"] = True
    flopy.mf6.ModflowGwfsto(gwf, **kwargs)
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), 8.0, 50.0] for i in range(NROW)],
        pname="ghb-1",
    )
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[WELL_CELL, rate]], pname="wel-1")
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{NAME}.hds",
        budget_filerecord=f"{NAME}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-15:])
    return ws


def _final_head(ws):
    head = flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds")
    return float(head.get_data(kstpkper=head.get_kstpkper()[-1])[OBS_CELL])


def _solve_adjoint(ws):
    """Sensitivity of the final head at the observation cell to the well rate."""
    ws = pl.Path(ws)
    k, i, j = OBS_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 {NSTP} {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / "adjoint_solution_obs.hd5", "r") as hf:
        return float(hf["composite"]["wel6_q"][WELL_CELL])


@pytest.mark.parametrize(
    "confined_only", [True, False], ids=["confined_only", "default"]
)
def test_storage_sensitivity(function_tmpdir, confined_only):
    """The adjoint matches a finite-difference derivative in a drained cell."""
    dq = -3.0  # small enough to stay in the linear range

    base_ws = _build_model(function_tmpdir / "base", WELL_RATE, confined_only)
    pert_ws = _build_model(function_tmpdir / "pert", WELL_RATE + dq, confined_only)
    finite_difference = (_final_head(pert_ws) - _final_head(base_ws)) / dq

    # the cells must be partly drained, or the two formulations agree trivially
    head = flopy.utils.HeadFile(base_ws / f"{NAME}.hds").get_data()
    saturation = np.clip((head - BOTM) / (TOP - BOTM), 0.0, 1.0)
    assert saturation.max() < 0.99, "the model has to leave cells partly drained"

    adjoint = _solve_adjoint(base_ws)
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} does not match the finite-difference "
        f"derivative {finite_difference:.6e}"
    )


def test_confined_only_drops_partial_cells(function_tmpdir):
    """SS_CONFINED_ONLY drops the specific-storage term below full saturation."""
    ws = _build_model(function_tmpdir / "run", WELL_RATE, True)
    k, i, j = OBS_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 {NSTP} {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.finalize()

    with h5py.File(ws / "fwd.hd5", "r") as hf:
        key = next(k for k in hf if k.startswith("solution_"))
        drhsdh = hf[key]["drhsdh"][:]
        sat = hf[key]["sat"][:]

    area, sy = 100.0 * 100.0, 0.2
    dt = 400.0 / NSTP
    partial = sat < 1.0
    assert partial.any(), "the model has to leave cells partly drained"
    # below full saturation only the specific-yield term survives
    expected = -area * sy / dt
    assert np.allclose(drhsdh[partial], expected, rtol=1e-8), (
        "specific storage is still applied where MODFLOW 6 drops it"
    )
