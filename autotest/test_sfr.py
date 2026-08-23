"""
Tests for streamflow routing (SFR) performance measures.

A reach stage follows the flow through the reach, so it is a dependent variable
in the same sense as a lake stage. Holding it fixed returns a partial
derivative. These cases measure how far that is from the total derivative.

Cases:
  - sfr_shallow      : a steep, shallow stream barely moves its stage, so the
                       frozen-stage result is already close.
  - sfr_deep         : a slow, deep stream moves its stage, and the reach
                       equation carries that.
  - sfr_fully_losing : a stream that loses all of its inflow has a leakage the
                       pumping cannot change, which needs the reach that gives
                       up its whole flow to be coupled to the reaches above it
                       (xfail).
"""

import glob
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

NROW, NCOL, DELRC = 8, 8, 100.0
REACH_ROW = 3
WELL_CELL = (0, 6, 6)
WELL_RATE = -2000.0
STRTOP = 6.0


def _build_model(ws, well_rate, inflow=5000.0, rhk=5.0, rgrd=1.0e-3, man=0.03):
    """Build a single-layer model with a chain of reaches across it."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    sim = flopy.mf6.MFSimulation(sim_name="sf", sim_ws=str(ws), exe_name=str(mf6_bin))
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0, 1, 1.0)])
    flopy.mf6.ModflowIms(
        sim,
        outer_dvclose=1e-11,
        inner_dvclose=1e-12,
        outer_maximum=500,
        complexity="complex",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="sf", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=NROW,
        ncol=NCOL,
        delr=DELRC,
        delc=DELRC,
        top=10.0,
        botm=[-20.0],
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0, icelltype=0)
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, sy=0.2, transient={0: True})
    spd = []
    for i in range(NROW):
        spd.append([(0, i, 0), 5.5, 1000.0])
        spd.append([(0, i, NCOL - 1), 4.5, 1000.0])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=spd, pname="ghb-edge")
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=[[WELL_CELL, well_rate]], pname="wel-1"
    )

    packagedata, connectiondata = [], []
    for n in range(NCOL):
        nconn = 1 if n in (0, NCOL - 1) else 2
        packagedata.append(
            [
                n,
                (0, REACH_ROW, n),
                DELRC,
                5.0,
                rgrd,
                STRTOP - 0.01 * n,
                1.0,
                rhk,
                man,
                nconn,
                1.0,
                0,
            ]
        )
        conn = [n]
        if n > 0:
            conn.append(n - 1)
        if n < NCOL - 1:
            # a downstream connection is given as a negative reach number
            conn.append(-(n + 1))
        connectiondata.append(conn)
    flopy.mf6.ModflowGwfsfr(
        gwf,
        nreaches=NCOL,
        packagedata=packagedata,
        connectiondata=connectiondata,
        perioddata={0: [[0, "inflow", inflow]]},
        unit_conversion=128390.0,
        pname="sfr-1",
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="sf.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-15:])
    return ws


def _write_adj(ws):
    """Measure the exchange between every reach and the aquifer."""
    path = pl.Path(ws) / "test.adj"
    with open(path, "w") as f:
        f.write("begin performance_measure pm\n")
        for n in range(NCOL):
            f.write(f"1 1 1 {REACH_ROW + 1} {n + 1} sfr-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return path


def _solve(ws):
    adj = mf6adj.Mf6Adj(
        "test.adj", str(lib_name), logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model()
    dfs = adj.solve_adjoint()
    adj.finalize()
    return dfs


def _measure_value(ws):
    """Return the reach exchange mf6adj summed."""
    path = sorted(glob.glob(str(pl.Path(ws) / "sf_*.hd5")))[-1]
    with h5py.File(path, "r") as hf:
        key = [k for k in hf if k.startswith("solution_")][-1]
        return float(np.sum(hf[key]["sfr-1"]["simvals"][:]))


def _compare(tmpdir, **kwargs):
    """Return the adjoint sensitivity and its finite-difference counterpart."""
    dq = -50.0
    base = _build_model(tmpdir / "base", WELL_RATE, **kwargs)
    _write_adj(base)
    _solve(base)
    pert = _build_model(tmpdir / "pert", WELL_RATE + dq, **kwargs)
    _write_adj(pert)
    _solve(pert)

    finite_difference = (_measure_value(pert) - _measure_value(base)) / dq
    with h5py.File(base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])
    return adjoint, finite_difference


def test_sfr_shallow(tmp_path):
    """A steep, shallow stream hardly moves its stage, so freezing it is close."""
    adjoint, finite_difference = _compare(tmp_path, rgrd=1.0e-3, man=0.03, rhk=5.0)
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} against finite difference {finite_difference:.6e}"
    )


def test_sfr_deep(tmp_path):
    """A slow, deep stream moves its stage, and the reach equation carries that.

    Depth follows flow, so a stream with a gentle slope and a rough bed carries
    its water deep and slowly. Pumping takes water from the stream, the flow
    drops, and the stage falls with it. Holding the stage fixed misses that
    second effect and is a quarter out.
    """
    adjoint, finite_difference = _compare(tmp_path, rgrd=1.0e-5, man=0.3, rhk=5.0)
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} against finite difference {finite_difference:.6e}"
    )


@pytest.mark.xfail(
    reason="the reach that gives up its whole flow leaks its own inflow rather "
    "than a head-dependent amount, and that coupling to the reaches above it is "
    "not formed (INTERA-Inc/mf6adj#78)",
    strict=False,
)
def test_sfr_fully_losing(tmp_path):
    """A stream that loses all its inflow has a leakage pumping cannot change.

    Every drop that enters the stream reaches the aquifer, so the total
    exchange is the inflow whatever the pumping does, and the adjoint has to
    return zero. The last reach gives up all of the water it carries, so its
    leakage follows the reaches above it rather than its own stage; until that
    coupling is formed the reach equations cannot pin the total.
    """
    adjoint, finite_difference = _compare(tmp_path, rgrd=1.0e-5, man=0.3, rhk=50.0)
    assert abs(finite_difference) < 1e-6, (
        "the stream should lose all of its inflow, so the exchange cannot "
        f"respond to pumping, but the finite difference is {finite_difference:.6e}"
    )
    assert abs(adjoint) < 1e-6, (
        f"the adjoint reports {adjoint:.6e} where there is no sensitivity"
    )
