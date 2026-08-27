"""
Tests for a well whose rate is reduced as its cell drains.

A well given AUTO_FLOW_REDUCE has its rate scaled by a smooth function of the
head in its cell, so the flow it produces follows the head as well as the rate
it was given. MODFLOW 6 applies the reduction only where the cell is
convertible, and puts the way it follows the head into the matrix only under
the Newton-Raphson formulation.

Cases:
  - reduced_rate    : the sensitivity to the rate a reduced well is given
                      matches a finite difference under Newton.
  - reduction_bites : that model really is reducing, so the case is not
                      vacuous.
  - no_newton_warns : the same model without Newton is reported rather than
                      returned as though it were exact.
  - picard_warns    : a convertible cell without Newton is reported on its own
                      account, whatever the wells are doing.
  - newton_quiet    : neither is reported where the model used Newton.
"""

import pathlib as pl
import shutil
import sys

import flopy
import h5py
import numpy as np

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NAME = "fr"
NROW = NCOL = 7
OBS_CELL = (0, 3, 3)
WELL_CELL = (0, 5, 5)
TOP, BOT = 10.0, 0.0
# the rate is reduced below this height over the cell bottom
FLOW_REDUCE = 0.5
THRESHOLD = BOT + FLOW_REDUCE * (TOP - BOT)
WELL_RATE = -1500.0


def _build_model(ws, rate=WELL_RATE, newton=True):
    """Build an unconfined model whose well draws its own cell down."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        outer_dvclose=1e-10,
        inner_dvclose=1e-11,
        outer_maximum=500,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=NAME,
        save_flows=True,
        newtonoptions="NEWTON" if newton else None,
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=NROW,
        ncol=NCOL,
        delr=100.0,
        delc=100.0,
        top=TOP,
        botm=[BOT],
    )
    flopy.mf6.ModflowGwfic(gwf, strt=8.0)
    # the cell must be convertible, which is the only case MODFLOW reduces
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=10.0)
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), 8.0, 500.0] for i in range(NROW)],
        pname="ghb-1",
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=[[WELL_CELL, rate]],
        auto_flow_reduce=FLOW_REDUCE,
        pname="wel-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])
    return ws


def _head(ws, cell=OBS_CELL):
    return float(flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds").get_data()[cell])


def _solve_adjoint(ws, logging_level="WARNING"):
    ws = pl.Path(ws)
    k, i, j = OBS_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level=logging_level, working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / "adjoint_solution_obs.hd5", "r") as hf:
        return np.array(hf["composite"]["wel6_q"][:])


def test_reduction_bites(function_tmpdir):
    """The model really does reduce its well, so the other case is not vacuous."""
    ws = _build_model(function_tmpdir / "base")
    well_head = _head(ws, WELL_CELL)
    assert BOT < well_head < THRESHOLD, (
        f"the well head {well_head} is not inside the reduction zone between "
        f"{BOT} and {THRESHOLD}"
    )


def test_reduced_rate(function_tmpdir):
    """The sensitivity to the rate a reduced well is given holds up."""
    dq = 1.0
    base_ws = _build_model(function_tmpdir / "base")
    pert_ws = _build_model(function_tmpdir / "pert", rate=WELL_RATE + dq)
    minus_ws = _build_model(function_tmpdir / "minus", rate=WELL_RATE - dq)
    finite_difference = (_head(pert_ws) - _head(minus_ws)) / (2.0 * dq)

    adjoint = float(_solve_adjoint(base_ws)[WELL_CELL])

    assert np.isclose(adjoint, finite_difference, rtol=1e-4), (
        f"adjoint {adjoint:.6e} against a finite difference of {finite_difference:.6e}"
    )


def test_no_newton_warns(function_tmpdir, caplog):
    """Without Newton the matrix lacks the term, and that is reported."""
    # a gentler rate, so the cell stays wet without the Newton formulation
    ws = _build_model(function_tmpdir / "picard", rate=-250.0, newton=False)
    assert BOT < _head(ws, WELL_CELL) < THRESHOLD

    _solve_adjoint(ws)

    assert any("reduces its rates" in record.message for record in caplog.records), (
        "the reduced well was not reported"
    )


def test_picard_warns(function_tmpdir, caplog):
    """A convertible cell without Newton is reported on its own account.

    The matrix leaves out how the transmissivity follows the head, which is a
    limitation of the model formulation rather than of any one package.
    """
    ws = _build_model(function_tmpdir / "picard", rate=-250.0, newton=False)
    _solve_adjoint(ws)
    assert any("convertible cells" in record.message for record in caplog.records), (
        "the standard formulation was not reported"
    )


def test_newton_quiet(function_tmpdir, caplog):
    """Neither condition is reported where the flow model used Newton."""
    ws = _build_model(function_tmpdir / "newton")
    _solve_adjoint(ws)
    assert not any(
        "convertible cells" in record.message or "reduces its rates" in record.message
        for record in caplog.records
    ), "an exact model was reported as approximate"
