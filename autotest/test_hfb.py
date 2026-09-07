"""
Tests for the horizontal flow barrier terms of the adjoint solution.

A barrier adds no equations and exchanges no water. It changes the conductance
of the connections it sits on, and MODFLOW writes that changed conductance into
the array the flow equations are assembled from. A sensitivity to hydraulic
conductivity is the derivative of that conductance, so it has to carry the
barrier; formed without it, the sensitivity answers for a connection the model
does not have.

The barrier is given as a hydraulic characteristic. A positive one puts a
barrier of that conductance in series with the connection, which always lowers
it. A value of zero or less is read as a multiplier on the conductance instead,
which lowers it, raises it, or closes the connection entirely depending on the
magnitude. The two forms have different derivatives.

Cases:
  - test_horizontal_barrier   : the sensitivity to k11 at a cell beside a
                                horizontal barrier matches a finite-difference
                                derivative, and a cell with no barrier is
                                unaffected.
  - test_vertical_barrier     : the same for k33 across a barrier on a vertical
                                connection.
  - test_multiplier_form      : the same for a hydraulic characteristic given
                                as a multiplier, where it lowers the
                                conductance, raises it, and closes it.
  - test_factor_follows_the_form : a barrier in series always lowers the
                                conductance, and a multiplier does whatever it
                                says, including raising it.
  - test_barrier_matters      : dropping the barrier from the derivative moves
                                the answer, so the tests above are testing it.
  - test_no_barrier_unchanged : a model with no barrier writes no factor and
                                reports what it did before.
  - test_factor_is_one_away_from_a_barrier : the factor is one on every
                                connection a barrier does not sit on.
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

NAME = "hfb"
NLAY, NROW, NCOL = 2, 5, 5
TOP, BOTM = 10.0, [0.0, -10.0]
K, K33 = 10.0, 1.0
OBS = (0, 2, 3)
# the vertical case is driven between layers, so it is watched below the barrier
VERTICAL_OBS = (1, 2, 3)
# a cell on the upstream side of the line of horizontal barriers
HORIZONTAL_CELL = (0, 2, 1)
# a cell above a vertical barrier
VERTICAL_CELL = (0, 2, 2)
# layer 1 carries no horizontal barrier
AWAY_CELL = (1, 2, 2)


def _build(
    ws, barrier="horizontal", hydchr=1.0e-3, kcell=None, kmult=1.0, k33cell=None
):
    """A confined model with constant heads at each end and one line of barriers."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    k = np.full((NLAY, NROW, NCOL), K)
    k33 = np.full((NLAY, NROW, NCOL), K33)
    if kcell is not None:
        k[kcell] *= kmult
    if k33cell is not None:
        k33[k33cell] *= kmult
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(
        sim, complexity="simple", outer_dvclose=1e-11, inner_dvclose=1e-12
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=NLAY, nrow=NROW, ncol=NCOL, delr=10.0, delc=10.0, top=TOP, botm=BOTM
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    # confined, so MODFLOW folds the barrier into the stored conductance
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=k, k33=k33)
    if barrier == "vertical":
        # in at the top on one side and out at the bottom on the other, so the
        # water has to cross the connections the barriers sit on
        chd = [[(0, i, 0), 6.0] for i in range(NROW)]
        chd += [[(1, i, NCOL - 1), 4.0] for i in range(NROW)]
    else:
        chd = [[(0, i, 0), 6.0] for i in range(NROW)]
        chd += [[(0, i, NCOL - 1), 4.0] for i in range(NROW)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, pname="chd-1")
    if barrier == "horizontal":
        spd = [[(0, i, 1), (0, i, 2), hydchr] for i in range(NROW)]
    elif barrier == "vertical":
        spd = [[(0, i, 2), (1, i, 2), hydchr] for i in range(NROW)]
    else:
        spd = None
    if spd is not None:
        flopy.mf6.ModflowGwfhfb(gwf, stress_period_data=spd, pname="hfb-1")
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])
    return ws


def _head(ws, obs=OBS):
    return float(flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds").get_data()[obs])


def _composite(ws, obs=OBS):
    """Solve the adjoint for a head measure and return the composite results."""
    ws = pl.Path(ws)
    k, i, j = obs
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / "adjoint_solution_obs.hd5", "r") as hf:
        return {key: hf["composite"][key][:] for key in hf["composite"]}


def _finite_difference(tmpdir, cell, vertical=False, obs=OBS, **kwargs):
    """A central difference of the observed head with respect to K at one cell."""
    d = 0.001
    key = "k33cell" if vertical else "kcell"
    base = K33 if vertical else K
    plus = _build(tmpdir / "plus", **{key: cell}, kmult=1 + d, **kwargs)
    minus = _build(tmpdir / "minus", **{key: cell}, kmult=1 - d, **kwargs)
    return (_head(plus, obs) - _head(minus, obs)) / (2.0 * d * base)


def test_horizontal_barrier(function_tmpdir):
    """A cell beside a horizontal barrier matches a finite-difference derivative."""
    composite = _composite(_build(function_tmpdir / "base"))

    for label, cell in (("barrier", HORIZONTAL_CELL), ("away", AWAY_CELL)):
        fd = _finite_difference(function_tmpdir, cell)
        assert abs(fd) > 1.0e-8, f"{label} cell has no sensitivity to test"
        assert composite["k11"][cell] == pytest.approx(fd, rel=1.0e-3), label


def test_vertical_barrier(function_tmpdir):
    """A barrier on a vertical connection is carried into the k33 sensitivity."""
    composite = _composite(
        _build(function_tmpdir / "base", barrier="vertical"), obs=VERTICAL_OBS
    )

    fd = _finite_difference(
        function_tmpdir,
        VERTICAL_CELL,
        vertical=True,
        obs=VERTICAL_OBS,
        barrier="vertical",
    )
    assert abs(fd) > 1.0e-10, "the vertical connection has no sensitivity to test"
    assert composite["k33"][VERTICAL_CELL] == pytest.approx(fd, rel=1.0e-3)


@pytest.mark.parametrize(
    "hydchr", [-0.01, -2.0, 0.0], ids=["reduces", "raises", "closes"]
)
def test_multiplier_form(function_tmpdir, hydchr):
    """A hydraulic characteristic of zero or less multiplies the conductance.

    The multiplier is not bounded by one, so this form can raise the
    conductance of a connection as well as lower it, and a value of zero closes
    the connection entirely. The derivative of a multiplier is not the
    derivative of a barrier in series, so the two forms cannot share one
    expression.
    """
    composite = _composite(_build(function_tmpdir / "base", hydchr=hydchr))

    fd = _finite_difference(function_tmpdir, HORIZONTAL_CELL, hydchr=hydchr)
    assert composite["k11"][HORIZONTAL_CELL] == pytest.approx(
        fd, rel=1.0e-3, abs=1.0e-12
    )


@pytest.mark.parametrize(
    "hydchr,expected", [(1.0e-3, "below"), (-2.0, "above"), (0.0, "zero")]
)
def test_factor_follows_the_form(function_tmpdir, hydchr, expected):
    """A barrier does not have to lower the conductance of what it sits on.

    In series it always does, because two conductances in series carry less
    than either. As a multiplier it does whatever the multiplier says, so a
    magnitude above one raises the conductance and the derivative with it.
    """
    ws = _build(function_tmpdir / "base", hydchr=hydchr)
    _composite(ws)

    with h5py.File(ws / "fwd.hd5", "r") as hf:
        factor = hf["solution_kper:00000_kstp:00000"]["hfb_factor"][:]
    scaled = factor[factor != 1.0]

    if expected == "below":
        assert (scaled < 1.0).all() and (scaled > 0.0).all()
    elif expected == "above":
        # the multiplier is applied once, not squared, so it is the factor
        assert scaled == pytest.approx(-hydchr)
        assert (scaled > 1.0).all()
    else:
        assert scaled == pytest.approx(0.0)


def test_barrier_matters(function_tmpdir):
    """Dropping the barrier from the derivative moves the answer a long way."""
    from mf6adj.packages import npf

    base = _build(function_tmpdir / "base")
    fd = _finite_difference(function_tmpdir, HORIZONTAL_CELL)

    original = npf.lam_dresdk_h

    def without_barrier(*args, **kwargs):
        kwargs["hfb_factor"] = None
        return original(*args, **kwargs)

    npf.lam_dresdk_h = without_barrier
    try:
        dropped = _composite(base)["k11"][HORIZONTAL_CELL]
    finally:
        npf.lam_dresdk_h = original

    assert abs(dropped - fd) > 10.0 * abs(fd), (
        "the barrier made no difference to the derivative, so this model does "
        "not exercise it"
    )


def test_no_barrier_unchanged(function_tmpdir):
    """A model with no barrier writes no factor and reports what it did before."""
    ws = _build(function_tmpdir / "base", barrier=None)
    composite = _composite(ws)

    with h5py.File(ws / "fwd.hd5", "r") as hf:
        step = hf["solution_kper:00000_kstp:00000"]
        assert not step.attrs["has_hfb"]
        assert "hfb_factor" not in step

    for cell in (HORIZONTAL_CELL, AWAY_CELL):
        fd = _finite_difference(function_tmpdir, cell, barrier=None)
        assert composite["k11"][cell] == pytest.approx(fd, rel=1.0e-3)


def test_factor_is_one_away_from_a_barrier(function_tmpdir):
    """Only the connections a barrier sits on are scaled."""
    ws = _build(function_tmpdir / "base")
    _composite(ws)

    with h5py.File(ws / "fwd.hd5", "r") as hf:
        step = hf["solution_kper:00000_kstp:00000"]
        assert step.attrs["has_hfb"]
        factor = step["hfb_factor"][:]

    scaled = factor != 1.0
    assert scaled.sum() == NROW, "one connection per barrier should be scaled"
    # the barrier is strong, so what it leaves is a small fraction of what the
    # connection carried without it
    assert (factor[scaled] < 1.0e-4).all()
