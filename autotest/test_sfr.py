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
                       pumping cannot change, and the adjoint returns zero.
  - two_packages     : two streamflow-routing packages in one model each keep
                       their own reaches.
  - multi_period     : a stream measured after several periods.
  - sfr_with_lake    : a model holding both a stream and a lake borders them
                       together.
  - branching        : a stream that splits sends its flow on in the shares the
                       reaches below it are given.
  - xs_rating        : the cross-section rating reproduces the discharge the
                       forward model solved the reach depth against, for a
                       channel with sloping banks and for one closed by a
                       vertical wall.
  - xs_derivative    : the derivative of that rating matches a central
                       difference through a break in the section.
  - xs_negative_p    : a section whose wetted perimeter has gone negative, as
                       MODFLOW lets it, still carries a leakage derivative.
  - sfr_cross_section: a stream whose reaches carry a cross section, whose
                       conductance follows the depth as well as the stage.
  - diversion_rule   : the derivative of each diversion rule is recovered from
                       the flows, in both regimes of the piecewise ones.
  - diversion_unknown: flows that leave two rules diverting the same amount
                       are reported as undetermined rather than guessed at.
  - diversion_warned : an undetermined rule is reported on whichever time step
                       it arises, and named, and reported only once.
  - sfr_diversion    : a stream carrying a diversion under each of the four
                       rules.
"""

import glob
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

import mf6adj.advanced_packages.sfr
from mf6adj.advanced_packages.sfr import leakage_ratio
from mf6adj.advanced_packages.sfr_cross_section import (
    mannings_section,
    wetted_perimeter,
)
from mf6adj.advanced_packages.sfr_diversion import diversion_derivative

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NROW, NCOL, DELRC = 8, 8, 100.0
REACH_ROW = 3
WELL_CELL = (0, 6, 6)
WELL_RATE = -2000.0
STRTOP = 6.0


# a trapezoidal section, given as a fraction of the reach width and a height
# over the streambed, with a roughness of the reach roughness throughout
CROSS_SECTION = [
    (0.0, 1.2, 1.0),
    (0.2, 0.0, 1.0),
    (0.8, 0.0, 1.0),
    (1.0, 1.2, 1.0),
]

# the same channel closed by a vertical wall standing on the bank, which is the
# geometry a wetted vertical face is formed for; the wall stands low enough that
# the water reaches it
WALLED_SECTION = [
    (0.0, 0.5, 1.0),
    (0.0, 0.03, 1.0),
    (0.3, 0.0, 1.0),
    (0.7, 0.0, 1.0),
    (1.0, 0.03, 1.0),
    (1.0, 0.5, 1.0),
]


# the reach the diversion is taken from, and the row the diverted reach sits in
DIVERT_FROM = 3
DIVERT_ROW = REACH_ROW + 2


def _build_model(
    ws,
    well_rate,
    inflow=5000.0,
    rhk=5.0,
    rgrd=1.0e-3,
    man=0.03,
    cross_section=False,
    section=None,
    diversion=None,
):
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
        ndv = 0
        if diversion is not None and n == DIVERT_FROM:
            nconn += 1
            ndv = 1
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
                ndv,
            ]
        )
        conn = [n]
        if n > 0:
            conn.append(n - 1)
        if n < NCOL - 1:
            # a downstream connection is given as a negative reach number
            conn.append(-(n + 1))
        if ndv:
            conn.append(-NCOL)
        connectiondata.append(conn)

    diversions = None
    if diversion is not None:
        rule = diversion[0]
        packagedata.append(
            [
                NCOL,
                (0, DIVERT_ROW, DIVERT_FROM),
                DELRC,
                5.0,
                rgrd,
                STRTOP - 0.01 * DIVERT_FROM,
                1.0,
                rhk,
                man,
                1,
                # a diverted reach takes what the diversion gives it rather
                # than a share, so MODFLOW requires an upstream fraction of zero
                0.0,
                0,
            ]
        )
        connectiondata.append([NCOL, DIVERT_FROM])
        diversions = [[DIVERT_FROM, 0, NCOL, rule]]
    crosssections = None
    if cross_section:
        crosssections = []
        for n in range(NCOL):
            name = f"xs{n}"
            flopy.mf6.ModflowUtlsfrtab(
                gwf,
                nrow=len(section or CROSS_SECTION),
                ncol=3,
                table=section or CROSS_SECTION,
                filename=f"{name}.txt",
                pname=name,
            )
            crosssections.append([n, f"{name}.txt"])
    perioddata = [[0, "inflow", inflow]]
    if diversion is not None:
        perioddata.append([DIVERT_FROM, "diversion", 0, diversion[1]])
    flopy.mf6.ModflowGwfsfr(
        gwf,
        nreaches=NCOL + (0 if diversion is None else 1),
        packagedata=packagedata,
        connectiondata=connectiondata,
        crosssections=crosssections,
        diversions=diversions,
        perioddata={0: perioddata},
        unit_conversion=128390.0,
        pname="sfr-1",
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="sf.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-15:])
    return ws


def _write_adj(ws, diversion=None):
    """Measure the exchange between every reach and the aquifer."""
    path = pl.Path(ws) / "test.adj"
    with open(path, "w") as f:
        f.write("begin performance_measure pm\n")
        for n in range(NCOL):
            f.write(f"1 1 1 {REACH_ROW + 1} {n + 1} sfr-1 direct 1.0 -1.0e+30\n")
        if diversion is not None:
            # the diverted reach carries exchange of its own
            f.write(
                f"1 1 1 {DIVERT_ROW + 1} {DIVERT_FROM + 1} sfr-1 direct 1.0 -1.0e+30\n"
            )
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
    _write_adj(base, diversion=kwargs.get("diversion"))
    _solve(base)
    pert = _build_model(tmpdir / "pert", WELL_RATE + dq, **kwargs)
    _write_adj(pert, diversion=kwargs.get("diversion"))
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


def test_sfr_fully_losing(tmp_path):
    """A stream that loses all its inflow has a leakage pumping cannot change.

    Every drop that enters the stream reaches the aquifer, so the total
    exchange is the inflow whatever the pumping does, and the adjoint has to
    return zero. The last reach gives up all of the water it carries, so its
    leakage follows the reaches above it rather than its own stage, and it is
    that coupling which pins the total.
    """
    adjoint, finite_difference = _compare(tmp_path, rgrd=1.0e-5, man=0.3, rhk=50.0)
    assert abs(finite_difference) < 1e-6, (
        "the stream should lose all of its inflow, so the exchange cannot "
        f"respond to pumping, but the finite difference is {finite_difference:.6e}"
    )
    assert abs(adjoint) < 1e-6, (
        f"the adjoint reports {adjoint:.6e} where there is no sensitivity"
    )


def test_two_sfr_packages(tmp_path):
    """Two streamflow-routing packages in one model each keep their own reaches.

    Every package writes its own forward terms and its own columns, so a
    measure on one must not pick up the other.
    """
    dq = -50.0
    first_row, second_row = 2, 5

    def build(ws, rate):
        ws = pl.Path(ws)
        if ws.exists():
            shutil.rmtree(ws)
        sim = flopy.mf6.MFSimulation(
            sim_name="sf", sim_ws=str(ws), exe_name=str(mf6_bin)
        )
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
            gwf, stress_period_data=[[WELL_CELL, rate]], pname="wel-1"
        )
        for name, row in (("sfr-a", first_row), ("sfr-b", second_row)):
            packagedata, connectiondata = [], []
            for n in range(NCOL):
                nconn = 1 if n in (0, NCOL - 1) else 2
                packagedata.append(
                    [
                        n,
                        (0, row, n),
                        DELRC,
                        5.0,
                        1.0e-3,
                        STRTOP - 0.01 * n,
                        1.0,
                        5.0,
                        0.03,
                        nconn,
                        1.0,
                        0,
                    ]
                )
                conn = [n]
                if n > 0:
                    conn.append(n - 1)
                if n < NCOL - 1:
                    conn.append(-(n + 1))
                connectiondata.append(conn)
            flopy.mf6.ModflowGwfsfr(
                gwf,
                nreaches=NCOL,
                packagedata=packagedata,
                connectiondata=connectiondata,
                perioddata={0: [[0, "inflow", 5000.0]]},
                unit_conversion=128390.0,
                pname=name,
                filename=f"sf.{name}.sfr",
            )
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord="sf.hds", saverecord=[("HEAD", "ALL")]
        )
        sim.write_simulation(silent=True)
        success, buff = sim.run_simulation(silent=True)
        assert success, "\n".join(buff[-15:])

        # measure the exchange of the first package only
        path = pl.Path(ws) / "test.adj"
        with open(path, "w") as f:
            f.write("begin performance_measure pm\n")
            for n in range(NCOL):
                f.write(f"1 1 1 {first_row + 1} {n + 1} sfr-a direct 1.0 -1.0e+30\n")
            f.write("end performance_measure\n")
        _solve(ws)
        return ws

    ws_base = build(tmp_path / "base", WELL_RATE)
    ws_pert = build(tmp_path / "pert", WELL_RATE + dq)

    def measured(ws):
        path = sorted(glob.glob(str(pl.Path(ws) / "sf_*.hd5")))[-1]
        with h5py.File(path, "r") as hf:
            key = [k for k in hf if k.startswith("solution_")][-1]
            names = sorted(k for k in hf[key] if k.startswith("sfr-"))
            return float(np.sum(hf[key]["sfr-a"]["simvals"][:])), names

    base, names = measured(ws_base)
    pert, _ = measured(ws_pert)
    assert names == ["sfr-a", "sfr-b"], (
        f"each package should write its own group, got {names}"
    )

    finite_difference = (pert - base) / dq
    with h5py.File(ws_base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} against finite difference {finite_difference:.6e}"
    )


def _build_with_lake(ws, well_rate, nper=1, with_lake=False, rgrd=1.0e-5, man=0.3):
    """Build the stream model, optionally over several periods or with a lake."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    lake_cells = [(6, 2), (6, 3)]
    nlay = 2 if with_lake else 1
    botm = [0.0, -20.0] if with_lake else [-20.0]
    sim = flopy.mf6.MFSimulation(sim_name="sf", sim_ws=str(ws), exe_name=str(mf6_bin))
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=[(50.0, 1, 1.0)] * nper)
    flopy.mf6.ModflowIms(
        sim,
        outer_dvclose=1e-11,
        inner_dvclose=1e-12,
        outer_maximum=500,
        complexity="complex",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="sf", save_flows=True)
    idomain = np.ones((nlay, NROW, NCOL), dtype=int)
    if with_lake:
        for i, j in lake_cells:
            idomain[0, i, j] = 0
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=NROW,
        ncol=NCOL,
        delr=DELRC,
        delc=DELRC,
        top=10.0,
        botm=botm,
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0, icelltype=0)
    flopy.mf6.ModflowGwfsto(
        gwf, ss=1e-5, sy=0.2, transient=dict.fromkeys(range(nper), True)
    )
    spd = []
    for k in range(nlay):
        for i in range(NROW):
            spd.append([(k, i, 0), 5.5, 1000.0])
            spd.append([(k, i, NCOL - 1), 4.5, 1000.0])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=spd, pname="ghb-edge")
    well = (nlay - 1, 6, 6)
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data={0: [[well, well_rate]]}, pname="wel-1"
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
                5.0,
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
            conn.append(-(n + 1))
        connectiondata.append(conn)
    flopy.mf6.ModflowGwfsfr(
        gwf,
        nreaches=NCOL,
        packagedata=packagedata,
        connectiondata=connectiondata,
        perioddata={0: [[0, "inflow", 5000.0]]},
        unit_conversion=128390.0,
        pname="sfr-1",
    )
    if with_lake:
        lconn = [
            [0, m, (1, i, j), "vertical", 0.1, 0.0, 0.0, 0.0, 0.0]
            for m, (i, j) in enumerate(lake_cells)
        ]
        flopy.mf6.ModflowGwflak(
            gwf,
            nlakes=1,
            noutlets=0,
            ntables=0,
            packagedata=[[0, 6.0, len(lconn)]],
            connectiondata=lconn,
            pname="lak-1",
        )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="sf.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-15:])

    path = pl.Path(ws) / "test.adj"
    with open(path, "w") as f:
        f.write("begin performance_measure pm\n")
        for n in range(NCOL):
            f.write(f"{nper} 1 1 {REACH_ROW + 1} {n + 1} sfr-1 direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    _solve(ws)
    return ws, well


def _compare_with_lake(tmpdir, **kwargs):
    dq = -50.0
    base, well = _build_with_lake(tmpdir / "base", WELL_RATE, **kwargs)
    pert, _ = _build_with_lake(tmpdir / "pert", WELL_RATE + dq, **kwargs)
    finite_difference = (_measure_value(pert) - _measure_value(base)) / dq
    with h5py.File(base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][well])
    return adjoint, finite_difference


def test_sfr_multi_period(tmp_path):
    """A stream measured after several periods.

    A reach carries no storage, so nothing passes between steps, but the reach
    equations still have to be built and taken apart at every step of the
    backward sweep.
    """
    adjoint, finite_difference = _compare_with_lake(tmp_path, nper=5)
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} against finite difference {finite_difference:.6e}"
    )


def test_sfr_with_lake(tmp_path):
    """A model holding both a stream and a lake borders them together.

    The reach rows sit outside the lake rows, so the two have to share one
    system without treading on each other.
    """
    adjoint, finite_difference = _compare_with_lake(
        tmp_path, with_lake=True, rgrd=1.0e-3, man=0.03
    )
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} against finite difference {finite_difference:.6e}"
    )


def test_sfr_branching(tmp_path):
    """A stream that splits sends its flow on in the shares given below it.

    Every other case is a single chain, where each reach passes its whole flow
    to one reach below. Here reach 1 splits into two branches taking 0.7 and
    0.3, and the measure follows one branch.

    The shares are checked directly. The finite difference does not isolate
    them: the share enters only the reach equations, and its contribution to
    this measure is small beside the head response, so a wrong share still
    reproduces the total.
    """
    dq = -50.0
    # reach: cell, and the reaches it flows to
    cells = [(3, 0), (3, 1), (2, 2), (4, 2), (2, 3), (4, 3)]
    downstream = {0: [1], 1: [2, 3], 2: [4], 3: [5], 4: [], 5: []}
    upstream = {0: [], 1: [0], 2: [1], 3: [1], 4: [2], 5: [3]}
    shares = {2: 0.7, 3: 0.3}

    def build(ws, rate):
        ws = pl.Path(ws)
        if ws.exists():
            shutil.rmtree(ws)
        sim = flopy.mf6.MFSimulation(
            sim_name="sf", sim_ws=str(ws), exe_name=str(mf6_bin)
        )
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
            gwf, stress_period_data=[[WELL_CELL, rate]], pname="wel-1"
        )

        packagedata, connectiondata = [], []
        for n, (i, j) in enumerate(cells):
            nconn = len(upstream[n]) + len(downstream[n])
            packagedata.append(
                [
                    n,
                    (0, i, j),
                    DELRC,
                    5.0,
                    1.0e-5,
                    STRTOP - 0.05 * n,
                    1.0,
                    5.0,
                    0.3,
                    nconn,
                    shares.get(n, 1.0),
                    0,
                ]
            )
            conn = [n] + list(upstream[n]) + [-d for d in downstream[n]]
            connectiondata.append(conn)
        flopy.mf6.ModflowGwfsfr(
            gwf,
            nreaches=len(cells),
            packagedata=packagedata,
            connectiondata=connectiondata,
            perioddata={0: [[0, "inflow", 5000.0]]},
            unit_conversion=128390.0,
            pname="sfr-1",
        )
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord="sf.hds", saverecord=[("HEAD", "ALL")]
        )
        sim.write_simulation(silent=True)
        success, buff = sim.run_simulation(silent=True)
        assert success, "\n".join(buff[-15:])

        path = pl.Path(ws) / "test.adj"
        with open(path, "w") as f:
            f.write("begin performance_measure pm\n")
            # one branch only: summing both would hide the split, since the
            # total leaving the junction is the same however it is shared
            for n in (2, 4):
                i, j = cells[n]
                f.write(f"1 1 1 {i + 1} {j + 1} sfr-1 direct 1.0 -1.0e+30\n")
            f.write("end performance_measure\n")
        _solve(ws)
        return ws

    ws_base = build(tmp_path / "base", WELL_RATE)
    ws_pert = build(tmp_path / "pert", WELL_RATE + dq)

    # the branches should carry the shares they were given
    path = sorted(glob.glob(str(ws_base / "sf_*.hd5")))[-1]
    with h5py.File(path, "r") as hf:
        key = [k for k in hf if k.startswith("solution_")][-1]
        fraction = hf[key]["sfr-1"]["reach_fraction"][:]
        idir = hf[key]["sfr-1"]["reach_idir"][:]
    split = sorted(round(float(f), 4) for f in fraction[idir > 0])
    assert split == [0.3, 0.7, 1.0, 1.0, 1.0], (
        f"the branches should take 0.3 and 0.7, got {split}"
    )

    def branch_leakage(ws):
        """Return the exchange of the measured branch only."""
        path = sorted(glob.glob(str(pl.Path(ws) / "sf_*.hd5")))[-1]
        with h5py.File(path, "r") as hf:
            key = [k for k in hf if k.startswith("solution_")][-1]
            simvals = hf[key]["sfr-1"]["simvals"][:]
        return float(simvals[2] + simvals[4])

    finite_difference = (branch_leakage(ws_pert) - branch_leakage(ws_base)) / dq
    with h5py.File(ws_base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])
    assert np.isclose(adjoint, finite_difference, rtol=2e-2), (
        f"adjoint {adjoint:.6e} against finite difference {finite_difference:.6e}"
    )


def _section_terms(ws):
    """Return what the forward model holds for a reach with a cross section."""
    mf6 = modflowapi.ModflowApi(str(lib_name), working_directory=str(ws))
    mf6.initialize()
    mf6.update()

    def value(name):
        return mf6.get_value(mf6.get_var_address(name, "SF", "SFR-1")).copy()

    terms = {
        name: value(name)
        for name in (
            "DEPTH",
            "USFLOW",
            "DSFLOW",
            "STATION",
            "XSHEIGHT",
            "XSROUGH",
            "ROUGH",
            "SLOPE",
        )
    }
    terms["IACROSS"] = value("IACROSS") - 1
    terms["UNITCONV"] = float(value("UNITCONV")[0])
    mf6.finalize()
    return terms


@pytest.mark.parametrize("section", [None, WALLED_SECTION])
def test_xs_rating(tmp_path, section):
    """The rating reproduces the discharge the reach depth was solved against."""
    ws = _build_model(tmp_path / "xs", WELL_RATE, cross_section=True, section=section)
    terms = _section_terms(ws)
    for n in range(1, NCOL):
        # the first reach is left out because its inflow is specified rather
        # than routed, so the flow entering it is not reported
        i0, i1 = int(terms["IACROSS"][n]), int(terms["IACROSS"][n + 1])
        discharge, _ = mannings_section(
            terms["STATION"][i0:i1],
            terms["XSHEIGHT"][i0:i1],
            terms["XSROUGH"][i0:i1],
            terms["ROUGH"][n],
            terms["UNITCONV"],
            terms["SLOPE"][n],
            terms["DEPTH"][n],
        )
        # MODFLOW sets the depth so the discharge is the mean of the flow in
        # and the flow out
        routed = 0.5 * (terms["USFLOW"][n] + terms["DSFLOW"][n])
        assert abs(discharge - routed) < 1.0e-6 * abs(routed), (
            f"reach {n} rating {discharge} against a routed flow of {routed}"
        )


@pytest.mark.parametrize("depth", [0.05, 0.3, 0.7, 0.999, 1.001, 1.5, 2.5])
def test_xs_derivative(depth):
    """The derivative of the rating matches a central difference."""
    # a flat bed between sloping banks that end in a vertical wall at a height
    # of one, so the section breaks at that depth
    station = np.array([0.0, 0.0, 1.0, 4.0, 5.0, 5.0])
    heights = np.array([3.0, 1.0, 0.0, 0.0, 1.0, 3.0])
    roughfracs = np.array([1.0, 0.8, 1.0, 1.0, 1.2, 1.0])

    def rating(d):
        return mannings_section(station, heights, roughfracs, 0.03, 1.0, 1e-3, d)

    _, derivative = rating(depth)
    step = 1.0e-7
    central = (rating(depth + step)[0] - rating(depth - step)[0]) / (2.0 * step)
    assert abs(derivative - central) < 1.0e-6 * abs(central), (
        f"depth {depth} derivative {derivative} against {central}"
    )


def test_sfr_cross_section(tmp_path):
    """A reach with a cross section carries the rating and the conductance."""
    adjoint, finite_difference = _compare(tmp_path, cross_section=True)
    # the conductance follows the wetted perimeter, so both it and the rating
    # move with the depth
    assert abs(adjoint - finite_difference) < 1.0e-3 * abs(finite_difference), (
        f"adjoint {adjoint} against a finite difference of {finite_difference}"
    )


def test_xs_negative_perimeter():
    """A section left with a negative perimeter still follows the depth.

    MODFLOW subtracts the height of a vertical face standing above the water
    surface, which can leave the wetted perimeter of a section negative. The
    conductance is then negative rather than absent, and the leakage still
    follows the depth through it.
    """
    # a tall wall over a short bank, so the face the water does not reach is
    # longer than everything it does
    station = np.array([0.0, 0.0, 0.1, 0.2])
    heights = np.array([10.0, 5.0, 0.0, 5.0])
    perimeter, dperimeter = wetted_perimeter(station, heights, 1.0)
    assert perimeter < 0.0, f"expected a negative perimeter, got {perimeter}"

    # a reach losing water, so the head difference across the streambed is
    # positive whatever the sign of the conductance
    cond = np.array([5.0 * 100.0 * perimeter / 1.0])
    head_difference = 0.5
    ratio = leakage_ratio(
        np.array([perimeter]),
        np.array([dperimeter]),
        cond,
        cond * head_difference,
    )
    expected = 1.0 + dperimeter / perimeter * head_difference
    assert ratio[0] == pytest.approx(expected)
    # the conductance alone would be the answer only if the section were flat
    assert ratio[0] != pytest.approx(1.0)


@pytest.mark.parametrize(
    "available, requested, taken, expected",
    [
        # FRACTION takes a share of whatever it is given
        (4000.0, 0.25, 1000.0, 0.25),
        # EXCESS passes the first `requested` on and takes the rest
        (4000.0, 3000.0, 1000.0, 1.0),
        (2000.0, 3000.0, 0.0, 0.0),
        # UPTO takes what it asks for until there is less than that
        (4000.0, 500.0, 500.0, 0.0),
        (300.0, 500.0, 300.0, 1.0),
        # THRESHOLD takes all or nothing, and moves with neither
        (4000.0, 1000.0, 1000.0, 0.0),
        (800.0, 1000.0, 0.0, 0.0),
    ],
)
def test_diversion_rule(available, requested, taken, expected):
    """The flows determine how a diverted flow follows the flow available."""
    derivative, determined = diversion_derivative(available, requested, taken)
    assert determined
    assert derivative == pytest.approx(expected)


@pytest.mark.parametrize(
    "available, requested, taken",
    [
        # EXCESS passes on the rate and takes the rest, so where the flow is
        # exactly twice the rate it takes the rate itself, as UPTO and
        # THRESHOLD do, and the three do not agree on how that would change
        (2000.0, 1000.0, 1000.0),
        # FRACTION takes the rate itself where the flow available is one
        (1.0, 0.4, 0.4),
    ],
)
def test_diversion_unknown(available, requested, taken):
    """Flows that leave the rule open are reported rather than guessed at."""
    derivative, determined = diversion_derivative(available, requested, taken)
    assert not determined
    assert derivative == 0.0


@pytest.mark.parametrize(
    "rule, rate",
    [
        ("FRACTION", 0.25),
        ("UPTO", 9000.0),
        ("EXCESS", 3000.0),
        ("THRESHOLD", 1000.0),
    ],
)
def test_sfr_diversion(tmp_path, rule, rate):
    """A diversion scales the flow the reaches below it are routed."""
    adjoint, finite_difference = _compare(tmp_path, diversion=(rule, rate))
    assert abs(adjoint - finite_difference) < 5.0e-5 * abs(finite_difference), (
        f"{rule} adjoint {adjoint} against a finite difference of {finite_difference}"
    )


def test_diversion_warned():
    """An undetermined rule is reported on the step it arises, and named once."""

    class Recorder:
        def __init__(self):
            self.messages = []

        def warning(self, message):
            self.messages.append(message)

    log = Recorder()
    coupling = mf6adj.advanced_packages.sfr.SfrCoupling(logger=log)
    clean = {
        "reach_flow_limited": np.zeros(9, dtype=int),
        "reach_undetermined_diversion": np.zeros(9, dtype=int),
    }
    tied = dict(clean)
    tied["reach_undetermined_diversion"] = np.array(
        [0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=int
    )

    # the sweep meets a step with nothing to report before the one that has it
    coupling._warn_once("sfr-1", clean)
    assert not log.messages

    coupling._warn_once("sfr-1", tied)
    assert len(log.messages) == 1
    assert "reach 4" in log.messages[0]

    # and the same condition is not reported again on later steps
    coupling._warn_once("sfr-1", tied)
    assert len(log.messages) == 1
