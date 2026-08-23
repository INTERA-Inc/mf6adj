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
