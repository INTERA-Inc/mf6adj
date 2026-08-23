"""
Tests for lake (LAK) performance measures and the solution-coupling check.

Builds a small two-layer model with an embedded lake, so the lake sensitivities
can be compared against an equivalent GHB representation of the same boundary.

Cases:
  - lak_perfmeas    : a lak6 performance measure solves and returns stage and
                      conductance sensitivities.
  - lak_matches_ghb : lake sensitivities match a GHB boundary with the same
                      stage and conductance.
  - duplicate_bnd   : two boundaries in one cell accumulate rather than
                      overwrite.
  - maw_rejected    : a model with maw6 is rejected because it adds equations
                      to the solution matrix.
  - lak_total_deriv : a free lake stage reproduces a finite-difference total
                      derivative.
  - two_lakes       : each lake gets its own connections and surface area, and
                      both reproduce a finite-difference total derivative.
  - stage_switches  : a lake that is free in one period and constant in another
                      does not carry lake state between time steps.
  - multi_period    : a lake free through several periods reproduces a
                      finite-difference total derivative, exercising the lake
                      storage carried backward in time.
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

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NLAY, NROW, NCOL = 2, 8, 8
DELRC = 100.0
LAKE_ROWS = (2, 3)
LAKE_COLS = (2, 3)
LAKE_STAGE = 9.0
STRT = 12.0
BEDLEAK = 0.1
WELL_CELL = (1, 6, 6)
WELL_RATE = -500.0


def _lake_cells():
    """Return the layer-1 cells the lake occupies."""
    cells = []
    for i in LAKE_ROWS:
        for j in LAKE_COLS:
            cells.append((0, i, j))
    return cells


def _lak_conductances(ws):
    """Return the connection conductances MODFLOW computed for the lake."""
    mf6 = modflowapi.ModflowApi(str(lib_name), working_directory=str(ws))
    mf6.initialize()
    mf6.update()
    bound = mf6.get_value(mf6.get_var_address("BOUND", "LK", "LAK-1"))
    mf6.finalize()
    return bound[:, 1].copy()


def _build_model(
    ws, boundary="lak", maw=False, lake_cond=None, constant_stage=True, well_rate=None
):
    """Build a two-layer model whose lake is either LAK, GHB, or absent."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    sim = flopy.mf6.MFSimulation(sim_name="lk", sim_ws=str(ws), exe_name=str(mf6_bin))
    # a free lake stage needs a transient run: the stage responds to what the
    # lake gains or loses over the time step
    steady = constant_stage
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0 if steady else 100.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-9, inner_dvclose=1e-10)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="lk", save_flows=True)

    # the lake replaces the top-layer cells it occupies; the GHB variant keeps
    # the same active grid so the two models are otherwise identical
    idomain = np.ones((NLAY, NROW, NCOL), dtype=int)
    for k, i, j in _lake_cells():
        idomain[k, i, j] = 0
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELRC,
        delc=DELRC,
        top=10.0,
        botm=[0.0, -10.0],
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=STRT)
    # confined, so the flow equations are linear and the lake and GHB
    # representations can be compared term by term
    flopy.mf6.ModflowGwfnpf(
        gwf, k=10.0, k33=1.0, icelltype=0, save_specific_discharge=True
    )
    if steady:
        flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, sy=0.2, steady_state={0: True})
    else:
        flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, sy=0.2, transient={0: True})

    # a gradient across the model so the lake exchanges water
    ghb_spd = []
    for k in range(NLAY):
        for i in range(NROW):
            ghb_spd.append([(k, i, 0), STRT + 0.5, 1000.0])
            ghb_spd.append([(k, i, NCOL - 1), STRT - 0.5, 1000.0])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd, pname="ghb-edge")

    rate = WELL_RATE if well_rate is None else well_rate
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[WELL_CELL, rate]], pname="wel-1")

    if boundary == "lak":
        # one vertical connection per lake cell, into the layer below
        connectiondata = []
        for iconn, (_, i, j) in enumerate(_lake_cells()):
            connectiondata.append(
                [0, iconn, (1, i, j), "vertical", BEDLEAK, 0.0, 0.0, 0.0, 0.0]
            )
        flopy.mf6.ModflowGwflak(
            gwf,
            nlakes=1,
            noutlets=0,
            ntables=0,
            packagedata=[[0, LAKE_STAGE, len(connectiondata)]],
            connectiondata=connectiondata,
            # a constant stage keeps the lake from adjusting to zero net
            # leakage in steady state, and makes it directly comparable to GHB
            perioddata={0: [[0, "status", "constant"]]} if constant_stage else None,
            pname="lak-1",
        )
    elif boundary == "ghb":
        # the same boundary written as GHB: bhead is the lake stage and the
        # conductance is what MODFLOW computed for each lake connection
        spd = [
            [(1, i, j), LAKE_STAGE, lake_cond[iconn]]
            for iconn, (_, i, j) in enumerate(_lake_cells())
        ]
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=spd, pname="ghb-lake")

    if maw:
        flopy.mf6.ModflowGwfmaw(
            gwf,
            nmawwells=1,
            packagedata=[[0, 0.5, -10.0, 10.0, "thiem", 1]],
            connectiondata=[[0, 0, (1, 5, 2), 10.0, -10.0, 0.0, 0.0]],
            perioddata={0: [[0, "rate", -100.0]]},
            pname="maw-1",
        )

    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="lk.hds",
        budget_filerecord="lk.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    return sim, gwf


def _write_adj(ws, cells, pm_type, name="pm", weight=1.0):
    """Write a direct performance measure over the given zero-based cells."""
    path = pl.Path(ws) / "test.adj"
    with open(path, "w") as f:
        f.write(f"begin performance_measure {name}\n")
        for k, i, j in cells:
            f.write(f"1 1 {k + 1} {i + 1} {j + 1} {pm_type} direct {weight} -1.0e+30\n")
        f.write("end performance_measure\n")
    return path


def _measure_value(ws, package):
    """Return the measure value mf6adj saw: the package flux it summed."""
    path = sorted(pl.Path(ws).glob("lk_*.hd5"))[-1]
    with h5py.File(path, "r") as hf:
        key = [k for k in hf if k.startswith("solution_")][-1]
        return float(np.sum(hf[key][package]["simvals"][:]))


def _solve(ws, adj_file):
    """Run the forward and adjoint solves and return the sensitivity frames."""
    adj = mf6adj.Mf6Adj(
        adj_file.name, str(lib_name), logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model()
    dfs = adj.solve_adjoint()
    adj.finalize()
    return dfs


def test_lak_perfmeas(function_tmpdir):
    ws = function_tmpdir / "lak"
    sim, _ = _build_model(ws, boundary="lak")
    sim.run_simulation(silent=True)

    cells = [(1, i, j) for _, i, j in _lake_cells()]
    dfs = _solve(ws, _write_adj(ws, cells, "lak-1", name="lakegw"))

    df = dfs["lakegw"]
    for column in ("lak-1_stage", "lak-1_cond", "k11", "wel6_q"):
        assert column in df.columns, f"'{column}' missing from {list(df.columns)}"

    # the lake exchange responds to the pumping well and to the conductivity
    assert np.all(np.isfinite(df["wel6_q"].values))
    assert np.abs(df["wel6_q"].values).max() > 0.0
    assert np.abs(df["lak-1_cond"].values).max() > 0.0


def test_lak_matches_ghb(function_tmpdir):
    """A lake and a GHB with the same stage and conductance give the same result."""
    ws_lak = function_tmpdir / "lak"
    sim_lak, _ = _build_model(ws_lak, boundary="lak")
    sim_lak.run_simulation(silent=True)

    ws_ghb = function_tmpdir / "ghb"
    sim_ghb, _ = _build_model(
        ws_ghb, boundary="ghb", lake_cond=_lak_conductances(ws_lak)
    )
    sim_ghb.run_simulation(silent=True)

    # the two models must reach the same heads before the sensitivities can be
    # compared; the lake cells themselves are inactive in the LAK model
    h_lak = flopy.utils.HeadFile(ws_lak / "lk.hds").get_data()
    h_ghb = flopy.utils.HeadFile(ws_ghb / "lk.hds").get_data()
    assert np.allclose(h_lak[1], h_ghb[1], atol=1e-6), (
        f"layer-2 heads differ by up to {np.abs(h_lak[1] - h_ghb[1]).max()}"
    )

    cells = [(1, i, j) for _, i, j in _lake_cells()]
    lak = _solve(ws_lak, _write_adj(ws_lak, cells, "lak-1", name="pm"))["pm"]
    ghb = _solve(ws_ghb, _write_adj(ws_ghb, cells, "ghb-lake", name="pm"))["pm"]

    for lak_col, ghb_col in (
        ("wel6_q", "wel6_q"),
        ("k11", "k11"),
        ("lak-1_stage", "ghb-lake_bhead"),
        ("lak-1_cond", "ghb-lake_cond"),
    ):
        assert np.allclose(lak[lak_col].values, ghb[ghb_col].values, atol=1e-8), (
            f"'{lak_col}' differs from '{ghb_col}' by up to "
            f"{np.abs(lak[lak_col].values - ghb[ghb_col].values).max()}"
        )


def test_duplicate_boundary_accumulates(function_tmpdir):
    """Two boundaries in one cell contribute as much as one of twice the size.

    Splitting a boundary into two halves in the same cell is physically the same
    boundary, so both models must return the same sensitivities.
    """
    cell = (1, 5, 3)
    cond = 400.0

    def build(ws, entries):
        sim, gwf = _build_model(ws, boundary="none")
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=entries, pname="ghb-dup")
        sim.write_simulation(silent=True)
        sim.run_simulation(silent=True)
        return _solve(ws, _write_adj(ws, [cell], "ghb-dup", name="pm"))["pm"]

    ws_split = function_tmpdir / "split"
    split = build(ws_split, [[cell, LAKE_STAGE, cond / 2.0]] * 2)

    ws_whole = function_tmpdir / "whole"
    whole = build(ws_whole, [[cell, LAKE_STAGE, cond]])

    # the split cell sees the same total conductance, so the head-side terms
    # match exactly
    for column in ("ghb-dup_bhead", "wel6_q"):
        assert np.allclose(split[column].values, whole[column].values, atol=1e-8), (
            f"'{column}' differs by up to "
            f"{np.abs(split[column].values - whole[column].values).max()}"
        )

    # the conductance derivative is per boundary, so perturbing each of two
    # halves moves the measure twice as much as perturbing the single boundary
    assert np.allclose(
        split["ghb-dup_cond"].values, 2.0 * whole["ghb-dup_cond"].values, atol=1e-8
    ), "the two boundaries in the shared cell did not both contribute"


def test_maw_rejected(function_tmpdir):
    """maw6 adds equations to the solution matrix and must be rejected."""
    ws = function_tmpdir / "maw"
    sim, _ = _build_model(ws, boundary="lak", maw=True)
    sim.run_simulation(silent=True)

    adj_file = _write_adj(ws, [(1, 4, 4)], "head", name="h")
    with pytest.raises(Exception, match="solution matrix"):
        mf6adj.Mf6Adj(
            adj_file.name,
            str(lib_name),
            logging_level="WARNING",
            working_directory=str(ws),
        )


def test_lak_total_derivative(function_tmpdir):
    """The lake sensitivity matches a finite-difference total derivative.

    A lake whose stage is free to move responds to pumping twice: the heads
    beneath it fall, and the stage falls with them. Solving the lake water
    balance with the flow equations captures both.
    """
    dq = -5.0  # small enough to stay in the linear range

    def exchange(rate):
        ws = function_tmpdir / f"q{abs(rate):.0f}"
        sim, _ = _build_model(ws, boundary="lak", constant_stage=False, well_rate=rate)
        sim.run_simulation(silent=True)
        budget = flopy.utils.CellBudgetFile(ws / "lk.cbc")
        records = budget.get_data(text="LAK", kstpkper=(0, 0))
        return float(np.sum(records[0]["q"])), ws

    base, ws_base = exchange(WELL_RATE)
    perturbed, _ = exchange(WELL_RATE + dq)
    finite_difference = (perturbed - base) / dq

    cells = [(1, i, j) for _, i, j in _lake_cells()]
    _solve(ws_base, _write_adj(ws_base, cells, "lak-1", name="pm"))
    # read the full-grid array: the lake removes cells from the top layer, so a
    # reduced node number is not the grid index
    with h5py.File(ws_base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])

    assert np.isclose(adjoint, finite_difference, rtol=1e-2), (
        f"adjoint {adjoint:.6e} does not match the finite-difference total "
        f"derivative {finite_difference:.6e}"
    )


def test_flux_measure_weight(function_tmpdir):
    """A weighted flux measure scales its sensitivities by that weight."""
    cells = [(1, i, j) for _, i, j in _lake_cells()]

    def solve(ws, weight):
        sim, _ = _build_model(ws, boundary="lak")
        sim.run_simulation(silent=True)
        return _solve(ws, _write_adj(ws, cells, "lak-1", weight=weight))["pm"]

    single = solve(function_tmpdir / "w1", 1.0)
    double = solve(function_tmpdir / "w2", 2.0)

    for column in ("wel6_q", "k11", "lak-1_stage", "lak-1_cond"):
        assert np.allclose(
            double[column].values, 2.0 * single[column].values, atol=1e-8
        ), (
            f"'{column}' is not twice the unweighted result; largest gap "
            f"{np.abs(double[column].values - 2.0 * single[column].values).max()}"
        )


def test_direct_effect_only_for_measured_package(function_tmpdir):
    """A package the measure does not name gets no direct contribution.

    The measure sums the lake flux, so its sensitivity to the edge GHB heads is
    only the adjoint response. Compare it with a finite difference that shifts
    every edge GHB head.
    """
    delta = 1.0e-3
    cells = [(1, i, j) for _, i, j in _lake_cells()]

    def run(ws, shift):
        sim, gwf = _build_model(ws, boundary="lak")
        edge = gwf.get_package("ghb-edge")
        spd = [list(record) for record in edge.stress_period_data.get_data(0)]
        edge.stress_period_data = {0: [[tuple(r[0]), r[1] + shift, r[2]] for r in spd]}
        sim.write_simulation(silent=True)
        sim.run_simulation(silent=True)
        df = _solve(ws, _write_adj(ws, cells, "lak-1"))["pm"]
        return df, _measure_value(ws, "lak-1")

    base, value = run(function_tmpdir / "base", 0.0)
    _, shifted = run(function_tmpdir / "shift", delta)

    finite_difference = (shifted - value) / delta
    adjoint = float(np.sum(base["ghb-edge_bhead"].values))

    assert np.isclose(adjoint, finite_difference, rtol=1e-3), (
        f"adjoint {adjoint:.6e} does not match the finite difference "
        f"{finite_difference:.6e}"
    )


def _build_two_lakes(ws, well_rate):
    """Build a model with two separate lakes, each of two cells."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    cells = {0: [(2, 2), (2, 3)], 1: [(5, 5), (5, 6)]}
    sim = flopy.mf6.MFSimulation(sim_name="lk", sim_ws=str(ws), exe_name=str(mf6_bin))
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0, 1, 1.0)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-11, inner_dvclose=1e-12)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="lk", save_flows=True)
    idomain = np.ones((NLAY, NROW, NCOL), dtype=int)
    for lake in cells.values():
        for i, j in lake:
            idomain[0, i, j] = 0
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELRC,
        delc=DELRC,
        top=10.0,
        botm=[0.0, -10.0],
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=9.5)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0, k33=1.0, icelltype=0)
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, sy=0.2, transient={0: True})
    spd = []
    for k in range(NLAY):
        for i in range(NROW):
            spd.append([(k, i, 0), 9.7, 1000.0])
            spd.append([(k, i, NCOL - 1), 9.3, 1000.0])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=spd, pname="ghb-edge")
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=[[WELL_CELL, well_rate]], pname="wel-1"
    )
    pkgdata, conn = [], []
    for ilak, lake in cells.items():
        pkgdata.append([ilak, 9.4, len(lake)])
        for n, (i, j) in enumerate(lake):
            conn.append([ilak, n, (1, i, j), "vertical", BEDLEAK, 0.0, 0.0, 0.0, 0.0])
    flopy.mf6.ModflowGwflak(
        gwf,
        nlakes=2,
        noutlets=0,
        ntables=0,
        packagedata=pkgdata,
        connectiondata=conn,
        pname="lak-1",
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="lk.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    sim.run_simulation(silent=True)
    return sim, cells


def test_two_lakes(function_tmpdir):
    """Two lakes each get their own connections, surface area, and sensitivity."""
    dq = -5.0
    cells = {0: [(2, 2), (2, 3)], 1: [(5, 5), (5, 6)]}
    measured = [(1, i, j) for lake in cells.values() for i, j in lake]

    def exchange(ws, rate):
        _build_two_lakes(ws, rate)
        _solve(ws, _write_adj(ws, measured, "lak-1", name="pm"))
        return _measure_value(ws, "lak-1")

    ws_base = function_tmpdir / "base"
    base = exchange(ws_base, WELL_RATE)
    pert = exchange(function_tmpdir / "pert", WELL_RATE + dq)
    finite_difference = (pert - base) / dq

    # each lake owns two connections of one cell each, so its surface area is
    # twice the cell area rather than the whole package's
    path = sorted(ws_base.glob("lk_*.hd5"))[-1]
    with h5py.File(path, "r") as hf:
        key = [k for k in hf if k.startswith("solution_")][-1]
        area = hf[key]["lak-1"]["lake_surface_area"][:]
        lake_of_conn = hf[key]["lak-1"]["lake_of_conn"][:]
    assert area.shape == (2,), f"expected one area per lake, got {area.shape}"
    assert np.allclose(area, 2.0 * DELRC * DELRC), (
        f"per-lake surface areas are wrong: {area}"
    )
    assert np.array_equal(lake_of_conn, [0, 0, 1, 1]), lake_of_conn

    with h5py.File(ws_base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])
    assert np.isclose(adjoint, finite_difference, rtol=1e-2), (
        f"adjoint {adjoint:.6e} does not match the finite-difference total "
        f"derivative {finite_difference:.6e}"
    )


def _build_switching(ws, well_rate):
    """Three periods with the lake free, then constant, then free again."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    sim = flopy.mf6.MFSimulation(sim_name="lk", sim_ws=str(ws), exe_name=str(mf6_bin))
    flopy.mf6.ModflowTdis(sim, nper=3, perioddata=[(100.0, 1, 1.0)] * 3)
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-11, inner_dvclose=1e-12)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="lk", save_flows=True)
    idomain = np.ones((NLAY, NROW, NCOL), dtype=int)
    for k, i, j in _lake_cells():
        idomain[k, i, j] = 0
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELRC,
        delc=DELRC,
        top=10.0,
        botm=[0.0, -10.0],
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=STRT)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0, k33=1.0, icelltype=0)
    flopy.mf6.ModflowGwfsto(gwf, ss=1e-5, sy=0.2, transient={0: True})
    spd = []
    for k in range(NLAY):
        for i in range(NROW):
            spd.append([(k, i, 0), STRT + 0.5, 1000.0])
            spd.append([(k, i, NCOL - 1), STRT - 0.5, 1000.0])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=spd, pname="ghb-edge")
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=[[WELL_CELL, well_rate]], pname="wel-1"
    )
    conn = [
        [0, n, (1, i, j), "vertical", BEDLEAK, 0.0, 0.0, 0.0, 0.0]
        for n, (_, i, j) in enumerate(_lake_cells())
    ]
    flopy.mf6.ModflowGwflak(
        gwf,
        nlakes=1,
        noutlets=0,
        ntables=0,
        packagedata=[[0, LAKE_STAGE, len(conn)]],
        connectiondata=conn,
        perioddata={
            0: [[0, "status", "active"]],
            1: [[0, "status", "constant"]],
            2: [[0, "status", "active"]],
        },
        pname="lak-1",
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="lk.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    sim.run_simulation(silent=True)
    return ws


def test_stage_switches_between_periods(function_tmpdir):
    """Lake state must not leak across a period where the stage is held fixed.

    The backward sweep meets the free periods either side of a constant one. If
    the constant period passes the later period's lake carry on to the earlier
    one, the earlier period's sensitivities are wrong.
    """
    dq = -5.0
    cells = [(1, i, j) for _, i, j in _lake_cells()]

    def measure(ws, rate):
        _build_switching(ws, rate)
        path = pl.Path(ws) / "test.adj"
        with open(path, "w") as f:
            f.write("begin performance_measure pm\n")
            for k, i, j in cells:
                f.write(f"3 1 {k + 1} {i + 1} {j + 1} lak-1 direct 1.0 -1.0e+30\n")
            f.write("end performance_measure\n")
        _solve(ws, path)
        return _measure_value(ws, "lak-1")

    ws_base = function_tmpdir / "base"
    base = measure(ws_base, WELL_RATE)
    pert = measure(function_tmpdir / "pert", WELL_RATE + dq)
    finite_difference = (pert - base) / dq

    with h5py.File(ws_base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])

    assert np.isclose(adjoint, finite_difference, rtol=1e-2), (
        f"adjoint {adjoint:.6e} does not match the finite-difference total "
        f"derivative {finite_difference:.6e}"
    )


def _build_multi_period(ws, well_rate, nper=10):
    """A transient model with the lake stage free in every period."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    sim = flopy.mf6.MFSimulation(sim_name="lk", sim_ws=str(ws), exe_name=str(mf6_bin))
    flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=[(50.0, 1, 1.0)] * nper)
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-11, inner_dvclose=1e-12)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="lk", save_flows=True)
    idomain = np.ones((NLAY, NROW, NCOL), dtype=int)
    for k, i, j in _lake_cells():
        idomain[k, i, j] = 0
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELRC,
        delc=DELRC,
        top=10.0,
        botm=[0.0, -10.0],
        idomain=idomain,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=STRT)
    flopy.mf6.ModflowGwfnpf(gwf, k=10.0, k33=1.0, icelltype=0)
    flopy.mf6.ModflowGwfsto(
        gwf, ss=1e-5, sy=0.2, transient={kper: True for kper in range(nper)}
    )
    spd = []
    for k in range(NLAY):
        for i in range(NROW):
            spd.append([(k, i, 0), STRT + 0.5, 1000.0])
            spd.append([(k, i, NCOL - 1), STRT - 0.5, 1000.0])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=spd, pname="ghb-edge")
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data={0: [[WELL_CELL, well_rate]]}, pname="wel-1"
    )
    conn = [
        [0, n, (1, i, j), "vertical", BEDLEAK, 0.0, 0.0, 0.0, 0.0]
        for n, (_, i, j) in enumerate(_lake_cells())
    ]
    flopy.mf6.ModflowGwflak(
        gwf,
        nlakes=1,
        noutlets=0,
        ntables=0,
        packagedata=[[0, LAKE_STAGE, len(conn)]],
        connectiondata=conn,
        pname="lak-1",
    )
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="lk.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    sim.run_simulation(silent=True)
    return ws


def test_multi_period_carry(function_tmpdir):
    """A lake free through several periods reproduces the total derivative.

    The measure is taken at the last period, so the lake storage has to be
    carried backward through all nine earlier periods. A single-period model
    never exercises that term, and a long chain compounds any error in it.
    """
    dq = -5.0
    nper = 10
    cells = [(1, i, j) for _, i, j in _lake_cells()]

    def measure(ws, rate):
        _build_multi_period(ws, rate, nper=nper)
        path = pl.Path(ws) / "test.adj"
        with open(path, "w") as f:
            f.write("begin performance_measure pm\n")
            for k, i, j in cells:
                f.write(f"{nper} 1 {k + 1} {i + 1} {j + 1} lak-1 direct 1.0 -1.0e+30\n")
            f.write("end performance_measure\n")
        _solve(ws, path)
        return _measure_value(ws, "lak-1")

    ws_base = function_tmpdir / "base"
    base = measure(ws_base, WELL_RATE)
    pert = measure(function_tmpdir / "pert", WELL_RATE + dq)
    finite_difference = (pert - base) / dq

    with h5py.File(ws_base / "adjoint_solution_pm.hd5", "r") as hf:
        adjoint = float(hf["composite"]["wel6_q"][WELL_CELL])

    assert np.isclose(adjoint, finite_difference, rtol=1e-2), (
        f"adjoint {adjoint:.6e} does not match the finite-difference total "
        f"derivative {finite_difference:.6e}"
    )
