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
  - lak_total_deriv : a free lake stage makes the adjoint disagree with a
                      finite-difference total derivative (xfail).
"""

import pathlib as pl
import shutil
import sys

import flopy
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


def _write_adj(ws, cells, pm_type, name="pm"):
    """Write a direct performance measure over the given zero-based cells."""
    path = pl.Path(ws) / "test.adj"
    with open(path, "w") as f:
        f.write(f"begin performance_measure {name}\n")
        for k, i, j in cells:
            f.write(f"1 1 {k + 1} {i + 1} {j + 1} {pm_type} direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return path


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


@pytest.mark.xfail(
    reason="the adjoint holds the lake stage fixed, so it returns a partial "
    "derivative rather than the total derivative (INTERA-Inc/mf6adj#78)",
    strict=False,
)
def test_lak_total_derivative(function_tmpdir):
    """The lake sensitivity should match a finite-difference total derivative.

    A lake whose stage is free to move responds to pumping twice: the heads
    beneath it fall, and the stage falls with them. The adjoint captures only
    the first, so this comparison fails until the lake equation is solved with
    the flow equations.
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
    df = _solve(ws_base, _write_adj(ws_base, cells, "lak-1", name="pm"))["pm"]
    node = np.ravel_multi_index(WELL_CELL, (NLAY, NROW, NCOL))
    adjoint = df["wel6_q"].values[node]

    assert np.isclose(adjoint, finite_difference, rtol=1e-2), (
        f"adjoint {adjoint:.6e} does not match the finite-difference total "
        f"derivative {finite_difference:.6e}"
    )
