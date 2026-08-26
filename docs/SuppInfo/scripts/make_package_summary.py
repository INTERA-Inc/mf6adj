"""Recompute the lake and streamflow archives the document plots.

Builds the models the mf6adj tests use, so the figures show the same lake and
the same stream the package behavior is tested on, and records the capture each
reports alongside a finite-difference total derivative at the pumped cell.

    python make_package_summary.py <archive directory>
"""

import pathlib as pl
import sys

import flopy
import h5py
import numpy as np

ROOT = pl.Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "autotest"))

import test_lak
import test_sfr

OUT = pl.Path(sys.argv[1])
WORK = OUT.parent / "package_runs"
OUT.mkdir(parents=True, exist_ok=True)


def lake_case():
    """Lake capture on the model the lake tests use."""
    dq = -5.0  # small enough to stay in the linear range

    def exchange(rate, tag):
        ws = WORK / f"lak_{tag}"
        sim, _ = test_lak._build_model(
            ws, boundary="lak", constant_stage=False, well_rate=rate
        )
        sim.run_simulation(silent=True)
        budget = flopy.utils.CellBudgetFile(str(ws / "lk.cbc"))
        records = budget.get_data(text="LAK", kstpkper=(0, 0))
        return float(np.sum(records[0]["q"])), ws

    base, ws = exchange(test_lak.WELL_RATE, "base")
    perturbed, _ = exchange(test_lak.WELL_RATE + dq, "pert")
    finite_difference = (perturbed - base) / dq

    cells = [(1, i, j) for _, i, j in test_lak._lake_cells()]
    test_lak._solve(ws, test_lak._write_adj(ws, cells, "lak-1", name="pm"))
    with h5py.File(ws / "adjoint_solution_pm.hd5", "r") as hf:
        capture = -1.0 * hf["composite"]["wel6_q"][:]
    return {
        "capture": capture,
        "boundary_cells": np.array([(k, i, j) for k, i, j in cells], dtype=int),
        "well_cell": np.array(test_lak.WELL_CELL, dtype=int),
        "adjoint": np.array([-float(capture.reshape(SHAPE_LAK)[test_lak.WELL_CELL])]),
        "finite_difference": np.array([finite_difference]),
        "shape": np.array(SHAPE_LAK),
        "delrc": np.array([test_lak.DELRC]),
    }


def stream_case():
    """Streamflow capture on the model the streamflow tests use."""
    # the tests already pair the adjoint with its finite-difference counterpart
    adjoint, finite_difference = test_sfr._compare(WORK / "sfr")
    ws = WORK / "sfr" / "base"
    with h5py.File(ws / "adjoint_solution_pm.hd5", "r") as hf:
        capture = -1.0 * hf["composite"]["wel6_q"][:]
    shape = (1, test_sfr.NROW, test_sfr.NCOL)
    cells = [(0, test_sfr.REACH_ROW, n) for n in range(test_sfr.NCOL)]
    return {
        "capture": capture,
        "boundary_cells": np.array(cells, dtype=int),
        "well_cell": np.array(test_sfr.WELL_CELL, dtype=int),
        "adjoint": np.array([adjoint]),
        "finite_difference": np.array([finite_difference]),
        "shape": np.array(shape),
        "delrc": np.array([test_sfr.DELRC]),
    }


SHAPE_LAK = (test_lak.NLAY, test_lak.NROW, test_lak.NCOL)

for name, case in (("lake", lake_case), ("stream", stream_case)):
    data = case()
    path = OUT / f"{name}-capture.npz"
    np.savez_compressed(path, **data)
    print(
        f"{name}: adjoint {float(data['adjoint'][0]):.6e} against a "
        f"finite difference of {float(data['finite_difference'][0]):.6e}"
        f"  -> {path.name}",
        flush=True,
    )
