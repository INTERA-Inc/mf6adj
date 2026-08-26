"""
Tests the adjoint solution against the Theis analytical solution.

A confined aquifer of uniform transmissivity and storativity is pumped by a
single well, which is the setting the Theis solution describes. The sensitivity
of head to a pumping rate is the unit response of that aquifer, so the drawdown
follows from the sensitivities and the rates and can be compared with the
analytical drawdown. The comparison uses no finite differences.

The model is finite where the Theis solution is not, so the drawdown is compared
only at distances the constant head does not yet reach.

Cases:
  - test_theis_superposition : drawdown rebuilt from the sensitivities matches
                               the analytical drawdown at four distances.
  - test_theis_reciprocity   : the sensitivity of head at the well to a rate at
                               an observation point equals the sensitivity of
                               head at that point to a rate at the well.
"""

import pathlib as pl
import shutil
import sys

import flopy
import h5py
import numpy as np
from scipy.special import exp1

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NAME = "theis"
DX = 500.0  # cell size (m)
HALF = 10000.0  # half the domain width (m)
THICK = 50.0  # aquifer thickness (m)
K = 10.0  # hydraulic conductivity (m/d)
SS = 4.0e-5  # specific storage (1/m)
NPER = 5  # stress periods
PERLEN = 20.0  # stress period length (d)
NSTP = 4  # time steps per period
RATE = -2000.0  # well rate (m3/d)

T = K * THICK  # transmissivity (m2/d)
S = SS * THICK  # storativity (dimensionless)
NCELL = int(2 * HALF / DX)
CENTER = NCELL // 2
# Distances at which the analytical solution is checked, in cells. The Theis
# solution is for an aquifer of infinite extent, and the constant head around
# this one suppresses the drawdown once the cone of depression reaches it, so
# the check is kept well inside the radius of influence.
OFFSETS = (1, 2, 3, 4)


def _cell(offset):
    """Return the row and column of a cell offset from the well."""
    return CENTER, CENTER + offset


def _build_model(ws):
    """A confined aquifer that meets the assumptions of the Theis solution."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(
        sim,
        nper=NPER,
        perioddata=[(PERLEN, NSTP, 1.0)] * NPER,
        time_units="days",
    )
    flopy.mf6.ModflowIms(
        sim,
        complexity="simple",
        outer_dvclose=1.0e-9,
        inner_dvclose=1.0e-10,
        linear_acceleration="bicgstab",
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=NCELL,
        ncol=NCELL,
        delr=DX,
        delc=DX,
        top=0.0,
        botm=-THICK,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    # confined, so transmissivity does not follow the head
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=K)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=SS, sy=0.0, transient={0: True})
    # a constant head around the edge, far enough that the drawdown there is
    # negligible; the corners belong to two sides, so collect the cells once
    edge = set()
    for i in range(NCELL):
        edge.update({(0, i, 0), (0, i, NCELL - 1)})
    for j in range(NCELL):
        edge.update({(0, 0, j), (0, NCELL - 1, j)})
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=[[cell, 0.0] for cell in sorted(edge)],
        pname="chd-1",
    )
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=[[(0, *_cell(0)), RATE]], pname="wel-1"
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{NAME}.hds",
        saverecord=[("HEAD", "ALL")],
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(str(line) for line in buff[-25:])
    return ws


def _solve_adjoint(ws, measure_cell):
    """Head at one cell at the end of the run, as a direct measure."""
    ws = pl.Path(ws)
    row, col = measure_cell
    with open(ws / "theis.adj", "w") as f:
        f.write("begin performance_measure pm\n")
        f.write(f"  {NPER} {NSTP} 1 {row + 1} {col + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "theis.adj", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="forward.hd5")
    adj.solve_adjoint()
    adj.finalize()

    # the rate acts at every time step of a period, so the sensitivity to a
    # rate held over that period is the sum over its time steps
    with h5py.File(ws / "adjoint_solution_pm.hd5", "r") as hf:
        per_period = {}
        for key in hf:
            if not key.startswith("solution_"):
                continue
            kper = int(key.split("kper:")[1].split("_")[0])
            values = hf[key]["wel6_q"][:]
            if kper in per_period:
                per_period[kper] = per_period[kper] + values
            else:
                per_period[kper] = values
    return per_period


def _analytical(offset, elapsed):
    """Theis drawdown at a distance from the well after an elapsed time."""
    radius = offset * DX
    u = radius**2 * S / (4.0 * T * elapsed)
    return -RATE * exp1(u) / (4.0 * np.pi * T)


def test_theis_superposition(function_tmpdir):
    """Drawdown rebuilt from the sensitivities matches the Theis solution."""
    ws = _build_model(function_tmpdir / "theis")
    # measure at the well, so one solve gives the response at every cell
    per_period = _solve_adjoint(ws, _cell(0))

    elapsed = NPER * PERLEN
    for offset in OFFSETS:
        row, col = _cell(offset)
        # the well pumps at the same rate in every period
        rebuilt = (
            -sum(float(values[(0, row, col)]) for values in per_period.values()) * RATE
        )
        expected = _analytical(offset, elapsed)
        assert np.isclose(rebuilt, expected, rtol=3.0e-2), (
            f"at {offset * DX:.0f} m the rebuilt drawdown {rebuilt:.4f} m does "
            f"not match the analytical drawdown {expected:.4f} m"
        )


def test_theis_reciprocity(function_tmpdir):
    """Swapping the pumped cell and the measured cell leaves the response."""
    offset = OFFSETS[1]
    ws = _build_model(function_tmpdir / "reciprocity")

    at_well = _solve_adjoint(ws, _cell(0))
    row, col = _cell(offset)
    forward = sum(float(v[(0, row, col)]) for v in at_well.values())

    at_observation = _solve_adjoint(ws, _cell(offset))
    row, col = _cell(0)
    reverse = sum(float(v[(0, row, col)]) for v in at_observation.values())

    assert np.isclose(forward, reverse, rtol=1.0e-6), (
        f"the response is not symmetric: {forward:.6e} pumping at the well "
        f"against {reverse:.6e} pumping at the observation point"
    )
