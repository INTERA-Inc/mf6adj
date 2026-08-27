"""
Tests for the auxiliary multiplier of the list-based stress packages.

A package given AUXMULTNAME scales the values it is asked to apply by an
auxiliary variable, and MODFLOW 6 applies that where it forms its terms rather
than folding it into the values it keeps. A sensitivity with respect to a value
the user gave therefore carries the multiplier. For a head-dependent boundary
the multiplier scales the conductance, so it is carried by the sensitivity to
the boundary head as well, which is the conductance itself.

Cases:
  - well_rate      : the rate a well is given.
  - well_zero_rate : a well given a rate of zero, whose flow says nothing about
                     the multiplier.
  - ghb_cond       : the conductance of a general-head boundary.
  - ghb_bhead      : the head of a general-head boundary.
  - riv_cond       : the conductance of a river.
  - drn_cond       : the conductance of a drain, which is active here.
  - two_wells      : two wells in one cell, given different multipliers, share
                     the one value the field carries for that cell.
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

NAME = "ax"
NROW = NCOL = 7
DELRC = 100.0
OBS_CELL = (0, 3, 3)
WELL_CELL = (0, 5, 5)
WELL_RATE = -500.0
GHB_HEAD, GHB_COND = 1.0, 100.0
RIV_STAGE, RIV_COND, RIV_BOT = 2.0, 50.0, -1.0
DRN_ELEV, DRN_COND = -2.0, 40.0


def _build_model(
    ws,
    auxmult=None,
    well_rate=WELL_RATE,
    ghb_cond=GHB_COND,
    ghb_head=GHB_HEAD,
    riv_cond=RIV_COND,
    drn_cond=DRN_COND,
):
    """Build a model carrying a well, a general-head boundary, a river, a drain."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    flopy.mf6.ModflowIms(
        sim, complexity="simple", outer_dvclose=1e-11, inner_dvclose=1e-12
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
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
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)

    aux = {} if auxmult is None else {"auxiliary": ["mult"], "auxmultname": "mult"}

    def row(cells):
        return [c + ([auxmult] if auxmult is not None else []) for c in cells]

    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=row([[(0, i, 0), ghb_head, ghb_cond] for i in range(NROW)]),
        pname="ghb-1",
        **aux,
    )
    flopy.mf6.ModflowGwfriv(
        gwf,
        stress_period_data=row(
            [[(0, i, NCOL - 1), RIV_STAGE, riv_cond, RIV_BOT] for i in range(NROW)]
        ),
        pname="riv-1",
        **aux,
    )
    flopy.mf6.ModflowGwfdrn(
        gwf,
        stress_period_data=row([[(0, 0, j), DRN_ELEV, drn_cond] for j in range(NCOL)]),
        pname="drn-1",
        **aux,
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        stress_period_data=row([[WELL_CELL, well_rate]]),
        pname="wel-1",
        **aux,
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-15:])
    return ws


def _head(ws):
    return float(flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds").get_data()[OBS_CELL])


def _solve_adjoint(ws, key):
    ws = pl.Path(ws)
    k, i, j = OBS_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / "adjoint_solution_obs.hd5", "r") as hf:
        return np.array(hf["composite"][key][:])


def _compare(tmpdir, auxmult, key, perturbed, step, node=None):
    """Return the adjoint sensitivity and a finite difference in one value."""
    base_ws = _build_model(tmpdir / "base", auxmult=auxmult)
    pert_ws = _build_model(tmpdir / "pert", auxmult=auxmult, **{perturbed: step})
    finite_difference = (_head(pert_ws) - _head(base_ws)) / (
        step - _DEFAULTS[perturbed]
    )
    sens = _solve_adjoint(base_ws, key)
    adjoint = float(sens[node]) if node is not None else float(np.sum(sens))
    return adjoint, finite_difference


_DEFAULTS = {
    "well_rate": WELL_RATE,
    "ghb_cond": GHB_COND,
    "ghb_head": GHB_HEAD,
    "riv_cond": RIV_COND,
    "drn_cond": DRN_COND,
}


@pytest.mark.parametrize("auxmult", [None, 0.4, 2.5])
def test_well_rate(function_tmpdir, auxmult):
    """The rate a well is given is scaled by the multiplier."""
    adjoint, finite_difference = _compare(
        function_tmpdir,
        auxmult,
        "wel6_q",
        "well_rate",
        WELL_RATE + 1.0,
        node=WELL_CELL,
    )
    assert np.isclose(adjoint, finite_difference, rtol=1e-3), (
        f"multiplier {auxmult}: adjoint {adjoint:.6e} against a finite "
        f"difference of {finite_difference:.6e}"
    )


@pytest.mark.parametrize(
    "auxmult, key, perturbed, step",
    [
        (0.4, "ghb-1_cond", "ghb_cond", GHB_COND + 1.0e-2),
        (2.5, "ghb-1_cond", "ghb_cond", GHB_COND + 1.0e-2),
        (0.4, "ghb-1_bhead", "ghb_head", GHB_HEAD + 1.0e-3),
        (0.4, "riv-1_cond", "riv_cond", RIV_COND + 1.0e-2),
        (0.4, "drn-1_cond", "drn_cond", DRN_COND + 1.0e-2),
    ],
)
def test_head_dependent_auxmult(function_tmpdir, auxmult, key, perturbed, step):
    """A head-dependent boundary carries the multiplier in both derivatives."""
    adjoint, finite_difference = _compare(
        function_tmpdir, auxmult, key, perturbed, step
    )
    assert np.isclose(adjoint, finite_difference, rtol=2e-3), (
        f"{key} with a multiplier of {auxmult}: adjoint {adjoint:.6e} against "
        f"a finite difference of {finite_difference:.6e}"
    )


def test_well_zero_rate(function_tmpdir):
    """A well given a rate of zero still carries the multiplier."""
    auxmult, dq = 0.4, 1.0
    base_ws = _build_model(function_tmpdir / "base", auxmult=auxmult, well_rate=0.0)
    pert_ws = _build_model(function_tmpdir / "pert", auxmult=auxmult, well_rate=dq)
    finite_difference = (_head(pert_ws) - _head(base_ws)) / dq

    sens = _solve_adjoint(base_ws, "wel6_q")
    adjoint = float(sens[WELL_CELL])

    assert np.isclose(adjoint, finite_difference, rtol=1e-3), (
        f"at a well given a rate of zero the adjoint {adjoint:.6e} does not "
        f"match the finite-difference derivative {finite_difference:.6e}"
    )


def test_two_wells_in_one_cell(function_tmpdir):
    """Two wells in one cell share the one value the field carries for it.

    The flow the cell produces is summed and divided by the rate it was given,
    so the factor is the response to changing the rate of the cell as a whole,
    shared out as the two rates already are.
    """
    first, second = -300.0, -700.0
    mult_first, mult_second = 0.4, 0.9

    def build(ws, scale=1.0):
        ws = pl.Path(ws)
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir(parents=True)
        sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
        flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
        flopy.mf6.ModflowIms(
            sim, complexity="simple", outer_dvclose=1e-11, inner_dvclose=1e-12
        )
        gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
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
        flopy.mf6.ModflowGwfic(gwf, strt=0.0)
        flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
        flopy.mf6.ModflowGwfghb(
            gwf,
            stress_period_data=[[(0, i, 0), GHB_HEAD, GHB_COND] for i in range(NROW)],
            pname="ghb-1",
        )
        # both wells sit in the same cell, with a multiplier of their own
        flopy.mf6.ModflowGwfwel(
            gwf,
            stress_period_data=[
                [WELL_CELL, first * scale, mult_first],
                [WELL_CELL, second * scale, mult_second],
            ],
            auxiliary=["mult"],
            auxmultname="mult",
            pname="wel-1",
        )
        flopy.mf6.ModflowGwfoc(
            gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
        )
        sim.write_simulation(silent=True)
        success, buff = sim.run_simulation(silent=True)
        assert success, "\n".join(buff[-15:])
        return ws

    # scaling both rates together is the perturbation the shared value answers
    step = 1.0e-3
    base_ws = build(function_tmpdir / "base")
    pert_ws = build(function_tmpdir / "pert", scale=1.0 + step)
    finite_difference = (_head(pert_ws) - _head(base_ws)) / (step * (first + second))

    sens = _solve_adjoint(base_ws, "wel6_q")
    adjoint = float(sens[WELL_CELL])

    assert np.isclose(adjoint, finite_difference, rtol=1e-3), (
        f"two wells in one cell: adjoint {adjoint:.6e} against a finite "
        f"difference of {finite_difference:.6e}"
    )
