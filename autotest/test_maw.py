"""
Tests for the multi-aquifer well terms of the adjoint solution.

A multi-aquifer well head is a dependent variable, and MODFLOW 6 solves it with
the flow equations rather than outside them, so the well's own equation is
already a row of the matrix the adjoint is taken from. What the adjoint forms
is the right-hand side of that row: the derivative of a performance measure
with respect to the well head, and the storage the well carries from one time
step to the next.

The exchange between a well and a cell follows the head in the cell twice over,
through the head difference and through the saturated fraction of the screen
that sets the conductance. A measure of that exchange has to carry both or it
is wrong by the second, which is the larger error where the screen is only
partly saturated.

Cases:
  - test_flux_measure            : a measure of the exchange at one cell
                                   matches a finite-difference derivative, for
                                   a well whose head is solved and one whose
                                   head is held, over one time step and four.
  - test_conductance_derivative  : dropping the conductance derivative moves
                                   the answer, so the test above is testing it.
  - test_head_measure            : a head measure in a model with a well
                                   matches a finite-difference derivative.
  - test_inactive_well           : an inactive well exchanges nothing and
                                   leaves the sensitivities alone.
  - test_two_packages            : two well packages keep their own rows.
  - test_measure_on_maw_accepted : a measure naming a maw6 package is accepted.
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

NAME = "maw"
NLAY, NROW, NCOL = 3, 7, 7
TOP, BOTM = 0.0, [-10.0, -60.0, -200.0]
MAW_CELL = (3, 3)
WELL_CELL = (0, 5, 5)
WELL_RATE = -50.0
GHB_HEAD = -4.0


def _build_model(
    ws,
    wel_rate=WELL_RATE,
    maw_rate=-300.0,
    status="ACTIVE",
    maw_head=-6.0,
    nstp=1,
    second_package=False,
):
    """An unconfined model pumped through a multi-aquifer well.

    The screen is only partly saturated in the upper layer, so the conductance
    of that connection follows the head.
    """
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(60.0, nstp, 1.0)])
    flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        outer_dvclose=1e-11,
        inner_dvclose=1e-12,
        outer_maximum=500,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=NAME, save_flows=True, newtonoptions="NEWTON"
    )
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=50.0,
        delc=50.0,
        top=TOP,
        botm=BOTM,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=GHB_HEAD)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=5.0, k33=0.5)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=1e-5, sy=0.15, transient={0: True})
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), GHB_HEAD, 500.0] for i in range(NROW)],
        pname="ghb-1",
    )
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=[[WELL_CELL, wel_rate]], pname="wel-1"
    )
    if status == "CONSTANT":
        period = [[0, "STATUS", "CONSTANT"], [0, "WELL_HEAD", maw_head]]
    elif status == "INACTIVE":
        period = [[0, "STATUS", "INACTIVE"]]
    else:
        period = [[0, "RATE", maw_rate]]
    i, j = MAW_CELL
    flopy.mf6.ModflowGwfmaw(
        gwf,
        nmawwells=1,
        packagedata=[[0, 0.5, BOTM[-1], TOP, "THIEM", NLAY]],
        connectiondata=[
            [0, k, (k, i, j), -999, -999, -999.0, -999.0] for k in range(NLAY)
        ],
        perioddata={0: period},
        pname="maw-1",
        save_flows=True,
    )
    if second_package:
        flopy.mf6.ModflowGwfmaw(
            gwf,
            nmawwells=1,
            packagedata=[[0, 0.5, BOTM[-1], TOP, "THIEM", 2]],
            connectiondata=[
                [0, k, (k, 1, 1), -999, -999, -999.0, -999.0] for k in range(2)
            ],
            perioddata={0: [[0, "RATE", -80.0]]},
            pname="maw-2",
            save_flows=True,
        )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{NAME}.hds",
        budget_filerecord=f"{NAME}.cbc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])
    return ws


def _connection_flux(ws, iconn=0):
    """The exchange between the well and one of the cells it connects to."""
    cbc = flopy.utils.CellBudgetFile(pl.Path(ws) / f"{NAME}.cbc")
    record = cbc.get_data(text="MAW", kstpkper=cbc.get_kstpkper()[-1])[0]
    return float(record["q"][iconn])


def _final_head(ws, cell):
    head = flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds")
    return float(head.get_data(kstpkper=head.get_kstpkper()[-1])[cell])


def _solve_adjoint(ws, entry, nstp=1, name="pm"):
    """Return the composite sensitivities of one performance measure."""
    ws = pl.Path(ws)
    with open(ws / "pm.dat", "w") as f:
        f.write(f"begin performance_measure {name}\n")
        f.write(f"  1 {nstp} {entry}\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / f"adjoint_solution_{name}.hd5", "r") as hf:
        return {key: hf["composite"][key][:] for key in hf["composite"]}


def _measure_entry(kind):
    """The performance-measure entry for the upper connection of the well."""
    i, j = MAW_CELL
    return f"1 {i + 1} {j + 1} {kind} direct 1.0 -1.0e+30"


@pytest.mark.parametrize("status", ["ACTIVE", "CONSTANT"])
@pytest.mark.parametrize("nstp", [1, 4])
def test_flux_measure(function_tmpdir, status, nstp):
    """A measure of the exchange matches a finite-difference derivative."""
    dq = -1.0  # small enough to stay in the linear range

    kwargs = {"status": status, "nstp": nstp}
    base = _build_model(function_tmpdir / "base", **kwargs)
    plus = _build_model(function_tmpdir / "plus", wel_rate=WELL_RATE + dq, **kwargs)
    minus = _build_model(function_tmpdir / "minus", wel_rate=WELL_RATE - dq, **kwargs)
    finite_difference = (_connection_flux(plus) - _connection_flux(minus)) / (2.0 * dq)

    # the measure has to respond, or it is testing nothing
    assert abs(finite_difference) > 1.0e-4

    composite = _solve_adjoint(base, _measure_entry("maw-1"), nstp=nstp)
    assert composite["wel6_q"][WELL_CELL] == pytest.approx(
        finite_difference, rel=1.0e-3
    )


def test_conductance_derivative(function_tmpdir):
    """The exchange follows the head through the conductance, not only the drop.

    The upper connection of the well is only partly saturated, so dropping the
    derivative of its conductance leaves a sensitivity that is wrong by more
    than the tolerance the test above holds the adjoint to.
    """
    from mf6adj.advanced_packages import MawCoupling

    dq = -1.0
    base = _build_model(function_tmpdir / "base")
    plus = _build_model(function_tmpdir / "plus", wel_rate=WELL_RATE + dq)
    minus = _build_model(function_tmpdir / "minus", wel_rate=WELL_RATE - dq)
    finite_difference = (_connection_flux(plus) - _connection_flux(minus)) / (2.0 * dq)

    original = MawCoupling.blocks

    def without_conductance_derivative(self, sol_dataset, gwf_package_dict, is_newton):
        blocks = original(self, sol_dataset, gwf_package_dict, is_newton)
        for block in blocks:
            block["dterm"] = np.zeros_like(block["dterm"])
        return blocks

    MawCoupling.blocks = without_conductance_derivative
    try:
        composite = _solve_adjoint(base, _measure_entry("maw-1"))
    finally:
        MawCoupling.blocks = original

    dropped = composite["wel6_q"][WELL_CELL]
    assert abs(dropped - finite_difference) > 0.05 * abs(finite_difference), (
        "the conductance derivative made no difference, so the model does not "
        "exercise a partly saturated screen"
    )


def test_head_measure(function_tmpdir):
    """A head measure in a model with a well matches a finite difference."""
    dq = -1.0
    observation = (0, 2, 2)

    base = _build_model(function_tmpdir / "base", nstp=4)
    plus = _build_model(function_tmpdir / "plus", wel_rate=WELL_RATE + dq, nstp=4)
    minus = _build_model(function_tmpdir / "minus", wel_rate=WELL_RATE - dq, nstp=4)
    finite_difference = (
        _final_head(plus, observation) - _final_head(minus, observation)
    ) / (2.0 * dq)

    k, i, j = observation
    entry = f"{k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30"
    composite = _solve_adjoint(base, entry, nstp=4)
    assert composite["wel6_q"][WELL_CELL] == pytest.approx(
        finite_difference, rel=1.0e-3
    )


def test_inactive_well(function_tmpdir):
    """An inactive well exchanges nothing and leaves the sensitivities alone."""
    observation = (0, 2, 2)
    k, i, j = observation
    entry = f"{k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30"

    inactive = _solve_adjoint(
        _build_model(function_tmpdir / "off", status="INACTIVE"), entry, name="off"
    )
    # the well's row stays in the matrix, holding the head it is given, so the
    # answer has to be the one the model gives with no well at all
    dq = -1.0
    plus = _build_model(
        function_tmpdir / "plus", wel_rate=WELL_RATE + dq, status="INACTIVE"
    )
    minus = _build_model(
        function_tmpdir / "minus", wel_rate=WELL_RATE - dq, status="INACTIVE"
    )
    finite_difference = (
        _final_head(plus, observation) - _final_head(minus, observation)
    ) / (2.0 * dq)
    assert inactive["wel6_q"][WELL_CELL] == pytest.approx(finite_difference, rel=1.0e-3)


def test_two_packages(function_tmpdir):
    """Two well packages keep their own rows in the solution."""
    dq = -1.0
    kwargs = {"second_package": True}
    base = _build_model(function_tmpdir / "base", **kwargs)
    plus = _build_model(function_tmpdir / "plus", wel_rate=WELL_RATE + dq, **kwargs)
    minus = _build_model(function_tmpdir / "minus", wel_rate=WELL_RATE - dq, **kwargs)
    finite_difference = (_connection_flux(plus) - _connection_flux(minus)) / (2.0 * dq)

    composite = _solve_adjoint(base, _measure_entry("maw-1"))
    assert composite["wel6_q"][WELL_CELL] == pytest.approx(
        finite_difference, rel=1.0e-3
    )


def test_measure_on_maw_accepted():
    """A measure naming a maw6 package is accepted rather than refused."""
    from mf6adj.utils.utils_pm_read import validate_pm_type

    validate_pm_type("maw-1", {"dis6": ["dis"], "maw6": ["maw-1"]})
