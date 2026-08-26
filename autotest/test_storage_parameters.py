"""
Tests the sensitivities to specific storage and to specific yield.

Storage releases water in two ways, and a convertible cell uses both: specific
storage as the head falls while the cell stays full, specific yield as the cell
drains. They are separate parameters of the flow model, so they carry separate
sensitivities, and each is checked here against a finite-difference derivative
taken by perturbing that parameter alone.

Cases:
  - test_specific_storage_sensitivity : the reported ss sensitivity matches a
                                        finite difference in ss.
  - test_specific_yield_sensitivity   : the reported sy sensitivity matches a
                                        finite difference in sy.
  - test_sensitivities_are_separate   : perturbing one parameter leaves the
                                        other one's finite difference alone,
                                        so the two are not the same quantity.
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

NAME = "sto"
NROW = NCOL = 7
DELRC = 100.0
TOP, BOTM = 10.0, 0.0
WELL_CELL = (0, 3, 3)
OBS_CELL = (0, 1, 1)
WELL_RATE = -60.0
NPER, NSTP, PERLEN = 1, 4, 400.0
SS = 1.0e-3
SY = 0.2


def _build_model(ws, ss=SS, sy=SY, confined_only=False, strt=8.0):
    """A partly drained aquifer, so both storage terms are active."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=NPER, perioddata=[(PERLEN, NSTP, 1.0)])
    flopy.mf6.ModflowIms(
        sim,
        complexity="complex",
        outer_dvclose=1.0e-9,
        inner_dvclose=1.0e-10,
        outer_maximum=200,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=NAME, newtonoptions="NEWTON", save_flows=True
    )
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=NROW, ncol=NCOL, delr=DELRC, delc=DELRC, top=TOP, botm=BOTM
    )
    flopy.mf6.ModflowGwfic(gwf, strt=strt)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=1, k=1.0)
    kwargs = {"iconvert": 1, "ss": ss, "sy": sy, "transient": {0: True}}
    if confined_only:
        kwargs["ss_confined_only"] = True
    flopy.mf6.ModflowGwfsto(gwf, **kwargs)
    flopy.mf6.ModflowGwfghb(
        gwf,
        stress_period_data=[[(0, i, 0), strt, 50.0] for i in range(NROW)],
        pname="ghb-1",
    )
    flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=[[WELL_CELL, WELL_RATE]], pname="wel-1"
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord=f"{NAME}.hds",
        saverecord=[("HEAD", "ALL")],
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(str(line) for line in buff[-20:])
    return ws


def _final_head(ws):
    head = flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds")
    return float(head.get_data(kstpkper=head.get_kstpkper()[-1])[OBS_CELL])


def _solve_adjoint(ws):
    """Sensitivities of the final head at the observation cell."""
    ws = pl.Path(ws)
    k, i, j = OBS_CELL
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 {NSTP} {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / "adjoint_solution_obs.hd5", "r") as hf:
        return {name: hf["composite"][name][:].copy() for name in ("ss", "sy")}


def _crosses_top(ws):
    """True where the water level starts above the cell top and falls past it."""
    head = flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds").get_alldata()
    return bool((head >= TOP).any() and (head < TOP).any())


def _drained(ws):
    """True where the model leaves cells between full and dry."""
    head = flopy.utils.HeadFile(pl.Path(ws) / f"{NAME}.hds").get_data()
    saturation = np.clip((head - BOTM) / (TOP - BOTM), 0.0, 1.0)
    return saturation.max() < 0.99 and saturation.min() > 0.01


@pytest.mark.parametrize("parameter", ["ss", "sy"])
@pytest.mark.parametrize(
    "confined_only", [False, True], ids=["default", "confined_only"]
)
def test_storage_parameter_sensitivity(function_tmpdir, parameter, confined_only):
    """Each storage sensitivity matches a finite difference in that parameter."""
    fraction = 1.0e-3
    tag = f"{parameter}_{int(confined_only)}"
    # under SS_CONFINED_ONLY specific storage acts only while a cell is full, so
    # the water level has to start above the top of the cell and fall past it
    strt = 12.0 if confined_only else 8.0
    base_ws = _build_model(
        function_tmpdir / f"base_{tag}", confined_only=confined_only, strt=strt
    )
    assert _crosses_top(base_ws) if confined_only else _drained(base_ws), (
        "the model has to exercise the saturation the option switches on"
    )

    if parameter == "ss":
        step = SS * fraction
        pert_ws = _build_model(
            function_tmpdir / f"pert_{tag}",
            ss=SS + step,
            confined_only=confined_only,
            strt=strt,
        )
    else:
        step = SY * fraction
        pert_ws = _build_model(
            function_tmpdir / f"pert_{tag}",
            sy=SY + step,
            confined_only=confined_only,
            strt=strt,
        )
    finite_difference = (_final_head(pert_ws) - _final_head(base_ws)) / step

    # the parameter was changed in every cell, so compare the summed sensitivity
    adjoint = float(np.sum(_solve_adjoint(base_ws)[parameter]))
    assert np.isclose(adjoint, finite_difference, rtol=5.0e-2), (
        f"the {parameter} sensitivity {adjoint:.6e} does not match the "
        f"finite-difference derivative {finite_difference:.6e}"
    )


def test_sensitivities_are_separate(function_tmpdir):
    """The two parameters move the head by different amounts."""
    fraction = 1.0e-3
    base_ws = _build_model(function_tmpdir / "base")
    base_head = _final_head(base_ws)

    ss_ws = _build_model(function_tmpdir / "ss", ss=SS * (1.0 + fraction))
    sy_ws = _build_model(function_tmpdir / "sy", sy=SY * (1.0 + fraction))
    from_ss = (_final_head(ss_ws) - base_head) / (SS * fraction)
    from_sy = (_final_head(sy_ws) - base_head) / (SY * fraction)

    assert not np.isclose(from_ss, from_sy, rtol=1.0e-2), (
        "the model responds to the two storage parameters identically, so the "
        "test cannot tell their sensitivities apart"
    )
    sensitivities = _solve_adjoint(base_ws)
    assert not np.allclose(sensitivities["ss"], sensitivities["sy"], rtol=1.0e-6), (
        "the reported ss and sy sensitivities are the same array"
    )
