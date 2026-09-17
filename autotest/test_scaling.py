"""
Tests the diagonal scaling the adjoint system is optionally solved with.

Scaling by the square root of the diagonal conditions the columns of the
transpose as well as its rows, so it is off by default. The sensitivities it
returns have to be the ones the unscaled solve returns, on its own and
alongside the normalization of lambda and the right-hand side.

Cases:
  - test_the_scaled_solve_matches : the scaled solve returns the adjoint state
                                   the unscaled solve does.
  - test_the_scaled_solve_matches_with_dvscale : the same, with lambda and the
                                   right-hand side normalized as well.
"""

import pathlib as pl
import shutil
import sys

import flopy
import h5py
import numpy as np

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()


def _forward_file(ws):
    """Return the working directory of a small transient model."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)

    sim = flopy.mf6.MFSimulation(sim_name="scl", sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(100.0, 2, 1.0)])
    flopy.mf6.ModflowIms(sim, complexity="simple")
    gwf = flopy.mf6.ModflowGwf(sim, modelname="scl", save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=1, nrow=1, ncol=10, delr=100.0, delc=100.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=10.0)
    # the conductivity spans three orders of magnitude, so the diagonal the
    # system is scaled by has a range to it
    flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=0, k=[[[0.1, 1.0, 10.0, 100.0, 10.0, 1.0, 0.1, 1.0, 10.0, 1.0]]]
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, transient={0: True})
    flopy.mf6.ModflowGwfdrn(
        gwf, stress_period_data=[[(0, 0, 0), 5.0, 1000.0]], pname="drn-1"
    )
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[[(0, 0, 9), 10.0]], pname="chd-1")
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="scl.hds", saverecord=[("HEAD", "ALL")])
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write("  1 2 1 1 4 head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")
    return ws


def _state(ws, **kwargs):
    """Return the adjoint state of every step, solved with the options given."""
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="WARNING", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    adj._performance_measures[0].solve_adjoint(
        ws / "fwd.hd5",
        hdf5_adjoint_solution_fname=str(ws / "adj.hd5"),
        linear_solver="bicgstab",
        linear_solver_kwargs={"rtol": 1.0e-12},
        dvclose=None,
        rclose=None,
        **kwargs,
    )
    adj.finalize()

    with h5py.File(ws / "adj.hd5", "r") as f:
        keys = sorted(k for k in f if k.startswith("solution_kper"))
        assert keys, "the solve wrote no step"
        return np.concatenate([np.asarray(f[k]["lambda"][:]) for k in keys])


def test_the_scaled_solve_matches(function_tmpdir):
    """The scaled solve returns the adjoint state the unscaled solve does."""
    plain = _state(_forward_file(function_tmpdir / "plain"), scale_system=False)
    scaled = _state(_forward_file(function_tmpdir / "scaled"), scale_system=True)

    assert np.isfinite(scaled).all(), "the scaled solve returned no number"
    assert np.abs(plain).max() > 0.0, "the unscaled solve returned nothing"
    assert np.allclose(scaled, plain, rtol=1.0e-6, atol=1.0e-6 * np.abs(plain).max()), (
        "the scaled solve returned a different adjoint state, off by "
        + f"{np.abs(scaled - plain).max():.3e}"
    )


def test_the_scaled_solve_matches_with_dvscale(function_tmpdir):
    """The same, with lambda and the right-hand side normalized as well.

    The normalization divides the vector the solver is handed, which is not
    the right-hand side itself once the system has been scaled.
    """
    plain = _state(_forward_file(function_tmpdir / "plain"), scale_system=False)
    scaled = _state(
        _forward_file(function_tmpdir / "scaled"), scale_system=True, dvscale=True
    )

    assert np.isfinite(scaled).all(), "the scaled solve returned no number"
    assert np.allclose(scaled, plain, rtol=1.0e-6, atol=1.0e-6 * np.abs(plain).max()), (
        "the scaled solve returned a different adjoint state, off by "
        + f"{np.abs(scaled - plain).max():.3e}"
    )
