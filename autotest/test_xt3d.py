"""
Tests for the refusal of a flow model that used XT3D.

The conductivity sensitivity is the derivative of a conductance between a cell
and one neighbor, over the length between them and the area they share. XT3D
does not form the flow that way: it uses a conductivity tensor over a wider set
of neighbors, so what links two cells depends on the conductivity of cells that
are neither of them. The derivative the adjoint forms is then not the
derivative of the equations the model solved, and unlike a horizontal flow
barrier, which reaches only the connections it sits on, this reaches all of
them.

Nothing here forms the XT3D derivative. The model is refused instead, where it
is read, before any forward solve.

Cases:
  - test_xt3d_is_refused        : a model that used XT3D is refused, in both
                                  the matrix and the right-hand-side variants.
  - test_without_xt3d_is_read   : the same model without it is read and solved.
"""

import pathlib as pl
import shutil
import sys

import flopy
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

NAME = "xt3d"
NLAY, NROW, NCOL = 1, 5, 5
OBS = (0, 2, 2)


def _build(ws, xt3d=None):
    """A small confined model, optionally solved with XT3D."""
    ws = pl.Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_bin)
    flopy.mf6.ModflowTdis(sim, nper=1, perioddata=[(1.0, 1, 1.0)])
    # XT3D leaves an asymmetric matrix, which the conjugate gradient method
    # cannot solve, so both variants of the model use the same accelerator
    flopy.mf6.ModflowIms(sim, complexity="simple", linear_acceleration="BICGSTAB")
    gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
    flopy.mf6.ModflowGwfdis(
        gwf, nlay=NLAY, nrow=NROW, ncol=NCOL, delr=10.0, delc=10.0, top=10.0, botm=0.0
    )
    flopy.mf6.ModflowGwfic(gwf, strt=5.0)
    npf = {"icelltype": 0, "k": 10.0}
    if xt3d == "matrix":
        npf["xt3doptions"] = ["xt3d"]
    elif xt3d == "rhs":
        npf["xt3doptions"] = ["xt3d", "rhs"]
    flopy.mf6.ModflowGwfnpf(gwf, **npf)
    flopy.mf6.ModflowGwfchd(
        gwf,
        stress_period_data=[[(0, i, 0), 6.0] for i in range(NROW)]
        + [[(0, i, NCOL - 1), 4.0] for i in range(NROW)],
        pname="chd-1",
    )
    flopy.mf6.ModflowGwfoc(
        gwf, head_filerecord=f"{NAME}.hds", saverecord=[("HEAD", "ALL")]
    )
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True)
    assert success, "\n".join(buff[-20:])

    k, i, j = OBS
    with open(ws / "pm.dat", "w") as f:
        f.write("begin performance_measure obs\n")
        f.write(f"  1 1 {k + 1} {i + 1} {j + 1} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    return ws


@pytest.mark.parametrize("xt3d", ["matrix", "rhs"])
def test_xt3d_is_refused(function_tmpdir, xt3d):
    """A model that used XT3D is refused as it is read.

    Both variants put the same flow between two cells; they differ in whether
    the terms go into the matrix or onto the right-hand side, and neither is a
    flow the adjoint can differentiate.
    """
    ws = _build(function_tmpdir / xt3d, xt3d=xt3d)
    with pytest.raises(Exception, match="XT3D"):
        mf6adj.Mf6Adj(
            "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
        )


def test_without_xt3d_is_read(function_tmpdir):
    """The same model without XT3D is read and solved, so the check is specific."""
    ws = _build(function_tmpdir / "plain")
    adj = mf6adj.Mf6Adj(
        "pm.dat", lib_name, logging_level="ERROR", working_directory=str(ws)
    )
    adj.solve_forward_model(hdf5_name="fwd.hd5")
    df = adj.solve_adjoint()
    adj.finalize()
    assert df["obs"]["k11"].notna().all()
