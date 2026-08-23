import os
import pathlib as pl
import platform
import shutil
import sys
from datetime import datetime

import flopy
import pytest

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

env_path = pl.Path(os.environ.get("CONDA_PREFIX", None))
assert env_path is not None, (
    "autotest script must be run from the mf6adj Conda environment"
)

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()


def test_ie_nomaw_1sp():
    prep = True

    org_d = pl.Path("ie_nomaw_1sp")
    new_d = "ie_nomaw_1sp_test"
    new_dir = pl.Path(new_d)

    adj_file = new_dir / "test.adj"
    if prep:
        if new_dir.exists():
            shutil.rmtree(new_dir)
        shutil.copytree(org_d, new_dir)

        flopy.run_model(exe_name=mf6_bin, namefile=None, model_ws=new_dir)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_dir, load_only=["dis", "sfr"])
    nstp = sim.tdis.perioddata.array[0][1]

    with open(adj_file, "w") as f:
        f.write("begin performance_measure single_all_times\n")
        for kper in range(sim.tdis.nper.data):
            f.write(f"{kper + 1} {nstp} {32} {1808} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")

    start = datetime.now()

    adj = mf6adj.Mf6Adj(
        adj_file.name,
        lib_name,
        logging_level="INFO",
        working_directory=new_dir,
    )

    adj.solve_forward_model()
    adj.solve_adjoint(
        linear_solver="bicgstab",
        linear_solver_kwargs={"maxiter": 500, "atol": 1e-5},
        use_precon=True,
    )
    adj.finalize()

    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)


def test_ie_1sp():
    """The same model with maw6 is rejected: MAW adds equations to the matrix."""
    org_d = pl.Path("ie_1sp")
    new_d = "ie_1sp_test"
    new_dir = pl.Path(new_d)

    adj_file = new_dir / "test.adj"
    if new_dir.exists():
        shutil.rmtree(new_dir)
    shutil.copytree(org_d, new_dir)

    # the coupling check runs before the adjoint file is read, so the file only
    # has to exist
    with open(adj_file, "w") as f:
        f.write("begin performance_measure single\n")
        f.write(f"1 1 {32} {1808} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n")

    with pytest.raises(Exception, match="solution matrix"):
        mf6adj.Mf6Adj(
            adj_file.name,
            lib_name,
            logging_level="INFO",
            working_directory=new_dir,
        )


if __name__ == "__main__":
    test_ie_1sp()
    test_ie_nomaw_1sp()
