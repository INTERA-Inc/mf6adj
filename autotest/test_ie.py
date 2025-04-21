import os
import pathlib as pl
import platform
import shutil
import sys
from datetime import datetime

import flopy
import pyemu

sys.path.insert(0, str(pl.Path("../").resolve()))
import mf6adj

env_path = pl.Path(os.environ.get("CONDA_PREFIX", None))
assert env_path is not None, (
    "autotest script must be run from the mf6adj Conda environment"
)

bin_path = "bin"
exe_ext = ""
if "linux" in platform.platform().lower():
    lib_ext = ".so"
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    lib_ext = ".dylib"
else:
    bin_path = "Scripts"
    lib_ext = ".dll"
    exe_ext = ".exe"
lib_name = env_path / f"{bin_path}/libmf6{lib_ext}"
mf6_bin = env_path / f"{bin_path}/mf6{exe_ext}"


def test_ie_nomaw_1sp():
    prep = True

    org_d = os.path.join("ie_nomaw_1sp")
    new_d = "ie_nomaw_1sp_test"

    adj_file = os.path.join(new_d, "test.adj")
    if prep:
        if os.path.exists(new_d):
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)

        pyemu.os_utils.run("mf6", cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d, load_only=["dis", "sfr"])

    with open(adj_file, "w") as f:
        f.write("begin performance_measure single_all_times\n")
        for kper in range(sim.tdis.nper.data):
            nstp = sim.tdis.perioddata.array[0][1]
            print(nstp)
            f.write(f"{kper + 1} {nstp} {32} {1808} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)

    adj = mf6adj.Mf6Adj(os.path.split(adj_file)[1], lib_name, verbose_level=2)

    adj.solve_gwf()
    adj.solve_adjoint(
        linear_solver="bicgstab",
        linear_solver_kwargs={"maxiter": 500, "atol": 1e-5},
        use_precon=True,
    )
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)


def test_ie_1sp():
    prep = True

    org_d = os.path.join("ie_1sp")
    new_d = "ie_1sp_test"

    adj_file = os.path.join(new_d, "test.adj")
    if prep:
        if os.path.exists(new_d):
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)

        pyemu.os_utils.run("mf6", cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d, load_only=["dis", "sfr"])

    with open(adj_file, "w") as f:
        f.write("begin performance_measure single_all_times\n")
        for kper in range(sim.tdis.nper.data):
            nstp = sim.tdis.perioddata.array[0][1]
            print(nstp)
            f.write(f"{kper + 1} {nstp} {32} {1808} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)

    adj = mf6adj.Mf6Adj(os.path.split(adj_file)[1], lib_name, verbose_level=2)

    adj.solve_gwf()
    adj.solve_adjoint(
        linear_solver="bicgstab",
        linear_solver_kwargs={"maxiter": 500, "atol": 1e-5},
        use_precon=False,
    )
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)


if __name__ == "__main__":
    test_ie_1sp()
    test_ie_nomaw_1sp()
