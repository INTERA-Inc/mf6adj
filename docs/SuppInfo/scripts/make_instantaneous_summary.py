"""Where an instantaneous measure is exact, and where it cannot be accumulated.

One model with a steady-state period followed by transient periods. A steady
step carries nothing between time steps, so an instantaneous measure and a
direct one agree. A transient step does, and they part company.
"""
import pathlib as pl
import shutil
import sys

import flopy
import h5py
import numpy as np

import mf6adj

ROOT = pl.Path(sys.argv[1])
OUT = pl.Path(sys.argv[2])
NAME = "inst"
NROW = NCOL = 21
TOP, BOTM = 0.0, -50.0
DELRC = 250.0
NPER, NSTP, PERLEN = 6, 4, 50.0
WELL = (0, 10, 10)
OBS = (0, 5, 5)
RATE = -500.0

mf6_exe, lib_name = mf6adj.get_conda_mf6_paths()

ws = ROOT / "model"
if ws.exists():
    shutil.rmtree(ws)
ws.mkdir(parents=True)
sim = flopy.mf6.MFSimulation(sim_name=NAME, sim_ws=str(ws), exe_name=mf6_exe)
# the first period is steady state, the rest transient
flopy.mf6.ModflowTdis(sim, nper=NPER, perioddata=[(PERLEN, NSTP, 1.0)] * NPER,
                      time_units="days")
flopy.mf6.ModflowIms(sim, complexity="simple", outer_dvclose=1e-9,
                     inner_dvclose=1e-10)
gwf = flopy.mf6.ModflowGwf(sim, modelname=NAME, save_flows=True)
flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=NROW, ncol=NCOL, delr=DELRC,
                        delc=DELRC, top=TOP, botm=BOTM)
flopy.mf6.ModflowGwfic(gwf, strt=0.0)
flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=10.0)
flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=1.0e-5, sy=0.0,
                        steady_state={0: True},
                        transient={1: True})
flopy.mf6.ModflowGwfghb(
    gwf, stress_period_data=[[(0, i, 0), 0.0, 500.0] for i in range(NROW)]
    + [[(0, i, NCOL - 1), 0.0, 500.0] for i in range(NROW)], pname="ghb-1")
flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[[WELL, RATE]], pname="wel-1")
flopy.mf6.ModflowGwfoc(gwf, head_filerecord=f"{NAME}.hds",
                       budget_filerecord=f"{NAME}.cbc",
                       saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
sim.write_simulation(silent=True)
ok, buff = sim.run_simulation(silent=True)
assert ok, "\n".join(buff[-12:])
head = flopy.utils.HeadFile(ws / f"{NAME}.hds").get_alldata()

values = {}
for form in ("direct", "instantaneous"):
    path = ws / f"{form}.adj"
    k, i, j = OBS
    with open(path, "w") as f:
        f.write(f"begin performance_measure {form}\n")
        for kper in range(NPER):
            for kstp in range(NSTP):
                f.write(f"  {kper + 1} {kstp + 1} {k + 1} {i + 1} {j + 1} "
                        f"head {form} 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")
    adj = mf6adj.Mf6Adj(path.name, str(lib_name), logging_level="WARNING",
                        working_directory=str(ws))
    adj.solve_forward_model(hdf5_name=f"fwd_{form}.hd5")
    adj.solve_adjoint()
    adj.finalize()
    with h5py.File(ws / f"adjoint_solution_{form}.hd5", "r") as hf:
        per_step = {}
        for key in hf:
            if not key.startswith("solution_"):
                continue
            kper = int(key.split("kper:")[1].split("_")[0])
            kstp = int(key.split("kstp:")[1].split("_")[0])
            per_step[(kper, kstp)] = float(hf[key]["wel6_q"][:][WELL])
    values[form] = per_step

order = [(p, s) for p in range(NPER) for s in range(NSTP)]
direct = np.array([values["direct"][k] for k in order])
inst = np.array([values["instantaneous"][k] for k in order])
simulated = -head[:, OBS[0], OBS[1], OBS[2]]
accumulated = np.cumsum(inst * RATE * -1.0)

print(f"{'step':>5} {'period':>7} {'direct':>12} {'instantaneous':>14} {'ratio':>8}")
for n, (p, s) in enumerate(order):
    if n < 4 or n in (4, 8, len(order) - 1):
        print(f"{n + 1:5d} {p + 1:7d} {direct[n]:12.6g} {inst[n]:14.6g} "
              f"{inst[n] / direct[n]:8.4f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT, direct=direct, instantaneous=inst, simulated=simulated,
    accumulated=accumulated, nstp=np.array([NSTP]), nper=np.array([NPER]),
    perlen=np.array([PERLEN]),
)
print(f"wrote {OUT}")
