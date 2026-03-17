import os
import pathlib as pl
import platform
import sys
from datetime import datetime

import flopy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import mf6adj
except ImportError:
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

perioddata = [(1.0, 1, 1.0), (1.0, 1, 1.0)]
nper = len(perioddata)

ncol, nrow = 3, 3
nlay, ncpl = 1, ncol * nrow
nodes = ncpl
delr = delc = 1.0

vertices = []
idx = 0
y = float(nrow) * delc
for i in range(nrow + 1):
    x = 0.0
    for j in range(ncol + 1):
        vertices.append([idx, x, y])
        idx += 1
        x += delr
    y -= delc

irow_starts = [0, 4, 8]
iverts = []
xc = []
yc = []
idx = 0
y = float(nrow) * delc - 0.5 * delc
for i in range(nrow):
    istart = irow_starts[i]
    x = 0.5 * delr
    for j in range(ncol):
        iverts.append([idx, istart, istart + 1, istart + 5, istart + 4])
        xc.append(x)
        yc.append(y)
        idx += 1
        istart += 1
        x += delr
    y -= delc

cell2d = []
for node in range(len(xc)):
    cell2d.append([node, xc[node], yc[node], len(iverts[node]) - 1] + iverts[node][1:])

iac = [3, 4, 3, 4, 5, 4, 3, 4, 3]
ja = [
    [0, 1, 3],
    [1, 0, 2, 4],
    [2, 1, 5],
    [3, 0, 4, 6],
    [4, 1, 3, 5, 7],
    [5, 2, 4, 8],
    [6, 3, 7],
    [7, 4, 6, 8],
    [8, 5, 7],
]
nja = 0
for jlist in ja:
    nja += len(jlist)
ja_arr = []
for j in ja:
    ja_arr += j
ja_arr = np.array(ja_arr, dtype=int)

ia = [0] + np.array(iac).cumsum().tolist()
ihc = []
cl12 = []
hwva = []
for idx in range(nodes):
    ihc.append(1)
    cl12.append(0.0)
    hwva.append(0.0)
    i0 = ia[idx] + 1
    i1 = ia[idx + 1]
    for jdx in range(i0, i1):
        ihc.append(1)
        jcol = ja_arr[jdx]
        if abs(idx - jcol) == 1:
            dist = 0.5 * delr
            width = delc
        else:
            dist = 0.5 * delc
            width = delr
        cl12.append(dist)
        hwva.append(width)

top = 2.0
botm = 0.0

riv_disv = {
    0: [
        ((0, 0), 1.0, 1.0, 0.5, "riv"),
        ((0, 3), 1.0, 1.0, 0.5, "riv"),
        ((0, 6), 1.0, 1.0, 0.5, "riv"),
    ]
}
drn_disv = {
    0: [
        ((0, 2), 0.25, 1.0, "drn"),
        ((0, 5), 0.25, 1.0, "drn"),
        ((0, 8), 0.25, 1.0, "drn"),
    ]
}
riv_disu = {
    0: [
        ((0,), 1.0, 1.0, 0.5, "riv"),
        ((3,), 1.0, 1.0, 0.5, "riv"),
        ((6,), 1.0, 1.0, 0.5, "riv"),
    ]
}
drn_disu = {
    0: [
        ((2,), 0.25, 1.0, "drn"),
        ((5,), 0.25, 1.0, "drn"),
        ((8,), 0.25, 1.0, "drn"),
    ]
}


def build_model(ws, name="disv"):
    sim = flopy.mf6.MFSimulation(sim_name=name, sim_ws=ws, exe_name=mf6_bin)
    tdis = flopy.mf6.ModflowTdis(sim, nper=nper, perioddata=perioddata)
    ims = flopy.mf6.ModflowIms(sim, complexity="simple", linear_acceleration="bicgstab")

    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, newtonoptions="newton")
    if name == "disv":
        dis = flopy.mf6.ModflowGwfdisv(
            gwf,
            nlay=nlay,
            ncpl=ncpl,
            top=top,
            botm=botm,
            nvert=len(vertices),
            vertices=vertices,
            cell2d=cell2d,
        )
    else:
        dis = flopy.mf6.ModflowGwfdisu(
            gwf,
            nodes=nodes,
            nja=nja,
            iac=iac,
            ja=ja_arr,
            area=delr * delc,
            ihc=ihc,
            cl12=cl12,
            hwva=hwva,
            top=top,
            bot=botm,
            nvert=len(vertices),
            vertices=vertices,
            cell2d=cell2d,
        )

    npf = flopy.mf6.ModflowGwfnpf(gwf, icelltype=1)
    sto = flopy.mf6.ModflowGwfsto(gwf, iconvert=1, transient={0: True})
    ic = flopy.mf6.ModflowGwfic(gwf, strt=0.75)
    if name == "disv":
        riv_spd = riv_disv
        drn_spd = drn_disv
    else:
        riv_spd = riv_disu
        drn_spd = drn_disu
    riv = flopy.mf6.ModflowGwfriv(
        gwf, boundnames=True, maxbound=nrow, stress_period_data=riv_spd
    )
    obs = {f"{name}.riv.obs.csv": [["riv", "riv", "riv"]]}
    riv.obs.initialize(
        filename=f"{name}.riv.obs",
        continuous=obs,
    )
    drn = flopy.mf6.ModflowGwfdrn(
        gwf, boundnames=True, maxbound=nrow, stress_period_data=drn_spd
    )
    obs = {f"{name}.drn.obs.csv": [["drn", "drn", "drn"]]}
    drn.obs.initialize(
        filename=f"{name}.drn.obs",
        continuous=obs,
    )

    sim.write_simulation()

    pm_fname = f"{name}_perfmeas.dat"
    with open(ws / pm_fname, "w") as fpm:
        for bnd in ("river", "drain"):
            if bnd == "river":
                spd = riv_spd
                pak = "riv_0"
            else:
                spd = drn_spd
                pak = "drn_0"
            cellids = [v[0] for v in spd[0]]

            fpm.write(f"begin performance_measure {bnd}\n")
            for kper in range(sim.tdis.nper.data):
                for cellid in cellids:
                    if name == "disv":
                        k, node = cellid
                        line = (
                            f"{kper + 1} 1 {k + 1} {node + 1} {pak} "
                            + "direct 1.0 -1.0e+30\n"
                        )
                    else:
                        node = cellid[0]
                        line = f"{kper + 1} 1 {node + 1} {pak} direct 1.0 -1.0e+30\n"
                    fpm.write(line)
            fpm.write("end performance_measure\n\n")

    return sim, pm_fname


def solve_adjoint(ws, pm_fname):
    bd = pl.Path.cwd()
    os.chdir(ws)

    forward_hdf5_name = "test_unstructured_forward.hdf5"
    start = datetime.now()

    adj = mf6adj.Mf6Adj(pm_fname, lib_name, logging_level="INFO")
    adj.solve_gwf(hdf5_name=forward_hdf5_name)
    dfsum = adj.solve_adjoint()
    adj.finalize()  # release components
    duration = (datetime.now() - start).total_seconds()
    print("adjoint took:", duration)

    os.chdir(bd)


def get_sensitivities(ws):
    results = {}
    for bnd in ("river", "drain"):
        result_hdf = ws / f"adjoint_solution_{bnd}.hdf5"
        hdf = h5py.File(result_hdf, "r")
        for key in hdf.keys():
            if key == "composite":
                arr = hdf[key]["wel6_q"][:]
                results[bnd] = arr
            else:
                arr = hdf[key]["wel6_q"][:]
            print(f"{key}: min ({arr.min()}) max ({arr.max()})")
    return results


def get_observations(sim):
    gwf = sim.get_model()

    pak = gwf.riv
    riv_obs = pak.output.obs().get_data()
    print(f"RIVER OBS CUMSUM: {np.cumsum(riv_obs['RIV'][::-1])}")

    pak = gwf.drn
    drn_obs = pak.output.obs().get_data()
    print(f"DRAIN OBS CUMSUM: {np.cumsum(drn_obs['DRN'][::-1])}")

    return riv_obs["RIV"], drn_obs["DRN"]


def test_unstructured():
    name = "disv"
    ws = pl.Path(f"unstructured_{name}_test")
    ws.mkdir(parents=True, exist_ok=True)
    _, pm_fname = build_model(ws, name=name)
    solve_adjoint(ws, pm_fname)

    result_disv = get_sensitivities(ws)

    name = "disu"
    ws = pl.Path(f"unstructured_{name}_test")
    ws.mkdir(parents=True, exist_ok=True)
    _, pm_fname = build_model(ws, name=name)
    solve_adjoint(ws, pm_fname)

    result_disu = get_sensitivities(ws)

    for bnd in ("river", "drain"):
        err_msg = f"results for {bnd} do not match"
        assert np.array_equal(result_disv[bnd], result_disu[bnd]), err_msg
        minval = float(result_disv[bnd].min())
        print(f"disv {bnd} min value: {minval}")
        # assert minval >= -1.0, f"disv min >= -1 ({minval})"
        minval = float(result_disu[bnd].min())
        print(f"disu {bnd} min value: {minval}")
        # assert minval >= -1.0, f"disu min >= -1 ({minval})"
