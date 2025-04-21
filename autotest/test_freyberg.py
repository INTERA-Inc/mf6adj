import os
import pathlib as pl
import platform
import shutil
import sys
from datetime import datetime

import flopy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyemu
from flopy.utils.gridgen import Gridgen
from matplotlib.backends.backend_pdf import PdfPages

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
gg_bin = env_path / f"{bin_path}/gridgen{exe_ext}"


def test_freyberg_structured():
    org_d = "freyberg_structured"
    new_d = "freyberg_structured_test"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)

    pyemu.os_utils.run("mf6", cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    sfr_data = pd.DataFrame.from_records(gwf.sfr.packagedata.array)

    lrcs = []
    k_dict = {}
    with open(os.path.join(new_d, "head.obs"), "r") as f:
        f.readline()
        for line in f:
            if line.strip().lower().startswith("end"):
                break
            raw = line.strip().split()
            lrcs.append(" ".join(raw[2:]))
            k = int(raw[2]) - 1
            i = int(raw[3]) - 1
            j = int(raw[4]) - 1
            if k not in k_dict:
                k_dict[k] = []
            k_dict[k].append([i, j])

    np.random.seed(11111)
    rvals = np.random.random(len(lrcs)) + 36
    with open(os.path.join(new_d, "test.adj"), "w") as f:
        f.write("begin performance_measure pm1\n")
        for rval, lrc in zip(rvals, lrcs):
            for kper in range(sim.tdis.nper.data):
                f.write(f"{kper + 1} 1 {lrc} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")

        sfr_data = pd.DataFrame.from_records(gwf.sfr.packagedata.array)
        bnames = sfr_data.boundname.unique()
        bnames.sort()
        for bname in bnames:
            bdf = sfr_data.loc[sfr_data.boundname == bname, :].copy()

            f.write(f"begin performance_measure {bname}\n")
            for kper in range(sim.tdis.nper.data):
                for kij in bdf.cellid.values:
                    f.write(
                        f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} sfr_1 direct 1.0 -1.0e+30\n"
                    )
            f.write("end performance_measure\n\n")

        f.write("begin performance_measure pm-combo\n")
        for rval, lrc in zip(rvals, lrcs):
            for kper in range(sim.tdis.nper.data):
                f.write(f"{kper + 1} 1 {lrc} head direct 1.0 -1.0e+30\n")
        for bname in bnames:
            bdf = sfr_data.loc[sfr_data.boundname == bname, :].copy()
            for kper in range(sim.tdis.nper.data):
                for kij in bdf.cellid.values:
                    f.write(
                        f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} sfr_1 direct 1.0 -1.0e+30\n"
                    )
        f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)
    adj = mf6adj.Mf6Adj("test.adj", lib_name, verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)

    result_hdfs = [
        f
        for f in os.listdir(new_d)
        if f.endswith("hd5") and f.startswith("adjoint_solution")
    ]
    assert len(result_hdfs) == 4, len(result_hdfs)
    for result_hdf in result_hdfs:
        print(result_hdf)

        hdf = h5py.File(os.path.join(new_d, result_hdf), "r")
        keys = list(hdf.keys())
        keys.sort()
        print(keys)

        idomain = np.loadtxt(os.path.join(new_d, "freyberg6.dis_idomain_layer1.txt"))
        with PdfPages(os.path.join(new_d, result_hdf + ".pdf")) as pdf:
            for key in keys:
                if key != "composite":
                    continue

                grp = hdf[key]
                plot_keys = [i for i in grp.keys() if len(grp[i].shape) == 3]
                for pkey in plot_keys:
                    arr = grp[pkey][:]
                    for k, karr in enumerate(arr):
                        karr[idomain < 1] = np.nan
                        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                        cb = ax.imshow(karr, cmap="gist_stern")
                        plt.colorbar(cb, ax=ax, label="composite sensitivity")
                        ax.set_title(key + ", " + pkey + f", layer:{k + 1}", loc="left")
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close(fig)
                        print("...", key, pkey, k + 1)


def test_freyberg_quadtree():
    org_d = "freyberg_quadtree"
    new_d = "freyberg_quadtree_test"
    prep_run = True
    run_adj = True

    sim = flopy.mf6.MFSimulation.load(sim_ws=os.path.join(org_d))
    m = sim.get_model()

    if prep_run:
        if os.path.exists(new_d):
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)
        pyemu.os_utils.run("mf6", cwd=new_d)

    if run_adj:
        df = pd.read_csv(
            os.path.join(new_d, "freyberg6.obs_continuous_heads.csv.txt"),
            header=None,
            names=["site", "otype", "layer", "node"],
        )
        df.loc[:, "layer"] = df.layer.astype(int)
        df.loc[:, "node"] = df.node.astype(int)

        np.random.seed(11111)
        rvals = np.random.random(df.shape[0]) + 36
        with open(os.path.join(new_d, "test.adj"), "w") as f:
            f.write("begin performance_measure pm1\n")
            for rval, lay, node in zip(rvals, df.layer, df.node):
                for kper in range(25):
                    f.write(f"{kper + 1} 1 {lay} {node} head residual 1.0  {rval}\n")
            f.write("end performance_measure\n\n")

            sfr_data = pd.DataFrame.from_records(m.sfr.packagedata.array)
            bnames = sfr_data.boundname.unique()
            bnames.sort()
            bnames = ["upstream", "downstream"]
            for bname in bnames:
                bdf = sfr_data.loc[sfr_data.boundname == bname, :].copy()
                assert bdf.shape[0] > 0

                f.write(f"begin performance_measure {bname}\n")
                for kper in range(sim.tdis.nper.data):
                    for kij in bdf.cellid.values:
                        print(kij)
                        f.write(
                            f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} sfr_0 direct 1.0 -1.0e+30\n"
                        )
                f.write("end performance_measure\n\n")

        start = datetime.now()
        os.chdir(new_d)
        adj = mf6adj.Mf6Adj("test.adj", lib_name, verbose_level=2)
        adj.solve_gwf()
        adj.solve_adjoint()
        adj.finalize()
        os.chdir("..")
        duration = (datetime.now() - start).total_seconds()
        print("took:", duration)

    result_hdf = [
        f
        for f in os.listdir(new_d)
        if f.endswith("hd5") and f.startswith("adjoint_solution_pm")
    ]

    result_hdf.sort()
    print(result_hdf)
    result_hdf = result_hdf[-1]
    print("using hdf", result_hdf)

    hdf = h5py.File(os.path.join(new_d, result_hdf), "r")
    keys = list(hdf.keys())
    keys.sort()
    print(keys)

    nplc = m.dis.top.array.shape[0]
    head = hdf["solution_kper:00000_kstp:00000"]["head"][:]
    nlay = int(head.shape[0] / nplc)
    head = head.reshape((nlay, nplc))
    idomain = m.dis.idomain.array.copy()
    idomain = idomain.reshape((nlay, nplc))

    with PdfPages(os.path.join(new_d, "results.pdf")) as _:
        for key in keys:
            print(key)
            if key != "composite":
                continue
            grp = hdf[key]
            plot_keys = [i for i in grp.keys()]

            for pkey in plot_keys:
                print(pkey)
                # if "k11" not in pkey:
                #    continue

                arr = grp[pkey][:]
                nlay = int(arr.shape[0] / nplc)
                arr = arr.reshape((nlay, nplc))  # .transpose()
                for k, karr in enumerate(arr):
                    print(karr)
                    karr[idomain[k] == 0] = np.nan
                    print(np.nanmin(karr), np.nanmax(karr))

                    m.dis.top = karr  # np.log10(np.abs(karr))
                    m.dis.top.plot(colorbar=True)
                    h = head[k].copy()
                    h[idomain[k] == 0] = np.nan
                    m.dis.top = h
                    print(np.nanmin(h), np.nanmax(h))

                    m.dis.top.plot(colorbar=True)
                    print(h)
                    return


def freyberg_structured_highres():
    org_d = "freyberg_highres"
    new_d = "freyberg_highres_test"

    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)

    pyemu.os_utils.run("mf6", cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    df = pd.read_csv(
        os.path.join(new_d, "freyberg6.obs_continuous_heads.csv.txt"),
        header=None,
        names=["site", "otype", "layer", "row", "col"],
        sep=r"\s+",
    )
    df.loc[:, "layer"] = df.layer.astype(int)
    df.loc[:, "row"] = df.row.astype(int)
    df.loc[:, "col"] = df.col.astype(int)

    np.random.seed(11111)
    rvals = np.random.random(df.shape[0]) + 36
    with open(os.path.join(new_d, "test.adj"), "w") as f:
        f.write("begin performance_measure pm1\n")
        for rval, lay, row, col in zip(rvals, df.layer, df.row, df.col):
            for kper in range(25):
                f.write(f"{kper + 1} 1 {lay} {row} {col} head direct 1.0 -1.0e+30\n")
        f.write("end performance_measure\n\n")

        sfr_data = pd.DataFrame.from_records(gwf.sfr.packagedata.array)
        bnames = sfr_data.boundname.unique()
        bnames.sort()
        for bname in bnames:
            bdf = sfr_data.loc[sfr_data.boundname == bname, :].copy()

            f.write(f"begin performance_measure {bname}\n")
            for kper in range(sim.tdis.nper.data):
                for kij in bdf.cellid.values:
                    f.write(
                        f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} sfr_1 direct 1.0 -1.0e+30\n"
                    )
            f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)
    adj = mf6adj.Mf6Adj("test.adj", lib_name, verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)

    result_hdf = [
        f
        for f in os.listdir(new_d)
        if f.endswith("hd5") and f.startswith("adjoint_solution_headwater")
    ]
    print(result_hdf)
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]

    hdf = h5py.File(os.path.join(new_d, result_hdf), "r")
    keys = list(hdf.keys())
    keys.sort()
    print(keys)

    idomain = np.loadtxt(os.path.join(new_d, "freyberg6.dis_idomain_layer1.txt"))
    with PdfPages(os.path.join(new_d, "results.pdf")) as pdf:
        for key in keys:
            if key != "composite":
                continue

            grp = hdf[key]
            plot_keys = [i for i in grp.keys() if len(grp[i].shape) == 3]
            for pkey in plot_keys:
                arr = grp[pkey][:]
                for k, karr in enumerate(arr):
                    karr[idomain < 1] = np.nan
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    mx = np.nanmax(np.abs(karr))
                    cb = ax.imshow(karr, vmax=mx, vmin=-mx, cmap="bwr")
                    plt.colorbar(cb, ax=ax, label="composite sensitivity")
                    ax.set_title(key + ", " + pkey + f", layer:{k + 1}", loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...", key, pkey, k + 1)


def test_freyberg_notional_unstruct():
    org_d = "freyberg_structured"
    new_d = "freyberg_notional_unstructured_test"
    if os.path.exists(new_d):
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)

    pyemu.os_utils.run("mf6", cwd=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    xcc, ycc = (
        np.atleast_2d(gwf.modelgrid.xcellcenters),
        np.atleast_2d(gwf.modelgrid.ycellcenters),
    )

    g = Gridgen(
        gwf.modelgrid,
        model_ws=new_d,
        exe_name=gg_bin,
    )
    g.build(verbose=True)
    gridprops = g.get_gridprops_disv()
    idomain = gwf.dis.idomain.array.copy()

    disv_idomain = []
    for k in range(gwf.dis.nlay.data):
        disv_idomain.append(idomain[k].flatten())
    gwf.remove_package("dis")
    disv = flopy.mf6.ModflowGwfdisv(gwf, idomain=disv_idomain, **gridprops)

    disv.write()

    sfr = gwf.get_package("sfr")
    sfr_pdata = pd.DataFrame.from_records(sfr.packagedata.array)
    print(sfr_pdata)
    sfr_pdata.loc[:, "cellid"] = sfr_pdata.cellid.apply(
        lambda x: (x[0], x[1] * ncol + x[2])
    )
    sfr.packagedata = sfr_pdata.to_records(index=False)
    sfr.write()

    wel = gwf.get_package("wel")
    if wel is not None:
        f_wel = open(os.path.join(new_d, "freyberg6_disv.wel"), "w")
        f_wel.write(
            "begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n"
        )

        f_wel.write(
            f"begin dimensions\nmaxbound {wel.stress_period_data.data[0].shape[0]}\nend dimensions\n\n"
        )

        for kper in range(sim.tdis.nper.data):
            if kper not in wel.stress_period_data.data:
                continue
            rarray = wel.stress_period_data.data[kper]
            print(rarray.dtype)
            xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
            ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
            ilay = [cid[0] for cid in rarray.cellid]
            xys = [(x, y) for x, y in zip(xs, ys)]

            # use zero for the layer so that we get the cell2d value back
            inodes = [g.intersect([xy], "point", 0)[0][0] for xy, il in zip(xys, ilay)]
            f_wel.write(f"begin period {kper + 1}\n")
            [
                f_wel.write(f"{il + 1:9d} {inode + 1:9d} {q:15.6E}\n")
                for il, inode, q in zip(ilay, inodes, rarray.q)
            ]
            f_wel.write(f"end period {kper + 1}\n\n")
        f_wel.close()

    ghb = gwf.get_package("ghb")
    if ghb is not None:
        f_ghb = open(os.path.join(new_d, "freyberg6_disv.ghb"), "w")
        f_ghb.write(
            "begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n"
        )

        f_ghb.write(
            f"begin dimensions\nmaxbound {ghb.stress_period_data.data[0].shape[0]}\nend dimensions\n\n"
        )

        ghb_spd = {}
        for kper in range(sim.tdis.nper.data):
            if kper not in ghb.stress_period_data.data:
                continue
            rarray = ghb.stress_period_data.data[kper]

            xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
            ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
            ilay = [cid[0] for cid in rarray.cellid]
            xys = [(x, y) for x, y in zip(xs, ys)]

            # use zero for the layer so that we get the cell2d value back
            inodes = [g.intersect([xy], "point", 0)[0][0] for xy, il in zip(xys, ilay)]
            data = [
                [(il, inode), bhead, cond]
                for il, inode, bhead, cond in zip(
                    ilay, inodes, rarray.bhead, rarray.cond
                )
            ]
            ghb_spd[kper] = data
            f_ghb.write(f"begin period {kper + 1}\n")
            [
                f_ghb.write(f"{il + 1:9d} {inode + 1:9d} {bhead:15.6E} {cond:15.6E}\n")
                for il, inode, bhead, cond in zip(
                    ilay, inodes, rarray.bhead, rarray.cond
                )
            ]
            f_ghb.write(f"end period {kper + 1}\n\n")
        f_ghb.close()

    df = pd.read_csv(
        os.path.join(new_d, "head.obs"),
        skipfooter=1,
        skiprows=1,
        header=None,
        names=["site", "otype", "l", "r", "c"],
        sep=r"\s+",
    )
    df.loc[:, "node"] = df.apply(lambda x: (int(x.r) * ncol) + int(x.c), axis=1)
    with open(os.path.join(new_d, "head.obs"), "w") as f:
        f.write("BEGIN CONTINUOUS FILEOUT heads.csv\n")
        for site, otype, lay, node in zip(df.site, df.otype, df.l, df.node):
            f.write(f"{site} {otype} {lay} {node}\n")
        f.write("END CONTINUOUS")

    # now hack the nam file
    nam_file = os.path.join(new_d, "freyberg6.nam")
    lines = open(nam_file, "r").readlines()

    with open(nam_file, "w") as f:
        for line in lines:
            if "dis" in line.lower():
                line = "DISV6 freyberg6.disv disv\n"
            elif "ghb" in line.lower():
                line = "GHB6 freyberg6_disv.ghb ghb\n"
            elif "wel" in line.lower():
                line = "WEL6 freyberg6_disv.wel wel\n"
            f.write(line)

    pyemu.os_utils.run("mf6", cwd=new_d)

    laynode = []
    k_dict = {}
    with open(os.path.join(new_d, "head.obs"), "r") as f:
        f.readline()
        for line in f:
            if line.strip().lower().startswith("end"):
                break
            raw = line.strip().split()
            laynode.append(" ".join(raw[2:]))

            k = int(raw[2]) - 1
            inode = int(raw[3]) - 1
            if k not in k_dict:
                k_dict[k] = []
            k_dict[k].append([inode])

    np.random.seed(11111)
    rvals = np.random.random(len(laynode)) + 36
    with open(os.path.join(new_d, "test.adj"), "w") as f:
        f.write("begin performance_measure pm1\n")
        for rval, ln in zip(rvals, laynode):
            for kper in range(25):
                f.write(f"{kper + 1} 1 {ln} head direct 1.0 {rval}\n")
        f.write("end performance_measure\n\n")

    start = datetime.now()
    os.chdir(new_d)
    adj = mf6adj.Mf6Adj("test.adj", lib_name, verbose_level=2)
    adj.solve_gwf()
    adj.solve_adjoint()
    adj.finalize()
    os.chdir("..")
    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)

    result_hdf = [
        f
        for f in os.listdir(new_d)
        if f.endswith("hd5") and f.startswith("adjoint_solution_pm")
    ]
    print(result_hdf)
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]

    hdf = h5py.File(os.path.join(new_d, result_hdf), "r")
    keys = list(hdf.keys())
    keys.sort()
    print(keys)

    idomain = np.loadtxt(os.path.join(new_d, "freyberg6.dis_idomain_layer1.txt"))
    with PdfPages(os.path.join(new_d, "results.pdf")) as pdf:
        for key in keys:
            if key != "composite":
                continue

            grp = hdf[key]
            plot_keys = [i for i in grp.keys() if len(grp[i]) == nlay * nrow * ncol]
            for pkey in plot_keys:
                arr = grp[pkey][:].reshape((nlay, nrow, ncol))
                for k, karr in enumerate(arr):
                    karr[idomain < 1] = np.nan
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    cb = ax.imshow(karr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(key + ", " + pkey + f", layer:{k + 1}", loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...", key, pkey, k + 1)
