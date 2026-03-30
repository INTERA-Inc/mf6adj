import os
import pathlib as pl
import platform
import shutil
import sys

import flopy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flopy.utils.gridgen import Gridgen
from matplotlib.backends.backend_pdf import PdfPages

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()
gg_bin = mf6_bin.parent / f"gridgen{mf6_bin.suffix}"
if not gg_bin.is_file():
    flopy.utils.get_modflow(":python", subset="gridgen")


# fmt: off
def setup_xd_box_model(
    new_d,
    sp_len=1.0, nper=1,
    hk=1.0, k33=1.0,
    q=-0.1,
    ss=1.0e-5,
    nlay=1, nrow=10, ncol=10,
    delr=1.0, delc=1.0,
    icelltype=0, iconvert=0,
    newton=False,
    top=1, botm=None,
    include_sto=True, include_id0=True,
    name="xdbox",
    full_sat_bnd=True,
    alt_bnd=None,
    is_anatest=False,
):
    tdis_pd = [(sp_len, 1, 1.0) for _ in range(nper)]
    new_d = pl.Path(new_d)
    if botm is None:
        botm = np.linspace(0, -nlay + 1, nlay)
    if new_d.exists():
        shutil.rmtree(new_d)

    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=mf6_bin,
        version="mf6",
        sim_ws=new_d,
        memory_print_option="ALL",
        continue_=True,
    )

    flopy.mf6.ModflowTdis(
        sim, pname="tdis", time_units="DAYS", nper=len(tdis_pd), perioddata=tdis_pd,
    )

    flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        inner_dvclose=1e-8,
        outer_dvclose=1e-8,
        outer_maximum=1000,
        inner_maximum=1000,
    )

    model_nam_file = f"{name}.nam"
    newtonoptions = []
    if newton:
        newtonoptions = ["NEWTON"]
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=model_nam_file,
        save_flows=True,
        newtonoptions=newtonoptions,
    )

    idm = np.ones((nlay, nrow, ncol))
    if ncol > 1 and nrow > 1 and include_id0:
        for k in range(nlay):
            idm[k, 0, 1] = 0  # just to have one in active cell...
    if ncol > 5 and nrow > 5 and include_id0:
        for k in range(nlay):
            idm[0, 1, 1] = 0
            idm[0, 3, 3] = 0

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idm,
    )

    # ### Create the initial conditions (`IC`) Package
    start = top * np.ones((nlay, nrow, ncol))
    if is_anatest:
        L1 = delr * ncol
        H1 = 1001.
        H2 = 1000.
        start = np.ones((nlay, nrow, ncol))
        for kk in range(nlay):
            for j in range(nrow):
                for i in range(ncol):
                    start[kk][j][i] = (
                        H1 + (H2 - H1) * gwf.modelgrid.xcellcenters[j][i] / L1
                    )
                    # start[kk][j][i] = H2
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)

    # ### Create the storage (`STO`) Package
    if include_sto:
        sy = []
        geo = [top]
        geo.extend(botm)
        for k in range(nlay):
            t, b = geo[k], geo[k + 1]
            if is_anatest:
                sy.append(0.15)
            else:
                sy.append(ss * (t - b))
        steady_state = [False]
        transient = [True]
        if len(tdis_pd) > 1:
            steady_state = dict.fromkeys(range(len(tdis_pd)), False)
            steady_state[0] = True
            transient = dict.fromkeys(range(len(tdis_pd)), True)
            transient[0] = False

        flopy.mf6.ModflowGwfsto(
            gwf,
            iconvert=iconvert,
            steady_state=steady_state,
            transient=transient,
            ss=ss,
            sy=sy,
        )

    alt_rec = []
    bnd_nper = np.arange(nper, dtype=int)
    if nper > 2:
        bnd_nper = np.arange(1, nper, dtype=int)
    if nper > 3:
        bnd_nper = np.arange(1, nper - 1, dtype=int)
    if ncol > 1:
        stage = top
        if not full_sat_bnd:
            if is_anatest:
                stage = botm[0] + ((top-botm[0])/4.0)
                stageL = 1001.0
                stageR = 1000.0
            else:
                stage = botm[0] + ((top - botm[0]) / 4.0)

        for k in [nlay - 1]:
            for i in range(nrow):
                if alt_bnd == "riv":
                    alt_rec.append(((k, i, 0), stage, 1000.0, botm[0]))
                elif alt_bnd == "drn":
                    alt_rec.append(((k, i, 0), stage, 1000.0))
                elif alt_bnd == "chd":
                    if is_anatest:
                        alt_rec.append(((k, i, 0), stageL))
                        alt_rec.append(((k, i, ncol-1), stageR)) 
                    else:
                        alt_rec.append(((k, i, 0), stage))
                elif alt_bnd is None:
                    alt_rec.append(((k, i, 0), stage, 1000.0))
                else:
                    raise Exception()

        if alt_bnd == "riv":
            _ = flopy.mf6.ModflowGwfriv(
                gwf, stress_period_data=dict.fromkeys(bnd_nper, alt_rec)
            )
        elif alt_bnd == "drn":
            _ = flopy.mf6.ModflowGwfdrn(
                gwf, stress_period_data=dict.fromkeys(bnd_nper, alt_rec)
            )
        elif alt_bnd == "chd":
            _ = flopy.mf6.ModflowGwfchd(
                gwf, stress_period_data=dict.fromkeys(bnd_nper, alt_rec)
            )
        elif alt_bnd is None:
            pass
        else:
            raise Exception()

    if not is_anatest:
        ghb_rec = []
        ghb_stage = top + 1
        if not full_sat_bnd:
            ghb_stage = botm[0] + (3.0 * (top - botm[0]) / 4.0)
        for k in [0]:
            for i in range(nrow):
                ghb_rec.append(((k, i, ncol - 1), ghb_stage, 1000.0))
        ghb_spd = dict.fromkeys(range(nper), ghb_rec)
        if alt_bnd is None:
            ghb_spd.update(dict.fromkeys(bnd_nper, alt_rec))
        flopy.mf6.ModflowGwfghb(gwf, stress_period_data=ghb_spd)

    wspd = {}
    start_well = 1
    if nper == 1:
        start_well = 0
    for kper in range(start_well, nper):
        wel_rec = []
        for k in range(nlay):
            for i in range(nrow):
                for j in range(ncol):
                    if idm[k, i, j] <= 0:
                        continue
                    wel_rec.append([(k, i, j), q * (kper + 1)])
        wspd[kper] = wel_rec
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=wspd)

    headfile = f"{name}.hds"
    head_filerecord = [headfile]
    budgetfile = f"{name}.cbb"
    budget_filerecord = [budgetfile]
    saverecord = {kper: [("HEAD", "ALL"), ("BUDGET", "ALL")] for kper in range(nper)}
    printrecord = {kper: [("HEAD", "ALL")] for kper in range(nper)}
    flopy.mf6.ModflowGwfoc(
        gwf,
        saverecord=saverecord,
        head_filerecord=head_filerecord,
        budget_filerecord=budget_filerecord,
        printrecord=printrecord,
    )

    flopy.mf6.ModflowGwfnpf(gwf, icelltype=icelltype, k=hk, k33=k33)

    if not is_anatest:
        rech = {kper: np.zeros((nrow, ncol)) + 0.0000001 for kper in range(nper)}
        flopy.mf6.ModflowGwfrcha(gwf, recharge=rech)

    # Write the datasets and run to make sure it works
    sim.write_simulation()

    # hack in tvk so we can use pert within api later
    with open(new_d / "blank.tvk", "w") as f:
        f.write("\n")

    sim.run_simulation()
    return sim


# fmt: off
def xd_box_compare(new_d, plot_compare=False, dif_thres=1e-6,):
    new_d = pl.Path(new_d)
    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)
    gwf = sim.get_model()
    id = gwf.dis.idomain.array

    adj_summary_files = [
        str(new_d / f)
        for f in os.listdir(new_d)
        if f.startswith("adjoint_solution") and f.endswith(".csv")
    ]

    assert len(adj_summary_files) > 0

    pert_summary = pd.read_csv(new_d / "pert_results.csv", index_col=0)
    for col in ["k", "i", "j"]:
        if col in pert_summary:
            pert_summary[col] = pert_summary[col].astype(int)
    if plot_compare:
        pdf = PdfPages(new_d / "compare.pdf")

    skip = ["epsilon", "k", "i", "j", "addr"]
    pm_names = [c for c in pert_summary.columns if c not in skip]
    pm_names.sort()
    results_dict = {}
    for pm_name in pm_names:
        adj_file = [f for f in adj_summary_files if pm_name in f]
        if len(adj_file) != 1:
            print(pm_name, adj_file)
            raise Exception()
        adj_file = adj_file[0]
        # print(adj_file,pm_name)
        adj = pd.read_csv(adj_file, index_col=0)
        adj_cols = adj.columns.tolist()
        adj_cols.sort()

        for col in adj_cols:
            # try to find the values in addr
            pertdf = pert_summary.loc[pert_summary.addr.str.contains(col), :].copy()
            if pertdf.shape[0] == 0:
                print(
                    f"...WARNING: no values to compare for pm {pm_name} " + 
                    f"and parameter {col}"
                )
                continue
            # print(pm_name,col)
            adjdf = adj.loc[pertdf.index.values, col].copy()

            dif = pertdf.loc[:, pm_name].values - adjdf.values
            print(pm_name, col, pertdf.shape[0], adjdf.shape[0])

            demon = pertdf[pm_name].max() - pertdf[pm_name].min()

            if demon < 0.0:
                raise Exception()
            elif demon == 0:
                pertdf[pm_name].max()
                # demon = plt_zero_thres

            if demon == 0:
                demon = 1.0
            absdif = np.abs(dif)
            abs_max_dif_percent = 100.0 * absdif / demon
            if np.nanmax(absdif) <= dif_thres:
                abs_max_dif_percent = 0.0

            print("...min vals:", pertdf[pm_name].values.min(), adjdf.values.min())
            print("...max vals:", pertdf[pm_name].values.max(), adjdf.values.max())
            print(
                "...dif vals",
                dif.min(),
                dif.max(),
                np.abs(dif).min(),
                np.abs(dif).max(),
            )
            print("...abs dif max:", np.nanmax(absdif))
            print("...abs max % dif:", np.nanmax(abs_max_dif_percent))
            results_dict[(pm_name, col)] = np.nanmax(abs_max_dif_percent)
            # if not np.isinf(np.nanmax(abs_max_dif_percent)):
            #    assert np.nanmax(abs_max_dif_percent) < 10.0 #?
            if plot_compare and "i" in pertdf.columns and "j" in pertdf.columns:
                kvals = pertdf.k.unique()
                kvals.sort()
                pertdf.loc[:, "dif"] = dif
                pertdf.loc[:, "absdif"] = abs_max_dif_percent
                for k in kvals:
                    kpertdf = pertdf.loc[pertdf.k == k, :]
                    kadjdf = adjdf.loc[kpertdf.index]

                    adj_arr = np.zeros((gwf.dis.nrow.data, gwf.dis.ncol.data)) - 1e30
                    pert_arr = np.zeros_like(adj_arr) - 1e30
                    adj_arr[kpertdf.i, kpertdf.j] = kadjdf.values
                    pert_arr[kpertdf.i, kpertdf.j] = kpertdf[pm_name].values
                    dif_arr = np.zeros_like(adj_arr) - 1.0e30
                    dif_arr[kpertdf.i, kpertdf.j] = kpertdf.dif.values

                    abs_arr = np.zeros_like(adj_arr) - 1.0e30
                    abs_arr[kpertdf.i, kpertdf.j] = kpertdf.absdif.values

                    for arr in [adj_arr, pert_arr, dif_arr, abs_arr]:
                        arr[arr == -1e30] = np.nan

                    fig, axes = plt.subplots(1, 4, figsize=(12, 2))

                    absd = np.abs(dif_arr)
                    absp = np.abs(abs_arr)

                    # absd[absd<plt_zero_thres] = 0
                    # pert_arr[absd==0] = np.nan
                    # pm_arr[absd==0] = np.nan
                    dif_arr[absd < dif_thres] = np.nan
                    abs_arr[absd < dif_thres] = np.nan
                    mx = max(np.nanmax(pert_arr), np.nanmax(adj_arr))
                    mn = min(np.nanmin(pert_arr), np.nanmin(adj_arr))
                    cb = axes[0].imshow(pert_arr, vmax=mx, vmin=mn)
                    plt.colorbar(cb, ax=axes[0])
                    axes[0].set_title(
                        pm_name + " " + col + " pert k:" + str(k), loc="left"
                    )

                    cb = axes[1].imshow(adj_arr, vmax=mx, vmin=mn)
                    plt.colorbar(cb, ax=axes[1])
                    axes[1].set_title(
                        pm_name + " " + col + " adj k:" + str(k), loc="left"
                    )
                    mx = np.nanmax(absd)

                    cb = axes[2].imshow(dif_arr, vmin=-mx, vmax=mx, cmap="coolwarm")
                    plt.colorbar(cb, ax=axes[2])
                    axes[2].set_title(
                        f"pert - adj, not showing abs(diff) <= {dif_thres}",
                        loc="left",
                    )
                    mx = np.nanmax(absp)
                    cb = axes[3].imshow(abs_arr, vmin=-mx, vmax=mx, cmap="coolwarm")
                    plt.colorbar(cb, ax=axes[3])
                    axes[3].set_title(
                        f"percent diff pert - adj, max:{np.nanmax(abs_arr):4.2g}",
                        loc="left",
                    )
                    if np.any(id == 0):
                        idp = id[k, :, :].copy().astype(float)
                        idp[idp != 0] = np.nan
                        axes[0].imshow(idp, cmap="magma")
                        axes[1].imshow(idp, cmap="magma")
                    for ax in axes.flatten():
                        ax.set_xticks([])
                        ax.set_yticks([])
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)

    df = pd.Series(results_dict).reset_index()
    df = df.pivot(columns="level_0", index="level_1", values=0)
    df.sort_index(inplace=True)
    cols = df.columns.values
    cols.sort()
    df = df.loc[:, cols]
    print(df)

    if plot_compare:
        pdf.close()
        fig, ax = plt.subplots(1, 1, figsize=list(df.shape[::-1]))
        mx = np.log10(df.values).max()
        cb = ax.imshow(np.log10(df.values), vmax=mx, vmin=0, cmap="plasma", alpha=0.5)
        for i, row in enumerate(df.index):
            for j, col in enumerate(df.columns):
                v = df.loc[row, col]
                if v < 1.0:
                    pass
                else:
                    ax.text(
                        j,
                        i,
                        f"{v:5.0f}%",
                        va="center",
                        ha="center",
                        fontsize=8,
                    )
        plt.colorbar(cb, ax=ax, label="log_10 percent abs diff")
        ax.set_xticks(np.arange(df.shape[1]))
        ax.set_xticklabels(df.columns.values, rotation=90, fontsize=8)
        ax.set_yticks(np.arange(df.shape[0]))
        ax.set_yticklabels(df.index.values, fontsize=8)
        plt.tight_layout()
        plt.savefig(new_d / "abs_percent_dif_results.pdf")
        plt.close(fig)

    return df


def test_xd_box():
    """
    permutations:
     - w and w/o sto
     - icelltype (including spatially varying)
     - inconvert (including spatially varying)
     - 2d/3d
     - nrow == 1 or not
     - ncol == 1 or not
     - has/has not id 0
     - spatially varying props
     - multiple kper (including mix of ss and tr, stresses turning on and off, 
       different lengths, with and without multiple timesteps)
     - newton/nonnewton
     - newton with dry cells/upstream weighting - possibly with turning off pumping well
     - unit/nonunit delrowcol
     - spatially varying top and/or botm
     - unstructured (pass thru gridgen with no refinement)
     - multiple instances of the same package type

    Todo:
     - add bcs to pert
     - setup assertions in compare
     - add trap for CHDs and warn...

    """
    # workflow flags
    include_id0 = True  # include idomain = 0 cells
    include_sto = True

    include_ghb_flux_pm = True

    clean = True

    run_adj = True
    plot_adj_results = False  # plot adj result

    plot_compare = False
    new_d = pl.Path("xd_box_1_test")
    nrow = 5
    ncol = 5
    nlay = 3
    nper = 3
    sp_len = 10
    delr = 10.0
    delc = 10.0
    botm = [-10, -100, -1000]
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-3,
            icelltype=1,
            iconvert=1,
            newton=True,
            delr=delr,
            delc=delc,
            full_sat_bnd=False,
            botm=botm,
            alt_bnd="riv",
            sp_len=sp_len,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    pm_locs = []
    for k in range(nlay):  # [0, int(nlay / 2), nlay-1]:
        for i in range(nrow):  # [0, int(nrow / 2), nrow-1]:
            # for j in [0, int(ncol / 2), ncol-1]:
            pm_locs.append((k, i, i))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    if run_adj:
        print("calculating mf6adj sensitivity")

        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = f"direct_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                        f"head direct {weight} -1e+30\n"
                    )
                f.write("end performance_measure\n\n")

                pm_name = f"phi_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                        f"head residual {weight} {obsval}\n"
                    )
                f.write("end performance_measure\n\n")

            if include_ghb_flux_pm:
                for k in range(nlay):
                    pm_name = f"ghb_0_k{k}_direct"
                    lines = [f"begin performance_measure {pm_name}\n"]

                    for kper in range(sim.tdis.nper.data):
                        ghb = gwf.get_package("ghb_0").stress_period_data.array[kper]
                        if ghb is None:
                            continue
                        kijs = [g[0] for g in ghb if g[0][0] == k]

                        for k, i, j in kijs:
                            lines.append(
                                f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                                "ghb_0 direct 1.0 -1.0e+30\n"
                            )

                    lines.append("end performance_measure\n\n")
                if len(lines) > 2:
                    for line in lines:
                        f.write(line)

        adj = mf6adj.Mf6Adj(
            "test.adj", 
            lib_name, 
            logging_level="WARNING",
            working_directory=new_d,
            )
        adj.solve_forward_model()
        adj.solve_adjoint(csv_summary=True)
        adj.perturbation_method(pert_mult=pert_mult)
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [
                f
                for f in os.listdir(new_d)
                if (f.startswith("pm-direct") or f.startswith("pm-phi"))
                and f.endswith(".dat")
            ]
            afiles_to_plot.sort()

            with PdfPages(new_d / "adj.pdf") as pdf:
                for i, afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(new_d / afile))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(afile, loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile, i, len(afiles_to_plot))


    xd_box_compare(new_d, plot_compare)
    return

def test_xd_box_unstruct():
    # workflow flags
    include_id0 = (
        True  # include an idomain = cell - has to be false for unstructured...
    )
    include_sto = True

    clean = True  # run the pertbuation process
    # run_pert = False # the pertubations
    # plot_pert_results = True #plot the pertubation results

    run_adj = True
    plot_adj_results = False  # plot adj result

    # plot_compare = True

    new_d = pl.Path("xd_box_1_unstruct_test")
    nrow = ncol = 5
    nlay = 2
    nper = 2
    name = "xdbox"
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-0.1,
            icelltype=1,
            iconvert=0,
            newton=True,
            delr=100.0,
            delc=100,
            full_sat_bnd=False,
            name=name,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    obsval = 1.0

    pert_mult = 1.0001
    weight = 1.0
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

    wel = gwf.get_package("wel")
    if wel is not None:
        with open(new_d / f"{name}_disv.wel", "w") as f_wel:
            f_wel.write(
                "begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n"
            )
            mxbnd = -999
            for kper in range(sim.tdis.nper.data):
                if kper in wel.stress_period_data.data:
                    mxbnd = max(mxbnd, wel.stress_period_data.data[kper].shape[0])
            f_wel.write(f"begin dimensions\nmaxbound {mxbnd}\nend dimensions\n\n")

            for kper in range(sim.tdis.nper.data):
                if kper not in wel.stress_period_data.data:
                    continue
                rarray = wel.stress_period_data.data[kper]
                xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
                ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
                ilay = [cid[0] for cid in rarray.cellid]
                xys = [(x, y) for x, y in zip(xs, ys)]
                # use zero for the layer so that we get the cell2d value back
                inodes = [g.intersect([xy], "point", 0)[0][0] 
                          for xy, il in zip(xys, ilay)]
                f_wel.write(f"begin period {kper + 1}\n")
                for il, inode, q in zip(ilay, inodes, rarray.q):
                    f_wel.write(f"{il + 1:9d} {inode + 1:9d} {q:15.6E}\n")
                f_wel.write(f"end period {kper + 1}\n\n")

    ghb = gwf.get_package("ghb")
    if ghb is not None:
        with open(new_d / f"{name}_disv.ghb", "w") as f_ghb:
            f_ghb.write(
                "begin options\nprint_input\nprint_flows\nsave_flows\nend options\n\n"
            )
            f_ghb.write(
                "begin dimensions\nmaxbound "
                f"{ghb.stress_period_data.data[0].shape[0]}\nend dimensions\n\n"
            )
            for kper in range(sim.tdis.nper.data):
                rarray = ghb.stress_period_data.data[kper]
                xs = [xcc[cid[1], cid[2]] for cid in rarray.cellid]
                ys = [ycc[cid[1], cid[2]] for cid in rarray.cellid]
                ilay = [cid[0] for cid in rarray.cellid]
                xys = [(x, y) for x, y in zip(xs, ys)]
                # use zero for the layer so that we get the cell2d value back
                inodes = [g.intersect([xy], "point", 0)[0][0] 
                          for xy, il in zip(xys, ilay)]
                f_ghb.write(f"begin period {kper + 1}\n")
                for il, inode, bhead, cond in zip(
                    ilay, inodes, rarray.bhead, rarray.cond
                ):
                    f_ghb.write(
                        f"{il + 1:9d} {inode + 1:9d} {bhead:15.6E} {cond:15.6E}\n"
                    )
                f_ghb.write(f"end period {kper + 1}\n\n")

    # now hack the nam file
    nam_file = new_d / f"{name}.nam"
    with open(nam_file, "r") as f:
        lines = f.readlines()

    with open(nam_file, "w") as f:
        for line in lines:
            if "dis" in line.lower():
                line = f"DISV6 {name}.disv disv\n"
            elif "ghb" in line.lower():
                line = f"GHB6 {name}_disv.ghb ghb\n"
            elif "wel" in line.lower():
                line = f"WEL6 {name}_disv.wel wel\n"
            f.write(line)

    sim.run_simulation

    pm_locs = []
    for k in [0, int(nlay / 2), nlay - 1]:
        for inode in range(0, gwf.disv.ncpl.data, int(nrow / 2)):
            pm_locs.append((k, inode))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    if run_adj:
        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\n\nend options\n\n")
            for p_kinode in pm_locs:
                k, inode = p_kinode
                pm_name = f"direct_pk{k:03d}_pinode{inode:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {inode + 1} head direct " + 
                        f"{weight} -1.0e+30\n"
                    )
                f.write("end performance_measure\n\n")

                pm_name = f"phi_pk{k:03d}_pinode{inode:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {inode + 1} head direct " + 
                        f"{weight} {obsval}\n"
                    )
                f.write("end performance_measure\n\n")

        adj = mf6adj.Mf6Adj(
            "test.adj", 
            lib_name, 
            logging_level="WARNING",
            working_directory=new_d,
            )
        adj.solve_forward_model()
        adj.solve_adjoint(csv_summary=True)
        adj.perturbation_method(pert_mult=pert_mult)
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [
                f
                for f in os.listdir(new_d)
                if (f.startswith("pm-direct") or f.startswith("pm-phi"))
                and f.endswith(".dat")
            ]
            afiles_to_plot.sort()

            with PdfPages(new_d / "adj.pdf") as pdf:
                for i, afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(new_d / afile))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(afile, loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile, i, len(afiles_to_plot))

    xd_box_compare(new_d, False)

def test_xd_box_chd():
    # workflow flags
    include_id0 = False  # include idomain = 0 cells
    include_sto = False
    include_ghb_flux_pm = True

    clean = True
    run_adj = True
    plot_adj_results = False  # plot adj result

    plot_compare = False
    new_d = pl.Path("xd_box_chd_test")
    nrow = 1
    ncol = 3
    nlay = 1
    nper = 2
    sp_len = 10
    delr = 10.0
    delc = 10.0
    botm = [-10]  # ,-100,-1000]
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-3,
            icelltype=1,
            iconvert=1,
            newton=True,
            delr=delr,
            delc=delc,
            full_sat_bnd=False,
            botm=botm,
            alt_bnd="chd",
            sp_len=sp_len,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    pm_locs = []
    for k in range(nlay):  # [0, int(nlay / 2), nlay-1]:
        for i in range(nrow):  # [0, int(nrow / 2), nrow-1]:
            # for j in [0, int(ncol / 2), ncol-1]:
            pm_locs.append((k, i, i))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    if run_adj:
        print("calculating mf6adj sensitivity")

        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = f"direct_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head direct " + 
                        f"{weight} -1e+30\n"
                    )
                f.write("end performance_measure\n\n")

                pm_name = f"phi_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head residual " + 
                        f"{weight} {obsval}\n"
                    )
                f.write("end performance_measure\n\n")

            if include_ghb_flux_pm:
                for k in range(nlay):
                    pm_name = f"ghb_0_k{k}_direct"
                    lines = [f"begin performance_measure {pm_name}\n"]

                    for kper in range(sim.tdis.nper.data):
                        ghb = gwf.get_package("ghb_0").stress_period_data.array[kper]
                        if ghb is None:
                            continue

                        kijs = [g[0] for g in ghb if g[0][0] == k]

                        for k, i, j in kijs:
                            lines.append(
                                f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                                "ghb_0 direct 1.0 -1.0e+30\n"
                            )

                    lines.append("end performance_measure\n\n")
                if len(lines) > 2:
                    for line in lines:
                        f.write(line)
        adj = mf6adj.Mf6Adj(
            "test.adj", 
            lib_name, 
            logging_level="WARNING",
            working_directory=new_d,
            )
        adj.solve_forward_model()
        adj.solve_adjoint(csv_summary=True)
        adj.perturbation_method(pert_mult=pert_mult)
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [
                f
                for f in os.listdir(new_d)
                if (f.startswith("pm-direct") or f.startswith("pm-phi"))
                and f.endswith(".dat")
            ]
            afiles_to_plot.sort()

            with PdfPages(new_d / "adj.pdf") as pdf:
                for i, afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(new_d / afile))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(afile, loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile, i, len(afiles_to_plot))

    xd_box_compare(new_d, plot_compare)
    return


def test_xd_box_chd_ana():
    # workflow flags
    include_id0 = False  # include idomain = 0 cells
    include_sto = True

    include_ghb_flux_pm = False

    clean = True

    run_adj = True
    plot_adj_results = False  # plot adj result

    plot_compare = False
    new_d = pl.Path("xd_box_chd_ana2_test")
    nrow = 80
    ncol = 100
    nlay = 1
    nper = 1
    sp_len = 1
    delr = 5.0
    delc = 5.0
    botm = [-1]  # ,-100]
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-0.0,
            hk=10.0,
            k33=10.0,
            icelltype=1,
            iconvert=0,
            newton=True,
            delr=delr,
            delc=delc,
            top=0,
            full_sat_bnd=False,
            botm=botm,
            alt_bnd="chd",
            sp_len=sp_len,
            is_anatest=True,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    pm_locs = []
    for k in range(nlay):
        for i in range(nrow):
            pm_locs.append((k, i, i + 2))

    pm_locs = [(0, int(nrow / 2), int(ncol / 5))]
    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0
    if run_adj:
        print("calculating mf6adj sensitivity")

        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = f"direct_pk{k:03d}_pi{i:03d}_pj{j:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head direct "
                        f"{weight} -1e+30\n"
                    )
                f.write("end performance_measure\n\n")
            
        adj = mf6adj.Mf6Adj(
            "test.adj",
            lib_name,
            logging_level="WARNING",
            working_directory=new_d,
        )
        adj.solve_forward_model()
        df_dict = adj.solve_adjoint(csv_summary=True)

        lamb = df_dict[pm_name]["wel6_q"]

        X = gwf.modelgrid.xcellcenters
        Y = gwf.modelgrid.ycellcenters
        
        lambana = np.loadtxt(pl.Path("testing_files") / "lamb_Analytical.txt")
        lambana = pd.Series(lambana)
        lamb = lamb.reindex(np.arange(nrow * ncol, dtype=int))
        lamb.loc[:] *= -1
        lambana.loc[pd.isna(lamb)] = np.nan

        arrana = lambana.values.reshape(nrow, ncol)
        arr = lamb.values.reshape(nrow, ncol)
        vmin = min(np.nanmin(arrana), np.nanmin(arr))
        vmax = max(np.nanmax(arrana), np.nanmax(arr))
        levels = np.linspace(vmin, vmax, 10)

        diff = arrana - arr
        # diff[np.abs(diff)<1] = np.nan
        diff[np.abs(arrana) < 1e-3] = np.nan
        diff[np.abs(arr) < 1e-3] = np.nan

        mx = np.nanmax(np.abs(diff))
        assert mx < 0.04

        _, axes = plt.subplots(1, 2, figsize=(8.5, 3))
        cb = axes[0].pcolormesh(X, Y, arrana, vmin=vmin, vmax=vmax)
        plt.colorbar(cb, ax=axes[0], label="adjoint state")
        axes[0].contour(
            X, Y, arrana, vmin=vmin, vmax=vmax, levels=levels,
            colors="k", linewidths=0.5, linestyles="-",
        )
        axes[0].set_title("A) analytical", loc="left")
        cb = axes[1].pcolormesh(X, Y, arr, vmin=vmin, vmax=vmax)
        plt.colorbar(cb, ax=axes[1], label="adjoint state")
        axes[1].contour(
            X, Y, arr, vmin=vmin, vmax=vmax, levels=levels,
            colors="k", linewidths=0.5, linestyles="-",
        )
        axes[1].set_title("B) MF6ADJ", loc="left")

        # cb = axes[2].pcolormesh(X,Y,diff,vmin=-mx,vmax=mx,cmap="coolwarm")
        # levels = np.linspace(-mx,mx,10)
        # plt.colorbar(cb,ax=axes[2],label="difference")
        # axes[2].contour(X,Y,diff,vmin=-mx,vmax=mx,levels=levels,
        #                colors="k",linewidths=0.5,linestyles="-")
        # axes[2].set_title("difference")

        for ax in axes:
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
        plt.tight_layout()
        plt.savefig(new_d / "compare.png", dpi=500)
        plt.close()


def test_xd_box_ss():
    # workflow flags
    include_id0 = True  # include idomain = 0 cells
    include_sto = False
    include_ghb_flux_pm = True

    clean = True
    run_adj = True
    plot_adj_results = False  # plot adj result

    plot_compare = False
    new_d = pl.Path("xd_box_ss_test")

    nrow = 5
    ncol = 5
    nlay = 3
    nper = 1
    sp_len = 10
    delr = 10.0
    delc = 10.0
    botm = [-10, -100, -1000]
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-3,
            icelltype=1,
            iconvert=1,
            newton=True,
            delr=delr,
            delc=delc,
            full_sat_bnd=False,
            botm=botm,
            alt_bnd="riv",
            sp_len=sp_len,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    pm_locs = []
    for k in range(nlay):  # [0, int(nlay / 2), nlay-1]:
        for i in range(nrow):  # [0, int(nrow / 2), nrow-1]:
            pm_locs.append((k, i, i))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    if run_adj:
        print("calculating mf6adj sensitivity")

        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = f"direct_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head direct " + 
                        f"{weight} -1e+30\n"
                    )
                f.write("end performance_measure\n\n")

                pm_name = f"phi_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head residual " + 
                        f"{weight} {obsval}\n"
                    )
                f.write("end performance_measure\n\n")

            if include_ghb_flux_pm:
                for k in range(nlay):
                    pm_name = f"ghb_0_k{k}_direct"
                    lines = [f"begin performance_measure {pm_name}\n"]

                    for kper in range(sim.tdis.nper.data):
                        ghb = gwf.get_package("ghb_0").stress_period_data.array[kper]
                        if ghb is None:
                            continue

                        kijs = [g[0] for g in ghb if g[0][0] == k]

                        for k, i, j in kijs:
                            lines.append(
                                f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                                "ghb_0 direct 1.0 -1.0e+30\n"
                            )

                    lines.append("end performance_measure\n\n")
                if len(lines) > 2:
                    for line in lines:
                        f.write(line)

        adj = mf6adj.Mf6Adj(
            "test.adj", 
            lib_name, 
            logging_level="WARNING",
            working_directory=new_d,
            )
        adj.solve_forward_model()
        adj.solve_adjoint(csv_summary=True)
        adj.perturbation_method(pert_mult=pert_mult)
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [
                f
                for f in os.listdir(new_d)
                if (f.startswith("pm-direct") or f.startswith("pm-phi"))
                and f.endswith(".dat")
            ]
            afiles_to_plot.sort()

            with PdfPages(new_d / "adj.pdf") as pdf:
                for i, afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(new_d / afile))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(afile, loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile, i, len(afiles_to_plot))

    xd_box_compare(new_d, plot_compare)
    return


def test_xd_box_drn():
    # workflow flags
    include_id0 = True  # include idomain = 0 cells
    include_sto = True

    include_ghb_flux_pm = True

    clean = True

    run_adj = True
    plot_adj_results = False  # plot adj result

    plot_compare = False
    new_d = pl.Path("xd_box_drn_test")
    nrow = 5
    ncol = 5
    nlay = 3
    nper = 3
    sp_len = 10
    delr = 10.0
    delc = 10.0
    botm = [-10, -100, -1000]
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-3,
            icelltype=1,
            iconvert=1,
            newton=True,
            delr=delr,
            delc=delc,
            full_sat_bnd=False,
            botm=botm,
            alt_bnd="drn",
            sp_len=sp_len,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    pm_locs = []
    for k in range(nlay):  # [0, int(nlay / 2), nlay-1]:
        for i in range(nrow):  # [0, int(nrow / 2), nrow-1]:
            pm_locs.append((k, i, i))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    if run_adj:
        print("calculating mf6adj sensitivity")

        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = f"direct_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head direct " + 
                        f"{weight} -1e+30\n"
                    )
                f.write("end performance_measure\n\n")

                pm_name = f"phi_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head residual " + 
                        f"{weight} {obsval}\n"
                    )
                f.write("end performance_measure\n\n")

            if include_ghb_flux_pm:
                for k in range(nlay):
                    pm_name = f"ghb_0_k{k}_direct"
                    lines = [f"begin performance_measure {pm_name}\n"]

                    for kper in range(sim.tdis.nper.data):
                        ghb = gwf.get_package("ghb_0").stress_period_data.array[kper]
                        if ghb is None:
                            continue

                        kijs = [g[0] for g in ghb if g[0][0] == k]

                        for k, i, j in kijs:
                            lines.append(
                                f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                                "ghb_0 direct 1.0 -1.0e+30\n"
                            )

                    lines.append("end performance_measure\n\n")
                if len(lines) > 2:
                    for line in lines:
                        f.write(line)

        adj = mf6adj.Mf6Adj(
            "test.adj", 
            lib_name, 
            logging_level="WARNING",
            working_directory=new_d,
            )
        adj.solve_forward_model()
        adj.solve_adjoint(csv_summary=True)
        adj.perturbation_method(pert_mult=pert_mult)
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [
                f
                for f in os.listdir(new_d)
                if (f.startswith("pm-direct") or f.startswith("pm-phi"))
                and f.endswith(".dat")
            ]
            afiles_to_plot.sort()

            with PdfPages(new_d / "adj.pdf") as pdf:
                for i, afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(new_d / afile))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(afile, loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile, i, len(afiles_to_plot))

    xd_box_compare(new_d, plot_compare)
    return

def test_xd_box_maw():
    """
    permutations:
     - w and w/o sto
     - icelltype (including spatially varying)
     - inconvert (including spatially varying)
     - 2d/3d
     - nrow == 1 or not
     - ncol == 1 or not
     - has/has not id 0
     - spatially varying props
     - multiple kper (including mix of ss and tr, stresses turning on and off, 
       different lengths, with and without multiple timesteps)
     - newton/nonnewton
     - newton with dry cells/upstream weighting - possibly with turning off pumping well
     - unit/nonunit delrowcol
     - spatially varying top and/or botm
     - unstructured (pass thru gridgen with no refinement)
     - multiple instances of the same package type

    Todo:
     - add bcs to pert
     - setup assertions in compare
     - add trap for CHDs and warn...

    """
    # workflow flags
    include_id0 = True  # include idomain = 0 cells
    include_sto = True

    include_ghb_flux_pm = True

    clean = True

    run_adj = True
    plot_adj_results = False  # plot adj result

    plot_compare = False
    new_d = pl.Path("xd_box_maw_test")
    nrow = 5
    ncol = 5
    nlay = 3
    nper = 3
    sp_len = 10
    delr = 10.0
    delc = 10.0
    botm = [-10, -100, -1000]
    maw_top = 0
    if clean:
        sim = setup_xd_box_model(
            new_d,
            nper=nper,
            include_sto=include_sto,
            include_id0=include_id0,
            nrow=nrow,
            ncol=ncol,
            nlay=nlay,
            q=-3,
            icelltype=1,
            iconvert=1,
            newton=True,
            delr=delr,
            delc=delc,
            full_sat_bnd=False,
            botm=botm,
            alt_bnd="riv",
            sp_len=sp_len,
        )
    else:
        sim = flopy.mf6.MFSimulation.load(sim_ws=new_d)

    gwf = sim.get_model()
    maw_pakdata = [[0, 1, botm[-1], maw_top, "THIEM", len(botm) - 1]]
    maw_conndata = [
        [0, i, (i + 1, 1, 1), -999, -999, -999.0, -999.0] for i in range(len(botm) - 1)
    ]
    maw_perioddata = {
        0: [[0, "STATUS", "INACTIVE"]],
        1: [[0, "STATUS", "ACTIVE"], [0, "RATE", -1.0]],
    }
    flopy.mf6.ModflowGwfmaw(
        gwf,
        nmawwells=len(maw_pakdata),
        packagedata=maw_pakdata,
        connectiondata=maw_conndata,
        perioddata=maw_perioddata,
    )
    sim.write_simulation()
    sim.run_simulation()
    
    id = gwf.dis.idomain.array
    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data
    obsval = 1.0

    pert_mult = 1.01
    weight = 1.0

    pm_locs = []
    for k in range(nlay):  # [0, int(nlay / 2), nlay-1]:
        for i in range(nrow):  # [0, int(nrow / 2), nrow-1]:
            # for j in [0, int(ncol / 2), ncol-1]:
            pm_locs.append((k, i, i))

    pm_locs = list(set(pm_locs))
    pm_locs.sort()

    assert len(pm_locs) > 0

    if run_adj:
        print("calculating mf6adj sensitivity")

        with open(new_d / "test.adj", "w") as f:
            f.write("\nbegin options\nhdf5_name out.h5\nend options\n\n")
            for p_kij in pm_locs:
                k, i, j = p_kij
                if id[k, i, j] <= 0:
                    continue
                pm_name = f"direct_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head direct " + 
                        f"{weight} -1e+30\n"
                    )
                f.write("end performance_measure\n\n")

                pm_name = f"phi_pk{0:03d}_pi{k:03d}_pj{i:03d}"
                f.write(f"begin performance_measure {pm_name}\n")
                for kper in range(sim.tdis.nper.data):
                    f.write(
                        f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} head residual " + 
                        f"{weight} {obsval}\n"
                    )
                f.write("end performance_measure\n\n")

            if include_ghb_flux_pm:
                for k in range(nlay):
                    pm_name = f"ghb_0_k{k}_direct"
                    lines = [f"begin performance_measure {pm_name}\n"]

                    for kper in range(sim.tdis.nper.data):
                        ghb = gwf.get_package("ghb_0").stress_period_data.array[kper]
                        if ghb is None:
                            continue

                        kijs = [g[0] for g in ghb if g[0][0] == k]

                        for k, i, j in kijs:
                            lines.append(
                                f"{kper + 1} 1 {k + 1} {i + 1} {j + 1} " + 
                                "ghb_0 direct 1.0 -1.0e+30\n"
                            )

                    lines.append("end performance_measure\n\n")
                if len(lines) > 2:
                    for line in lines:
                        f.write(line)

        adj = mf6adj.Mf6Adj(
            "test.adj", 
            lib_name, 
            logging_level="WARNING",
            working_directory=new_d,
            )
        adj.solve_forward_model()
        adj.solve_adjoint(csv_summary=True)
        adj.perturbation_method(pert_mult=pert_mult)
        adj.finalize()

        if plot_adj_results:
            afiles_to_plot = [
                f
                for f in os.listdir(new_d)
                if (f.startswith("pm-direct") or f.startswith("pm-phi"))
                and f.endswith(".dat")
            ]
            afiles_to_plot.sort()

            with PdfPages(new_d / "adj.pdf") as pdf:
                for i, afile in enumerate(afiles_to_plot):
                    arr = np.atleast_2d(np.loadtxt(new_d / afile))
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    cb = ax.imshow(arr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(afile, loc="left")
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print(afile, i, len(afiles_to_plot))

    xd_box_compare(new_d, plot_compare)
    return


def test_nested():
    org_d = "nested"
    new_d = pl.Path("nested_test")
    if new_d.exists():
        shutil.rmtree(new_d)
    shutil.copytree(org_d, new_d)

    with open(new_d / "test.adj", "w") as f:
        f.write("\nbegin options\n\nend options\n\n")
        f.write("begin performance_measure pm1\n")
        f.write("1 1 1 80 head direct 1.0 -1.0e30 \n")
        f.write("end performance_measure\n\n")

    adj = mf6adj.Mf6Adj(
        "test.adj",
        lib_name,
        logging_level="WARNING",
        working_directory=new_d,
    )
    adj.solve_forward_model()
    adjdf = adj.solve_adjoint(csv_summary=True)["pm1"]
    pertdf1 = adj.perturbation_method(pert_mult=1.1)
    adj.finalize()

    pertssdf = pertdf1.loc[pertdf1.addr.str.contains("ss"), "pm1"]
    diff = 100.0 * (adjdf["ss"] - pertssdf) / pertssdf
    print(diff[~np.isnan(diff)].shape)
    assert diff[~np.isnan(diff)].shape[0] == 107
    print(np.nanmax(np.abs(diff)))
    assert np.nanmax(np.abs(diff)) < 0.1



if __name__ == "__main__":
    test_nested()
