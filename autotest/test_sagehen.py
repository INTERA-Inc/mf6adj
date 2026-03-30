import os
import pathlib as pl
import platform
import shutil
import sys
from datetime import datetime

import flopy
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

try:
    import mf6adj
except ImportError:
    sys.path.insert(0, str(pl.Path("../").resolve()))
    import mf6adj

mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()


def test_sagehen():
    prep = True

    org_d = "ex-gwf-sagehen-external"
    new_d = "sagehen_test"

    adj_file = pl.Path(new_d) / "test.adj"
    if prep:
        if pl.Path(new_d).exists():
            shutil.rmtree(new_d)
        shutil.copytree(org_d, new_d)
        flopy.run_model(exe_name=mf6_bin, namefile=None, model_ws=new_d)

    sim = flopy.mf6.MFSimulation.load(sim_ws=new_d, load_only=["dis", "sfr"])
    gwf = sim.get_model()

    with open(adj_file, "w") as f:
        sfr_data = pd.DataFrame.from_records(gwf.sfr.packagedata.array)
        f.write("begin performance_measure swgw\n")
        for kper in range(sim.tdis.nper.data):
            for kij in sfr_data.cellid.values:
                f.write(
                    f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} "
                    + "sfr-1 direct 1.0 -1.0e+30\n"
                )
        f.write("end performance_measure\n\n")

        # now a direct head pm at the terminal sfr reach (the last kij covered)
        f.write("begin performance_measure terminalhead\n")
        for kper in range(sim.tdis.nper.data):
            f.write(
                f"{kper + 1} 1 {kij[0] + 1} {kij[1] + 1} {kij[2] + 1} "
                + "head direct 1.0 -1.0e+30\n"
            )
        f.write("end performance_measure\n")

    start = datetime.now()

    print("calculating adjoint...")
    adj = mf6adj.Mf6Adj(
        adj_file.name,
        lib_name,
        logging_level="INFO",
        working_directory=new_d,
    )
    adj.solve_forward_model()
    adj.solve_adjoint()
    adj.finalize()

    duration = (datetime.now() - start).total_seconds()
    print("took:", duration)

    result_hdf = [
        f
        for f in os.listdir(new_d)
        if f.endswith("hd5") and f.startswith("adjoint_solution_swgw")
    ]
    assert len(result_hdf) == 1
    result_hdf = result_hdf[0]

    hdf = h5py.File(pl.Path(new_d) / result_hdf, "r")
    keys = list(hdf.keys())
    keys.sort()

    nlay, nrow, ncol = gwf.dis.nlay.data, gwf.dis.nrow.data, gwf.dis.ncol.data

    idomain = gwf.dis.idomain.array
    thresh = 0.0001
    with PdfPages(pl.Path(new_d) / "results.pdf") as pdf:
        for key in keys:
            if key != "composite":
                continue
            grp = hdf[key]

            plot_keys = [i for i in grp.keys() if grp[i].shape == (nlay, nrow, ncol)]

            for pkey in plot_keys:
                arr = grp[pkey][:].reshape((nlay, nrow, ncol))
                for k, karr in enumerate(arr):
                    karr[idomain[k, :, :] < 1] = np.nan
                    ib = idomain[k, :, :].copy().astype(float)
                    ib[ib > 0] = np.nan
                    # karr[np.abs(karr)>1e20] = np.nan
                    karr[np.abs(karr) < thresh] = np.nan
                    # karr = np.log10(karr)
                    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                    ax.imshow(ib, cmap="Greys_r")
                    cb = ax.imshow(karr)
                    plt.colorbar(cb, ax=ax)
                    ax.set_title(
                        key
                        + ", "
                        + pkey
                        + f", layer:{k + 1}, masked where abs < {thresh}",
                        loc="left",
                    )
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close(fig)
                    print("...", key, pkey, k + 1)
