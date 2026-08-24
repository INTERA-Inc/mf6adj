"""Recompute the capture-fraction archive the supplemental document plots.

Runs the synthetic valley model and the adjoint solution once and writes a small
archive of the quantities the figures need. The archive is committed beside the
document, so this only has to be run when the results themselves change.

The synthetic valley model and the helper that prepares it are distributed with
the MODFLOW 6 training repository rather than with mf6adj, so the path to a
clone of it is given on the command line:

    python make_summary.py <mf6-training clone> <archive> <workspace>
"""

import pathlib as pl
import sys

import flopy
import h5py
import numpy as np

import mf6adj

TRAINING = pl.Path(sys.argv[1]).resolve()
ARCHIVE = pl.Path(sys.argv[2])
WORKSPACE = pl.Path(sys.argv[3])

sys.path.insert(0, str(TRAINING / "examples" / "notebooks"))
import mf6_adj_helpers as helpers

helpers.DATA_ROOT = TRAINING / "examples" / "data" / "synthetic-valley"
helpers.MODEL_ROOT = WORKSPACE.parent

mf6_exe, lib_name = mf6adj.get_conda_mf6_paths()

print("preparing the synthetic valley model", flush=True)
ws = helpers.prepare_model(
    WORKSPACE.name, variant="advanced", sample_frequency="annual"
)
helpers.run_model(ws, mf6_exe)

sim = flopy.mf6.MFSimulation.load(sim_ws=str(ws), verbosity_level=0)
gwf = sim.get_model()
shape = gwf.dis.idomain.array.shape
print(f"grid {shape}", flush=True)

# the two measures the document reports: streamflow and lake capture
measures = {}
cell_index = {}
for name, package in (("sfr", "sfr-1"), ("lak", "lak-1")):
    cells = helpers.package_cells(gwf, package)
    nper = sim.tdis.nper.get_data()
    measures[name] = [(nper - 1, 0, k, i, j, package) for k, i, j in cells]
    cell_index[name] = np.array(cells, dtype=int)
    print(f"  {name}: {len(cells)} cells in {package}", flush=True)

helpers.write_adj_file(ws, "capture.adj", measures)

adj = mf6adj.Mf6Adj(
    "capture.adj", lib_name, logging_level="WARNING", working_directory=str(ws)
)
adj.solve_forward_model(hdf5_name="forward.hd5")
adj.solve_adjoint()
adj.finalize()

data = {}
for name in measures:
    with h5py.File(ws / f"adjoint_solution_{name}.hd5", "r") as hf:
        data[f"{name}_capture"] = -1.0 * hf["composite"]["wel6_q"][:]
    print(
        f"{name}: maximum capture {np.nanmax(data[f'{name}_capture']):.4f}",
        flush=True,
    )

for name, cells in cell_index.items():
    data[f"{name}_cells"] = cells
data["idomain"] = gwf.dis.idomain.array
data["delr"] = gwf.dis.delr.array
data["delc"] = gwf.dis.delc.array
data["shape"] = np.array(shape)

ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(ARCHIVE, **data)
print(f"wrote {ARCHIVE} ({ARCHIVE.stat().st_size / 1024:.0f} kB)", flush=True)
