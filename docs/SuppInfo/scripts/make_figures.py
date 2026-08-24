"""Draw the capture-fraction maps for the supplemental document.

Reads the archive written beside the document rather than the model output, so
the figures redraw from a clone without running a simulation.
"""

import pathlib as pl
import sys

import flopy
import flopy.plot.styles as styles
import matplotlib.pyplot as plt
import numpy as np

ARCHIVE = pl.Path(sys.argv[1])
OUTDIR = pl.Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)

data = np.load(ARCHIVE)
nlay, nrow, ncol = (int(v) for v in data["shape"])
idomain = data["idomain"]
grid = flopy.discretization.StructuredGrid(
    delr=data["delr"],
    delc=data["delc"],
    nlay=nlay,
    nrow=nrow,
    ncol=ncol,
    idomain=idomain,
)

# EPS is a possible output, and flopy sets only the pdf font type
plt.rcParams["ps.fonttype"] = 42


def capture_layer(values):
    """Return the layer carrying the most capture, and that layer's map."""
    arr = np.array(values, dtype=float).reshape((nlay, nrow, ncol))
    arr[idomain <= 0] = np.nan
    totals = [np.nansum(np.abs(arr[k])) for k in range(nlay)]
    k = int(np.argmax(totals))
    return k, arr[k]


# cell centers follow from the cell sizes, so the archive does not have to
# carry the vertical discretization the grid would otherwise need
DELR, DELC = data["delr"], data["delc"]
XC = np.cumsum(DELR) - DELR / 2.0
YC = DELC.sum() - (np.cumsum(DELC) - DELC / 2.0)


def cell_centers(cells):
    """Return the map coordinates of the cells a package occupies."""
    return XC[cells[:, 2]], YC[cells[:, 1]]


def draw(values, cells, boundary_label, bar_label, outfile):
    k, layer = capture_layer(values)
    vmax = float(np.nanpercentile(np.abs(layer[np.isfinite(layer)]), 99.5))

    with styles.USGSMap():
        # the grid is taller than it is wide and the aspect is equal, so the
        # panel is sized to the extent rather than left to pad
        fig, ax = plt.subplots(figsize=(4.2, 5.0), layout="constrained")
        mv = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=k)
        img = mv.plot_array(layer, cmap="viridis", vmin=0.0, vmax=vmax)
        mv.plot_inactive(color_noflow="0.85")
        x, y = cell_centers(cells)
        ax.plot(
            x,
            y,
            "o",
            markersize=2.5,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=boundary_label,
            linestyle="none",
        )
        ax.set_aspect("equal")
        styles.xlabel(ax=ax, label="x position, in meters")
        styles.ylabel(ax=ax, label="y position, in meters")
        styles.graph_legend(ax=ax, loc="upper right")
        cbar = fig.colorbar(img, ax=ax, shrink=0.55)
        cbar.set_label(bar_label)
        styles.remove_edge_ticks(ax=ax)
        fig.savefig(outfile)
        plt.close(fig)
    print(
        f"wrote {outfile.name} (layer {k + 1}, maximum plotted {vmax:.3f})", flush=True
    )


if "sfr_capture" in data:
    draw(
        data["sfr_capture"],
        data["sfr_cells"],
        "Stream reach",
        "Streamflow capture fraction, dimensionless",
        OUTDIR / "sfr-capture.pdf",
    )
if "lak_capture" in data:
    draw(
        data["lak_capture"],
        data["lak_cells"],
        "Lake cell",
        "Lake capture fraction, dimensionless",
        OUTDIR / "lake-capture.pdf",
    )
