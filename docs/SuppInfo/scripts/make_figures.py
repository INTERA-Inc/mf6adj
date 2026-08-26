"""Draw the figures for the supplemental technical information document.

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
        # a single column figure; the grid is taller than it is wide and the
        # aspect is equal, so the height follows the extent rather than padding
        fig, ax = plt.subplots(figsize=(3.5, 4.6), layout="constrained")
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


def draw_instantaneous(archive, outfile):
    """Compare an instantaneous measure with a direct one, step by step."""
    data = np.load(archive)
    direct = data["direct"]
    instantaneous = data["instantaneous"]
    steps = np.arange(1, direct.size + 1)
    # the first stress period is steady state, so its steps carry nothing on
    steady_end = int(data["nstp"][0])

    with styles.USGSPlot():
        fig, ax = plt.subplots(figsize=(3.5, 2.9), layout="constrained")
        ax.axvspan(0.5, steady_end + 0.5, color="0.90", zorder=0)
        ax.plot(steps, direct, "-", color="black", lw=1.2, label="direct")
        ax.plot(
            steps,
            instantaneous,
            "o",
            color="tab:red",
            ms=3.5,
            markerfacecolor="none",
            label="instantaneous",
        )
        styles.xlabel(ax=ax, label="Time step")
        styles.ylabel(
            ax=ax,
            label="Sensitivity of head to the\nwell rate, in days per square meter",
        )
        styles.add_text(
            ax=ax,
            x=0.06,
            y=0.10,
            text="steady\nstate",
            transform=True,
            fontsize=6,
            ha="left",
        )
        ax.set_xlim(0.5, direct.size + 0.5)
        ax.set_ylim(0.0, 1.35 * direct.max())
        styles.graph_legend(ax=ax, loc="upper right")
        styles.remove_edge_ticks(ax=ax)
        fig.savefig(outfile)
        plt.close(fig)

    agree = np.isclose(direct, instantaneous, rtol=1e-9)
    print(f"wrote {outfile.name} ({int(agree.sum())} of {agree.size} steps agree)")


draw_instantaneous(
    ARCHIVE.parent / "instantaneous-comparison.npz", OUTDIR / "instantaneous.pdf"
)
