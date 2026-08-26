"""Draw the figures for the supplemental technical information document.

Reads the archive written beside the document rather than the model output, so
the figures redraw from a clone without running a simulation.
"""

import pathlib as pl
import sys

import flopy.plot.styles as styles
import matplotlib.pyplot as plt
import numpy as np

DATA = pl.Path(sys.argv[1])
OUTDIR = pl.Path(sys.argv[2])
OUTDIR.mkdir(parents=True, exist_ok=True)

# EPS is a possible output, and flopy sets only the pdf font type
plt.rcParams["ps.fonttype"] = 42


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


draw_instantaneous(DATA / "instantaneous-comparison.npz", OUTDIR / "instantaneous.pdf")


def draw_package(archive, boundary_label, bar_label, outfile):
    """Map the capture a lake or a stream takes, on the model it is tested on."""
    data = np.load(archive)
    nlay, nrow, ncol = (int(v) for v in data["shape"])
    delrc = float(data["delrc"][0])
    arr = np.array(data["capture"], dtype=float).reshape((nlay, nrow, ncol))
    totals = [np.nansum(np.abs(arr[k])) for k in range(nlay)]
    k = int(np.argmax(totals))
    layer = arr[k]
    vmax = float(np.nanpercentile(np.abs(layer[np.isfinite(layer)]), 99.5))

    centers = (np.arange(ncol) + 0.5) * delrc
    rows = (np.arange(nrow) + 0.5) * delrc
    cells = data["boundary_cells"]
    well = data["well_cell"]

    with styles.USGSMap():
        fig, ax = plt.subplots(figsize=(3.5, 3.2), layout="constrained")
        img = ax.pcolormesh(
            np.append(np.arange(ncol) * delrc, ncol * delrc),
            np.append(np.arange(nrow) * delrc, nrow * delrc),
            layer,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.plot(
            centers[cells[:, 2]],
            rows[cells[:, 1]],
            "o",
            markersize=3.5,
            markerfacecolor="none",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            label=boundary_label,
        )
        ax.plot(
            centers[well[2]],
            rows[well[1]],
            "s",
            markersize=5,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=1.0,
            linestyle="none",
            label="Pumped cell",
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        styles.xlabel(ax=ax, label="x position, in meters")
        styles.ylabel(ax=ax, label="y position, in meters")
        styles.graph_legend(ax=ax, loc="upper left")
        cbar = fig.colorbar(img, ax=ax, shrink=0.75)
        cbar.set_label(bar_label)
        styles.remove_edge_ticks(ax=ax)
        fig.savefig(outfile)
        plt.close(fig)
    print(
        f"wrote {outfile.name} (layer {k + 1}, adjoint "
        f"{float(data['adjoint'][0]):.6e} against a finite difference of "
        f"{float(data['finite_difference'][0]):.6e})"
    )


draw_package(
    DATA / "lake-capture.npz",
    "Lake cell",
    "Lake capture fraction, dimensionless",
    OUTDIR / "lake-capture.pdf",
)
draw_package(
    DATA / "stream-capture.npz",
    "Stream reach",
    "Streamflow capture fraction, dimensionless",
    OUTDIR / "sfr-capture.pdf",
)
