"""Draw the figures for the supplemental technical information document.

Reads the archive written beside the document rather than the model output, so
the figures redraw from a clone without running a simulation.
"""

import pathlib as pl
import tempfile
import sys

import flopy
import flopy.plot.styles as styles
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

ROOT = pl.Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT / "autotest"))
import test_lak
import test_sfr

DATA = pl.Path(sys.argv[1])
WORK = pl.Path(tempfile.mkdtemp(prefix="suppinfo-"))

# the width of a column of the document, so no figure is scaled in LaTeX
COLUMN_WIDTH = 3.43
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
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.9), layout="constrained")
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


def draw_package(archive, builder, boundary, bar_label, outfile):
    """Map the capture a lake or a stream takes, on the model it is tested on.

    The capture comes from the archive. The model is rebuilt so the boundaries
    can be drawn from the packages themselves rather than from a list of cells
    kept beside them.
    """
    data = np.load(archive)
    nlay, nrow, ncol = (int(v) for v in data["shape"])
    arr = np.array(data["capture"], dtype=float).reshape((nlay, nrow, ncol))
    totals = [np.nansum(np.abs(arr[k])) for k in range(nlay)]
    k = int(np.argmax(totals))
    layer = arr[k]
    vmax = float(np.nanpercentile(np.abs(layer[np.isfinite(layer)]), 99.5))

    gwf = builder()
    with styles.USGSMap():
        fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 3.4), layout="constrained")
        mv = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=k)
        img = mv.plot_array(layer, cmap="viridis", vmin=0.0, vmax=vmax)
        mv.plot_grid(lw=0.3, color="0.6")
        # the boundaries the capture is shared with, drawn from the packages
        for name, color in boundary:
            try:
                mv.plot_bc(name=name, color=color, plotAll=True)
            except Exception:
                continue
        handles = [
            mpatches.Patch(facecolor=color, edgecolor="none", label=name)
            for name, color in boundary
        ]
        ax.set_aspect("equal")
        styles.xlabel(ax=ax, label="x position, in meters")
        styles.ylabel(ax=ax, label="y position, in meters")
        # below the axes, so the legend covers none of the capture
        styles.graph_legend(
            ax=ax,
            handles=handles,
            labels=[h.get_label() for h in handles],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(handles),
        )
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


def _lake_model():
    """The model the lake tests use."""
    sim, _ = test_lak._build_model(
        WORK / "lake",
        boundary="lak",
        constant_stage=False,
        well_rate=test_lak.WELL_RATE,
    )
    return sim.get_model()


def _stream_model():
    """The model the streamflow tests use."""
    sim = test_sfr._build_model(WORK / "stream", well_rate=test_sfr.WELL_RATE)
    return sim.get_model()


draw_package(
    DATA / "lake-capture.npz",
    _lake_model,
    [("LAK", "tab:blue"), ("GHB", "tab:green"), ("WEL", "tab:red")],
    "Lake capture fraction, dimensionless",
    OUTDIR / "lake-capture.pdf",
)
draw_package(
    DATA / "stream-capture.npz",
    _stream_model,
    [("SFR", "tab:blue"), ("GHB", "tab:green"), ("WEL", "tab:red")],
    "Streamflow capture fraction, dimensionless",
    OUTDIR / "sfr-capture.pdf",
)
