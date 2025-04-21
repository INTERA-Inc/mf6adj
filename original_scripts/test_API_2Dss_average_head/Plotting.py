from GWF_2D_SS import *


def plot_contour(x, y, l_anal, l_num, contour_intervals):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        # vmin, vmax = lam_anal[i].min(), lam_anal[i].max()
        # contour_intervals = np.arange(vmin, vmax, 0.0001)
        # c1 = ax.contour(x, y, lam_anal[i], contour_intervals, colors="black", linestyles='solid')
        # c2 = ax.contour(x, y, lam_3d[i][0], contour_intervals, colors="black", linestyles='dotted')
        c1 = ax.contour(x, y, l_anal, contour_intervals, colors="black", linestyles='solid')
        c2 = ax.contour(x, y, l_num, contour_intervals,colors="black", linestyles='dashed')
        # plt.clabel(c1, fmt="%2.2e")
        plt.clabel(c2, fmt="%1.1f")
        h1, _ = c1.legend_elements()
        h2, _ = c2.legend_elements()
        ax.legend([h1[0], h2[0]], ["Analytical", "Numerical (MF6-ADJ)"],
                  bbox_to_anchor=(0.78, 1.16), fontsize=12)
        plt.savefig('contour')

def plot_colorbar(x, y, l_anal, l_num, contour_intervals):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # first subplot
        ax = axes[0]
        ax.set_title("Analytical", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        pa = modelmap.plot_array(l_anal)
        # quadmesh = modelmap.plot_bc("CHD")
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        contours = modelmap.contour_array(
            l_anal,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%1.1f")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)

        # second subplot
        ax = axes[1]
        ax.set_title("Numerical (MF6-ADJ)", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        pa = modelmap.plot_array(l_num)
        # quadmesh = modelmap.plot_bc("CHD")
        contours = modelmap.contour_array(
            l_num,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%1.1f")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)
        plt.savefig('colorbar')

def plot_colorbar_sensitivity(x, y, Sadj, Sper, contour_intervals):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # first subplot
        ax = axes[0]
        ax.set_title("MF6-ADJ", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        pa = modelmap.plot_array(Sadj)
        # quadmesh = modelmap.plot_bc("CHD")
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        contours = modelmap.contour_array(
            Sadj,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.2f")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)

        # second subplot
        ax = axes[1]
        ax.set_title("Perturbation", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        pa = modelmap.plot_array(Sper)
        # quadmesh = modelmap.plot_bc("CHD")
        contours = modelmap.contour_array(
            Sper,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.2f")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)
        plt.savefig('colorbar_sensitivity')



