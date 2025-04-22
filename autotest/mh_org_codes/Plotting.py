from freyberg6 import *


def plot_contour(x, y, l_anal, l_num, contour_intervals):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        # vmin, vmax = lam_anal[i].min(), lam_anal[i].max()
        # contour_intervals = np.arange(vmin, vmax, 0.0001)
        # c1 = ax.contour(x, y, l_anal, contour_intervals, colors="black", linestyles='solid')
        # c2 = ax.contour(x, y, l_num, contour_intervals,colors="black", linestyles='dashed')
        c1 = ax.contour(x, y, l_anal, colors="black", linestyles='solid')
        c2 = ax.contour(x, y, l_num, colors="black", linestyles='dashed')
        # plt.clabel(c1, fmt="%2.2e")
        # plt.clabel(c2, fmt="%1.1f")
        plt.clabel(c2, fmt="%2.2e")
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


def plot_contour_sensitivity(x, y, Sadj, Sper, contour_intervals):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    # vmin, vmax = lam_anal[i].min(), lam_anal[i].max()
    # contour_intervals = np.arange(vmin, vmax, 0.0001)
    c1 = ax.contour(x, y, Sadj, contour_intervals, colors="black", linestyles='solid')
    c2 = ax.contour(x, y, Sper, contour_intervals,colors="blue", linestyles='dashed')
    # c1 = ax.contour(x, y, Sadj, colors="black", linestyles='solid')
    # c2 = ax.contour(x, y, Sper, colors="black", linestyles='dashed')
    # plt.clabel(c1, fmt="%2.2e")
    # plt.clabel(c2, fmt="%1.1f")
    plt.clabel(c2, fmt="%2.2e")
    h1, _ = c1.legend_elements()
    h2, _ = c2.legend_elements()
    ax.legend([h1[0], h2[0]], ["MF6-ADJ", "Perturbation"],
              bbox_to_anchor=(0.78, 1.16), fontsize=12)
    plt.savefig('contour_sensitivity')

def plot_colorbar_sensitivity(x, y, Sadj, Sper, layer_nb, parameter_name, contour_intervals):
        fig, axes = plt.subplots(1, 2, figsize=(9, 7), constrained_layout=True)
        # first subplot
        ax = axes[0]
        ax.set_title("MF6-ADJ", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        pa = modelmap.plot_array(Sadj)
        # quadmesh = modelmap.plot_bc("CHD")
        # linecollection = modelmap.plot_grid(lw=0.5, color="0.4")
        contours = modelmap.contour_array(
            Sadj,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.1e")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)

        # second subplot
        ax = axes[1]
        ax.set_title("Perturbation", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        # linecollection = modelmap.plot_grid(lw=0.5, color="0.4")
        pa = modelmap.plot_array(Sper)
        # quadmesh = modelmap.plot_bc("CHD")
        contours = modelmap.contour_array(
            Sper,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.1e")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)
        plt.suptitle("Freyberg model transient (Layer %s) \n Sensitivity parameter: %s" % (layer_nb, parameter_name),
                     fontsize=16)
        plt.savefig('Sensistivity_%s_%s' %(layer_nb, parameter_name))


def plot_contour_list(x, y, list_times, l_anal, l_num):
    for i in range(len(list_times)):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        # vmin, vmax = lam_anal[i].min(), lam_anal[i].max()
        # contour_intervals = np.arange(vmin, vmax, 0.0001)
        # c1 = ax.contour(x, y, lam_anal[i], contour_intervals, colors="black", linestyles='solid')
        # c2 = ax.contour(x, y, lam_3d[i][0], contour_intervals, colors="black", linestyles='dotted')
        c1 = ax.contour(x, y, l_anal[i], colors="black", linestyles='solid')
        c2 = ax.contour(x, y, l_num[i][0], colors="black", linestyles='dashed')
        plt.clabel(c1, fmt="%2.2e")
        # plt.clabel(c2, fmt="%1.1e")
        h1, _ = c1.legend_elements()
        h2, _ = c2.legend_elements()
        ax.legend([h1[0], h2[0]], ["Analytical: t = %2.1f days" % list_times[i], "Numerical: t = %2.2f day" % list_times[i]],
                  bbox_to_anchor=(0.78, 1.16), fontsize=12)
        plt.savefig('contour_%s' % i)

def plot_colorbar_list(x, y, list_times, l_anal, l_num, vmin, vmax):
    for i in range(len(list_times)):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # first subplot
        ax = axes[0]
        ax.set_title("Analytical: t = %2.2f day" % list_times[i], fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        pa = modelmap.plot_array(l_anal[i], vmin=vmin[i], vmax=vmax[i])
        # quadmesh = modelmap.plot_bc("CHD")
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        contours = modelmap.contour_array(
            l_anal[i],
            # levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.2e")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)

        # second subplot
        ax = axes[1]
        ax.set_title("Numerical: t = %2.1f days" % list_times[i], fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
        linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
        pa = modelmap.plot_array(l_num[i][0], vmin=vmin[i], vmax=vmax[i])
        # quadmesh = modelmap.plot_bc("CHD")
        contours = modelmap.contour_array(
            l_num[i][0],
            # levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.2e")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)
        plt.savefig('colorbar_%s' % i)

def plot_colorbar_sensitivity_Single(x, y, Sadj):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    ax.set_title("MF6-ADJ", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(Sadj)
    vmin, vmax = Sadj.min(), Sadj.max()
    contour_intervals = np.arange(vmin, vmax, 1.0e-2)
    # quadmesh = modelmap.plot_bc("CHD")
    linecollection = modelmap.plot_grid(lw=0.5, color="0.4", vmin=vmin, vmax=vmax)
    contours = modelmap.contour_array(
        Sadj,
        # levels=contour_intervals,
        colors="white",
    )
    ax.clabel(contours, fmt="%2.1e")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)
    plt.savefig('colorbar_sensitivity_single')


def plot_contour_list_Single(x, y, list_times, l_num):
    for i in range(len(list_times)):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        c = ax.contour(x, y, l_num[i][0], colors="black", linestyles='solid')
        plt.clabel(c, fmt="%.1f")
        plt.title("Adjoint State: t = %.1f day" % list_times[i], fontsize=14)
        plt.savefig('contour_Adj_state_%s' % i)


def plot_colorbar_arrays(arr1, arr2, layer_nb, parameter_name):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
    # fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    # ax = axes[0]
    ax1.set_title("Adjoint State", fontsize=16)
    arr1[arr1 == -999.] = np.nan
    cb1 = ax1.imshow(arr1)
    plt.colorbar(cb1, ax=ax1)
    # ax2 = axes[1]
    ax2.set_title("Perturbation", fontsize=16)
    arr2[arr2 == -999.] = np.nan
    cb2 = ax2.imshow(arr2)
    plt.colorbar(cb2, ax=ax2)
    plt.show()
    plt.suptitle("Freyberg model transient (Layer %s) \n Sensitivity parameter: %s" % (layer_nb, parameter_name), fontsize=16)
    # plt.savefig('Adj_vs_perturbation_SteadyState')

def plot_colorbar_arrays_RCH(arr1, arr2, number_of_stress_period, parameter_name):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 7))
    # fig, axes = plt.subplots(1, 2, figsize=(9, 6), constrained_layout=True)
    # ax = axes[0]
    ax1.set_title("Adjoint State", fontsize=16)
    arr1[arr1 == -999.] = np.nan
    cb1 = ax1.imshow(arr1)
    plt.colorbar(cb1, ax=ax1)
    # ax2 = axes[1]
    ax2.set_title("Perturbation", fontsize=16)
    arr2[arr2 == -999.] = np.nan
    cb2 = ax2.imshow(arr2)
    plt.colorbar(cb2, ax=ax2)
    plt.show()
    plt.suptitle("Freyberg model transient (SP %s) \n Sensitivity parameter: %s" % (number_of_stress_period, parameter_name), fontsize=16)

def plot_colorbar_sensitivity_RCH(Sadj, Sper, number_of_stress_period, parameter_name, contour_intervals):
        fig, axes = plt.subplots(1, 2, figsize=(9, 7), constrained_layout=True)
        # first subplot
        ax = axes[0]
        ax.set_title("MF6-ADJ", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        pa = modelmap.plot_array(Sadj)
        # quadmesh = modelmap.plot_bc("CHD")
        # linecollection = modelmap.plot_grid(lw=0.5, color="0.4")
        contours = modelmap.contour_array(
            Sadj,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.1e")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)

        # second subplot
        ax = axes[1]
        ax.set_title("Perturbation", fontsize=16)
        modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
        # linecollection = modelmap.plot_grid(lw=0.5, color="0.4")
        pa = modelmap.plot_array(Sper)
        # quadmesh = modelmap.plot_bc("CHD")
        contours = modelmap.contour_array(
            Sper,
            levels=contour_intervals,
            colors="white",
        )
        ax.clabel(contours, fmt="%2.1e")
        cb = plt.colorbar(pa, shrink=1.0, ax=ax)
        plt.suptitle("Freyberg model transient (SP %s) \n Sensitivity parameter: %s" % (number_of_stress_period, parameter_name),
                     fontsize=16)
        plt.savefig('Sensistivity_%s_%s' % (number_of_stress_period, parameter_name))


def plot_sens_arrays():
    from matplotlib.backends.backend_pdf import PdfPages
    gwf = sim.get_model()
    nlay,nrow,ncol = gwf.dis.nlay.data,gwf.dis.nrow.data,gwf.dis.ncol.data
    obs_arr = np.zeros((nlay,nrow,ncol))
    lines = open("head.obs",'r').readlines()[1:-1]
    for line in lines:
        l,r,c = [int(i) for i in line.strip().split()[2:]]
        obs_arr[l-1,r-1,c-1] = 1.0
    obs_arr[obs_arr==0] = np.nan
    sens_array_tags = ["k_k","k33_k","ss_k","hghb_k","cghb_k","rech_kper","wel_kper"]
    files = os.listdir(".")
    sens_array_files = []
    for f in files:
        for tag in sens_array_tags:
            if f.startswith(tag):
                sens_array_files.append(f)
                break
    sens_array_files.sort()
    #print(sens_array_files)
    with PdfPages("sens_arrays.pdf") as pdf:
         for f in sens_array_files:
                k = int(f.split(".")[0].split("_")[-1][1:])
                arr = np.loadtxt(f)
                arr[arr==-999] = np.nan
                arr[arr==0.0] = np.nan
                fig,ax = plt.subplots(1,1,figsize=(8,5))
                cb = ax.imshow(arr)
                ax.imshow(obs_arr[k,:,:],alpha=0.5,cmap="gist_rainbow")
                plt.colorbar(cb,ax=ax)
                ax.set_title(f)
                plt.tight_layout()
                pdf.savefig()
                plt.close(fig)
                print(f)

if __name__ == "__main__":
     plot_sens_arrays()