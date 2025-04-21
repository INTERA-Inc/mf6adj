# ## Imports
import os

import flopy
import matplotlib.pyplot as plt
import modflowapi
import numpy as np
from GridData import *

# from APIData import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# ### Executables
exe = os.path.join('D:/','github','MFexe','mf6.exe')
dll = os.path.join('D:/','github','MFexe','libmf6.dll')

# ### Create the FloPy simulation object
sim = flopy.mf6.MFSimulation(
      sim_name=name,
      exe_name=exe,
      version="mf6",
      sim_ws=".",
      memory_print_option="ALL"
)

# ### Create the Flopy `TDIS` object
tdis = flopy.mf6.ModflowTdis(
       sim,
       pname="tdis",
       time_units="DAYS",
       nper=1,
       perioddata=[(tf, 1, 1.0)]
)

# ### Create the Flopy `IMS` Package object
ims = flopy.mf6.ModflowIms(
    sim,
    pname="ims",
    complexity="SIMPLE",
    linear_acceleration="BICGSTAB",
)

# Create the Flopy groundwater flow (gwf) model object
model_nam_file = f"{name}.nam"
gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=name,
    model_nam_file=model_nam_file,
    save_flows=True,
    newtonoptions="NEWTON UNDER_RELAXATION",
)

# Now that the overall simulation is set up, we can focus on building the
# groundwater flow model. The groundwater flow model will be built by
# adding packages to it that describe the model characteristics.

# ### Create the discretization (`DIS`) Package
idm = np.ones((Nlay, N, N))
# idm[0][0][1] = 1
# idm[0][1][2] = 0
dis = flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=Nlay,
    nrow=N,
    ncol=N,
    delr=delrow,
    delc=delcol,
    top=0.0,
    botm=bot,
    idomain=idm
)


# ### Create the initial conditions (`IC`) Package
start = h1 * np.ones((Nlay, N, N))
ic = flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=start)


# ### Create the storage (`STO`) Package
sto = flopy.mf6.ModflowGwfsto(
      gwf,
      # steady_state={0: False},
      # transient={0: True}
        )

# ### Create the constant head (`CHD`) Package
chd_rec = []
layer = 0
for row_col in range(0, N):
    chd_rec.append(((layer, row_col, 0), h1))
    chd_rec.append(((layer, row_col, N - 1), h1))
    if row_col != 0 and row_col != N - 1:
        chd_rec.append(((layer, 0, row_col), h1))
        chd_rec.append(((layer, N - 1, row_col), h1))
chd = flopy.mf6.ModflowGwfchd(
    gwf,
    stress_period_data=chd_rec,
)

# The `CHD` Package stored the constant heads in a structured array,
# also called a `numpy.recarray`. We can get a pointer to the recarray
# for the first stress period (iper = 0) as follows.
iper = 0
ra = chd.stress_period_data.get_data(key=iper)
ra
list_ch = [ra[item][0] for item in range(len(ra))]

# # ### Create the well (`WEL`) Package
wel_rec = [(Nlay - 1, int(N / 2), int(N / 2), q)]
wel = flopy.mf6.ModflowGwfwel(
    gwf,
    stress_period_data=wel_rec,
)

# ### Create the output control (`OC`) Package
#
# Save heads and budget output to binary files and print heads to the model
# listing file at the end of the stress period.
headfile = f"{name}.hds"
head_filerecord = [headfile]
budgetfile = f"{name}.cbb"
budget_filerecord = [budgetfile]
saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
printrecord = [("HEAD", "ALL")]
oc = flopy.mf6.ModflowGwfoc(
    gwf,
    saverecord=saverecord,
    head_filerecord=head_filerecord,
    budget_filerecord=budget_filerecord,
    printrecord=printrecord,
)

# ### Create the node property flow (`NPF`) Package
list_cond = [8.5113804,
7.0794578,
9.1201084,
8.7096359,
12.0226443,
10.4712855,
7.7624712,
8.7096359,
12.8824955,
13.1825674,
9.7723722,
8.5113804,
8.1283052,
11.4815362,
9.5499259,
13.1825674,
11.4815362,
8.9125094,
10.9647820,
11.4815362,
11.2201845,
12.3026877,
8.7096359,
9.5499259,
10.7151931,
11.4815362,
10.4712855,
9.7723722,
9.5499259,
7.0794578,
10.0000000,
7.0794578,
9.7723722,
11.2201845,
12.0226443,
12.3026877,
8.3176377,
8.3176377,
7.0794578,
13.8038426,
12.3026877,
12.5892541,
8.5113804,
10.7151931,
9.5499259,
10.9647820,
10.2329299,
12.8824955,
8.1283052,
11.7489755,
10.7151931,
14.1253754,
10.0000000,
7.2443596,
12.0226443,
8.3176377,
10.2329299,
8.5113804,
8.7096359,
9.1201084,
12.8824955,
12.8824955,
12.5892541,
9.1201084,
7.0794578,
13.4896288,
10.0000000,
9.3325430,
13.4896288,
10.0000000,
12.0226443,
10.7151931,
10.9647820,
7.7624712,
9.3325430,
9.3325430,
13.1825674,
7.4131024,
13.8038426,
7.4131024,
7.7624712,
9.5499259,
8.5113804,
7.0794578,
10.4712855,
12.5892541,
12.0226443,
8.5113804,
13.1825674,
9.3325430,
7.5857758,
10.4712855,
8.7096359,
11.7489755,
7.2443596,
10.2329299,
8.7096359,
13.8038426,
12.0226443,
7.0794578,
14.1253754,
7.0794578,
7.0794578,
12.3026877,
10.7151931,
12.0226443,
9.3325430,
10.9647820,
9.1201084,
8.3176377,
13.1825674,
9.5499259,
8.1283052,
11.4815362,
11.4815362,
9.1201084,
14.1253754,
13.1825674,
13.4896288,
8.9125094,
10.2329299
]

array_cond = np.array(list_cond)
array_cond_3D = np.reshape(array_cond, (Nlay, Nrow, Ncol))
for kk in range(Nlay):
    for jj in range(Nrow):
        for ii in range(Ncol):
            if (kk,jj,ii) in list_ch:
                array_cond_3D[kk][jj][ii] = k

kkk = np.array(array_cond_3D)
# kkk = np.ones((Nlay, N, N))
# kkk = k*kkk
# kkk[0][1][1] = k + epsilon

npf = flopy.mf6.ModflowGwfnpf(
    gwf,
    icelltype=1,
    k=kkk,
    # alternative_cell_averaging='logarithmic'
    # alternative_cell_averaging='amt-lmk',
    # alternative_cell_averaging='amt-hmk',
)

# # ### Write the datasets
sim.write_simulation()

#-----------------------------------------------------------------

# success, buff = sim.run_simulation()
# if not success:
#     raise Exception("MODFLOW 6 did not terminate normally.")
#
# h = gwf.output.head().get_data(kstpkper=(0, 0))

#-------------------------------------------------------------------------
# ### API-----------------------------------------------------------------
mf6api = modflowapi.ModflowApi(dll)
mf6api.initialize()
current_time = mf6api.get_current_time()
end_time = mf6api.get_end_time()
max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
CELLAREA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("AREA", "%s/DIS" % name)))
DELR = np.array(mf6api.get_value_ptr(mf6api.get_var_address("DELR", "%s/DIS" % name)))
DELC = np.array(mf6api.get_value_ptr(mf6api.get_var_address("DELC", "%s/DIS" % name)))
CELLTOP = np.array(mf6api.get_value_ptr(mf6api.get_var_address("TOP", "%s/DIS" % name)))
CELLBOT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("BOT", "%s/DIS" % name)))
JA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("JA", "%s/CON" % name)))
IA = np.array(mf6api.get_value_ptr(mf6api.get_var_address("IA", "%s/CON" % name)))
JA_p = np.subtract(JA, 1)
IA_p = np.subtract(IA, 1)
IAC = np.array([IA[i+1] - IA[i] for i in range(len(IA)-1)])
SAT = np.array(mf6api.get_value_ptr(mf6api.get_var_address("SAT", "%s/NPF" % name)))
K11 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K11", "%s/NPF" % name)))
K22 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K22", "%s/NPF" % name)))
K33 = np.array(mf6api.get_value_ptr(mf6api.get_var_address("K33", "%s/NPF" % name)))
NODES = np.array(mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name)))[0]

head = []
amat = []
time = []
deltat = []

print('JA = ', JA)
print('IA = ', IA)
print('IAC = ', IAC)

h = []
MAT = []
iii = -1
# Solving Forward MF6 Model
while current_time < end_time:
    time.append(current_time)
    dt = mf6api.get_time_step()
    mf6api.prepare_time_step(dt)
    kiter = 0
    mf6api.prepare_solve(1)
    while kiter < max_iter:
        has_converged = mf6api.solve(1)
        kiter += 1
        if has_converged:
            iii += 1
            if iii == 0:
                print('****************************************************************')
                hi = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s"%name))
                h.append([hi[item] for item in range(len(hi))])
            break
    mf6api.finalize_solve(1)
    mf6api.finalize_time_step()
    current_time = mf6api.get_current_time()
    dt1 = mf6api.get_time_step()
    deltat.append(dt1)
    amat = mf6api.get_value_ptr(mf6api.get_var_address("AMAT", "SLN_1"))
    MAT.append([amat[item] for item in range(len(amat))])
    rhs = mf6api.get_value_ptr(mf6api.get_var_address("RHS", "SLN_1"))
    head = mf6api.get_value_ptr(mf6api.get_var_address("X", "%s" % name))
    h.append([head[item] for item in range(len(head))])
    head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "%s" % name))
    if not has_converged:
        print("model did not converge")
        break
try:
    mf6api.finalize()
    success = True
except:
    raise RuntimeError

#-----------------------------------------------------------------------------------------------------------------------
time.append(end_time)

#-----------------------------------------------------------------------------------------------------------------------
## Post-Process Head Results
h = gwf.output.head().get_alldata()[-1]
hh = np.reshape(h[0], (Nlay * Nrow *Ncol))
h1 = gwf.output.head().get_alldata()[0]
h2 = gwf.output.head().get_alldata()[-1]
x = np.linspace(0, L1, Ncol)
y = np.linspace(0, L2, Nrow)
y = y[::-1]
vmin, vmax = 90.0, 100.0
contour_intervals = np.arange(90, 100, 1.0)
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1, aspect="equal")
c = ax.contour(x, y, h[0], contour_intervals, colors="black")
plt.clabel(c, fmt="%2.1f")

def plot_colorbar_cond(x, y, Sadj, Sper, contour_intervals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    # first subplot
    ax = axes[0]
    ax.set_title("Hydraulic conductivity", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
    pa = modelmap.plot_array(Sadj)
    # quadmesh = modelmap.plot_bc("CHD")
    linecollection = modelmap.plot_grid(lw=0.5, color="0.4")
    # contours = modelmap.contour_array(
    #     Sadj,
    #     levels=contour_intervals,
    #     colors="white",
    # )
    # ax.clabel(contours, fmt="%2.1e")
    cb = plt.colorbar(pa, shrink=1.0, ax=ax)
    cb.set_label('K (m/s)', fontsize=16)

    # second subplot
    ax = axes[1]
    ax.set_title("Perturbation", fontsize=16)
    modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
    linecollection = modelmap.plot_grid(lw=0.5, color="0.4")
    pa = modelmap.plot_array(Sper)
    # quadmesh = modelmap.plot_bc("CHD")
    # contours = modelmap.contour_array(
    #     Sper,
    #     levels=contour_intervals,
    #     colors="white",
    # )
    # ax.clabel(contours, fmt="%2.1e")
    cb=plt.colorbar(pa, shrink=1.0, ax=ax)
    plt.savefig('colorbar_cond')

x = np.linspace(0, L1, Ncol)
y = np.linspace(0, L2, Nrow)
y = y[::-1]
contour_intervals = np.array([-3.0e-01, -2.0e-01, -1.0e-01, -9.0e-02, -8.0e-02, -7.0e-02, -6.0e-02, -5.0e-02, -4.0e-02, -3.0e-02, -2.0e-02, -1.0e-02, -9.0e-03, -8.0e-03, -7.0e-03, -6.0e-03, -5.0e-03, -4.0e-03, -3.0e-03, -2.0e-03, -1.0e-03, 0.0, 1.0e-03, 2.0e-03, 3.0e-03, 4.0e-03, 5.0e-03, 6.0e-03, 7.0e-03, 8.0e-03, 9.0e-03, 1.0e-02, 2.0e-02, 3.0e-02, 4.0e-02, 5.0e-02, 6.0e-02, 7.0e-02, 8.0e-02, 9.0e-02, 1.0e-01, 2.0e-01, 3.0e-01])
plot_colorbar_cond(x, y, kkk[0], kkk[0], contour_intervals)



