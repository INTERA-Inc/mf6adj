# ## Imports
# from GridData import *
# from ADJ import *
import os
import sys

# this is so py can find the api deps
sys.path.insert(0,os.path.join(".."))

import platform
from operator import add

import flopy
import matplotlib.pyplot as plt
import modflowapi
import numpy as np

# from APIData import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# from scipy.linalg import solve_banded

# ### Executables
#exe = os.path.join('D:/','github','MFexe','mf6.exe')
#dll = os.path.join('D:/','github','MFexe','libmf6.dll')

if "linux" in platform.platform().lower():
    dll = os.path.join("..","..", "bin", "linux", "libmf6.so")
    exe = os.path.join("..","..", "bin", "linux", "mf6")
elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
    dll = os.path.join("..","..", "bin", "mac", "libmf6.dylib")
    exe = os.path.join("..","..", "bin", "mac", "mf6")
else:
    dll = os.path.join("..","..", "bin", "win", "libmf6.dll")
    exe = os.path.join("..","..", "bin", "win", "mf6.exe")


name = 'FREYBERG6'
sim = flopy.mf6.MFSimulation.load()
gwf = sim.get_model()



list_ch = []
list_ghb = []
dict_wel ={}
with open(r'%s.nam' % gwf.name, 'r') as fp:
    lines = fp.readlines()
    for row in lines:
        # check if string present on a current line
        chdfile = '%s.chd' %gwf.name
        ghbfile = '%s.ghb' % gwf.name
        welfile = '%s.wel' % gwf.name
        #         print(row.find(word))
        # find() method returns -1 if the value is not found,
        # if found it return 0
        print('++++++++++++++++', row.find(ghbfile))
        if row.find(chdfile) != -1:
           for i in range(gwf.chd.stress_period_data.array[0].size):
               list_ch.append(gwf.chd.stress_period_data.array[0][i][0])
        if row.find(ghbfile) != -1:
           for i in range(gwf.ghb.stress_period_data.array[0].size):
               list_ghb.append([gwf.ghb.stress_period_data.array[0][i][0], gwf.ghb.stress_period_data.array[0][i][1], gwf.ghb.stress_period_data.array[0][i][2]])
        if row.find(welfile) != -1:
           dict_wel = gwf.wel.stress_period_data.data


## rcha package
# list_nsp = list(range(gwf.modeltime.nper))
# list_rcha_rates = [6.16896000E-05, 9.54311085E-05, 1.01775880E-04, 9.64490835E-05, 8.10016620E-05, 5.99313330E-05, 3.93729420E-05, 2.53122870E-05, 2.18433075E-05, 2.99760090E-05, 4.73424840E-05, 6.88862895E-05, 8.83346835E-05, 1.00025069E-04, 1.00553660E-04, 8.97665265E-05, 7.08045030E-05, 4.91885625E-05, 3.12124365E-05, 2.21100810E-05, 2.45317485E-05, 3.77723325E-05, 5.79767055E-05, 7.92621165E-05, 9.54311085E-05]
# list_array_recha = []
# for nsp in range(gwf.modeltime.nper):
#     list_array_recha.append(np.zeros((gwf.modelgrid.nrow, gwf.modelgrid.ncol)) + list_rcha_rates[nsp])
# dict_recha = dict(zip(list_nsp,list_array_recha))
#
# rcha = flopy.mf6.ModflowGwfrcha(
#     gwf,
#     recharge=dict_recha,
# )
# sim.write_simulation()

##

# for i in range(gwf.chd.stress_period_data.array[0].size):
#     list_ch.append(gwf.chd.stress_period_data.array[0][i][0])


# # Create the Flopy groundwater flow (gwf) model object
# model_nam_file = f"{m.name}.nam"
# gwf = flopy.mf6.ModflowGwf(
#     sim,
#     modelname=m.name,
#     model_nam_file=model_nam_file,
# )
#
# print(m.dis.nrow.data)
# print(m.npf.icelltype)
#
# list_ch = m.ghb.stress_period_data.array[0]


#
# # ### API-----------------------------------------------------------------
mf6api = modflowapi.ModflowApi(dll)
mf6api.initialize()
current_time = mf6api.get_current_time()
end_time = mf6api.get_end_time()
max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))

# NPER = mf6api.get_value_ptr(mf6api.get_var_address("NPER", "TDIS"))
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
STORAGE = np.array(mf6api.get_value_ptr(mf6api.get_var_address("SS", "%s/STO" % name)))
NODES = np.array(mf6api.get_value_ptr(mf6api.get_var_address("NODES", "%s/CON" % name)))[0]
NODESUSER = np.array(mf6api.get_value_ptr(mf6api.get_var_address("NODESUSER", "%s/DIS" % name)))[0]

#
head = []
amat = []
time = []
deltat = []
h = []
h_old = []
MAT = []
iii = -1

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
                # print('hi == ', hi)
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
    # head = mf6api.get_value_ptr(mf6api.get_var_address("X", "TEST_API_2D"))
    # h.append([head[item] for item in range(len(head))])
    # head_old = mf6api.get_value_ptr(mf6api.get_var_address("XOLD", "TEST_API_2D"))
    # h_old.append([head_old[item] for item in range(len(head_old))])
    # print('head_old = ', head_old)
    # A = csr_matrix((amat, JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
    # print(A)
    # head_p = spsolve(A, rhs)
    # print('-----------------------------------------------------------------------------')
    # print(head)
    # print(head_p)
    # print('-----------------------------------------------------------------------------')
    if not has_converged:
        print("model did not converge")
        break
try:
    mf6api.finalize()
    success = True
except:
    raise RuntimeError

# #-----------------------------------------------------------------------------------------------------------------------
time.append(end_time)
# t = [20.0, 40.0, 60.0, 80.0, 100.0]
# #-----------------------------------------------------------------------------------------------------------------------
# # ## Post-Process Head Results
# # h = gwf.output.head().get_alldata()[-1]
# # h1 = gwf.output.head().get_alldata()[0]
# # h2 = gwf.output.head().get_alldata()[-1]
# # x = np.linspace(0, L1, Ncol)
# # y = np.linspace(0, L2, Nrow)
# # y = y[::-1]
# # vmin, vmax = 1000.0, 1001.0
# # contour_intervals = np.arange(1000, 1001.1, 0.1)
# # fig = plt.figure(figsize=(6, 6))
# # ax = fig.add_subplot(1, 1, 1, aspect="equal")
# # c = ax.contour(x, y, h[0], contour_intervals, colors="black")
# # plt.clabel(c, fmt="%2.1f")
#
#
# # # ### Plot a Map of head at two different times
# # fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
# # # first subplot
# # ax = axes[0]
# # ax.set_title("Model Layer 1")
# # modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax)
# # pa = modelmap.plot_array(h1, vmin=vmin, vmax=vmax)
# # quadmesh = modelmap.plot_bc("CHD")
# # linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
# # contours = modelmap.contour_array(
# #     h,
# #     levels=contour_intervals,
# #     colors="black",
# # )
# # ax.clabel(contours, fmt="%2.1f")
# # cb = plt.colorbar(pa, shrink=0.5, ax=ax)
# # # second subplot
# # ax = axes[1]
# # ax.set_title(f"Model Layer 2")
# # modelmap = flopy.plot.PlotMapView(model=gwf, ax=ax, layer=Nlay - 1)
# # linecollection = modelmap.plot_grid(lw=0.5, color="0.5")
# # pa = modelmap.plot_array(h2, vmin=vmin, vmax=vmax)
# # quadmesh = modelmap.plot_bc("CHD")
# # contours = modelmap.contour_array(
# #     h,
# #     levels=contour_intervals,
# #     colors="black",
# # )
# # ax.clabel(contours, fmt="%2.1f")
# # cb = plt.colorbar(pa, shrink=0.5, ax=ax)
#
