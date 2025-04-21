from ADJ import *

# from Analytical_2D import *
from Plotting import *


def get_analytical_adj_state(t, xs, ys, ts, MM, NN):
    list_adj_state = []
    for j in range(Nrow):
        for i in range(Ncol):
            x = gwf.modelgrid.xcellcenters[j][i]
            y = gwf.modelgrid.ycellcenters[j][i]
            list_adj_state.append(phi_s(x, y, t, xs, ys, ts, MM, NN))
    array_adj_state = np.array(list_adj_state)
    array_adj_state_2D = np.reshape(array_adj_state, (Nrow, Ncol))
    return array_adj_state_2D


#-------------------------------------------Adjoint System--------------------------------------------------------------
lam = np.zeros(len(IA-1))
reverse_deltat = deltat[::-1]
print('reverse_deltat = ', reverse_deltat)
print('rev_time = ', time[::-1])
print(len(deltat))

# Solving the Adjoint State System
# idx = []
# lam_anal = []
# for i in range(len(t)):
#     idx.append(time.index(t[i]))
#     lam_anal.append(get_analytical_adj_state(t[i], L1 / 2, L2 / 2, tf, 20, 20))
#     # lam_anal.append(get_analytical_adj_state(t[i], L1 / 2, L2 / 2, tf, 20, 20))

# list_AS = SolveAdjointHeadAtPoint(0, int(Nrow / 2), int(Ncol / 2), len(deltat)-1, len(deltat), len(IA)-1, reverse_deltat)
# list_AS = SolveAdjointHeadAtPoint(0, int(Nrow / 2), int(Ncol / 2), len(deltat)-1, len(deltat), len(IA)-1, reverse_deltat)

# list_AS = SolveAdjointLS(len(IA)-1, h, "OBSMF6", reverse_deltat)
# list_of_kji = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
# list_of_weights = [1.0, 1.0, 1.0]
# list_of_kji = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 1, 1), (0, 1, 2), (0, 1, 3), (0, 2, 1), (0, 2, 2), (0, 2, 3)]
# list_of_weights = [1.0, 0.4, 1.2, 3.5, 4.6, 1.8, 5.2, 8.2, 2.3]
# list_AS = SolveAdjointSWH(len(IA)-1, 0, len(deltat), list_of_kji, list_of_weights, reverse_deltat)
# list_of_kji = [(0, 0, 1)]
# list_of_weights = [1.0]
# list_AS = SolveAdjointSWH(len(IA)-1, 1, len(deltat), list_of_kji, list_of_weights, reverse_deltat)

MF6OUTPUTFILE = "MF6-ADJ.OUT"
# Check if the output file exists then remove it
if os.path.exists(MF6OUTPUTFILE):
    os.remove(MF6OUTPUTFILE)
fo = open(MF6OUTPUTFILE, "a")
MF6LOGO = """===============================================================================         
¦             @   @  @@@@@    @               @@@   @@@@@    @@@@@            ¦    
¦	      @@ @@  @       @	             @   @  @    @     @              ¦
¦	      @ @ @  @@@    @@@@    @@@@@    @@@@@  @    @     @              ¦
¦	      @   @  @     @    @	     @   @  @    @     @              ¦
¦	      @   @  @     @@@@@	     @   @  @@@@@   @@@               ¦
===============================================================================
                         MODFLOW-6-ADJOINT (MF6-ADJ)
          SENSITIVITY CALCULATIONS USING THE ADJOINT STATE METHOD FOR
                U.S. GEOLOGICAL SURVEY Hydrologic Model MODFLOW 6
===============================================================================\n"""
dashes = """-------------------------------------------------------------------------------\n"""
stars = """*******************************************************************************\n"""
TITLE_PM_LS = """            Sensitivity coefficients for Performance measure type:           
                    Sum of Weighted Least Square Heads:                      
                    Number of Sensitivity parameters: %d\n"""

TITLE_PM_SWH = """            Sensitivity coefficients for Performance measure type:           
                    Sum of Weighted Heads at a Given Time:\n"""

NBR_ASP = """                      Number of Adjoint State Problems: %d\n"""
text_pbm = """                      ---------------------------------
                      ¦          Problem: %d           ¦
                      ---------------------------------
                      ¦         Time step: %d          ¦
                      ---------------------------------
                      ¦            Region             ¦ 
                      ¦           --------            ¦\n"""
text_K = """                %d-Total conductivity K,     Number of zones: %d\n"""
text_Kz = """                %d-Vertical conductivity Kz, Number of zones: %d\n"""
text_SS = """                %d-Storage coefficient Ss,   Number of zones: %d\n"""
text_HGHB = """                %d-Storage coefficient HGHB,   Number of zones: %d\n"""
text_CGHB = """                %d-Storage coefficient CGHB,   Number of zones: %d\n"""
text_QWEL = """                %d-Storage coefficient QWEL,   Number of zones: %d\n"""
text_K1 = """%d-Sensitivity with respect to K: 
================================\n"""
text_Kz1 = """%d-Sensitivity with respect to Kz: 
=================================\n"""
text_SS1 = """%d-Sensitivity with respect to Ss: 
=================================\n"""
text_HGHB1 = """%d-Sensitivity with respect to HGHB: 
===================================\n"""
text_CGHB1 = """%d-Sensitivity with respect to CGHB: 
===================================\n"""
text_QWEL = """%d-Sensitivity with respect to QWEL: 
===================================\n"""
text_in_zone = """in zone %d: %.6e
----------\n"""
text_SC = """                           Sensitivity Coefficients:
                          ---------------------------\n"""
text_SC_K = """                        %d-Sensitivity with respect to K
                       =================================\n"""
text_SC_Kz = """                        %d-Sensitivity with respect to Kz
                       =================================\n"""
text_SC_SS = """                        %d-Sensitivity with respect to Ss
                       =================================\n"""
text_SC_HGHB = """                        %d-Sensitivity with respect to HGHB
                       =================================\n"""
text_SC_CGHB = """                        %d-Sensitivity with respect to CGHB
                       =================================\n"""
text_SC_QWEL = """                        %d-Sensitivity with respect to QWEL
                       =================================\n"""
fo.write(MF6LOGO)

# list of acceptable sensitivity parameters names
list_of_SENSI_NAMES = ['COND', 'CONDV', 'STOR', 'HGHB', 'CGHB', 'QWEL']

# ### k
d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k12 = d_amat_k()
for arr,tag in zip([d_mat_k11,d_mat_k22,d_mat_k33,d_mat_k12],["dadk11","dadk22","dadk33","dadk12"]):
    np.savetxt(tag+".dat",arr,fmt="%15.6E")
    
inputfile = "MF6-ADJ.INP"
with open(inputfile, 'r') as f:
    lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].startswith('PERFM'):
            PMname = lines[i+1].split()[0]
            if PMname == 'SWH':
                NPROB = int(lines[i + 1].split()[1])
                list_of_kji = []
                list_of_weights = []
                list_regions = []
                for j in range(2, len(lines)-i):
                    if len(lines[i+j].split()) == 1 and not lines[i+j].startswith('SENSI'):
                        nts = int(lines[i + j].split()[0])
                        list_of_kji = []
                        list_of_weights = []
                    elif len(lines[i+j].split()) > 1:
                        list_of_kji.append((int(lines[i + j].split()[0]), int(lines[i + j].split()[1]), int(lines[i + j].split()[2])))
                        list_of_weights.append(float(lines[i + j].split()[3]))
                    elif len(lines[i+j].split()) == 0:
                        list_regions.append([nts, list_of_kji, list_of_weights])
                    if len(list_regions) == NPROB:
                        break
                # Solving the adjoint state problems for PM = 'SWH'
                list_AS = []
                list_lam_dAdk_h = []
                list_lam_dAdk33_h = []
                list_sens_ss_indirect = []
                list_S_K_adj = []
                list_S_K33_adj = []
                list_S_ss_adj = []
                for k in range(NPROB):
                    # Adjoint States
                    list_AS.append(SolveAdjointSWH(len(IA) - 1, list_regions[k][0], len(deltat), list_regions[k][1], list_regions[k][2], reverse_deltat))
                    list_lam_dAdk_h.append([lam_dAdk_h(list_AS[k][::-1][kk - 1], d_mat_k12, h[kk]) for kk in range(len(time))[1:]])
                    list_lam_dAdk33_h.append([lam_dAdk_h(list_AS[k][::-1][kk - 1], d_mat_k33, h[kk]) for kk in range(len(time))[1:]])
                    list_sens_ss_indirect = [sens_ss_indirect(list_AS[k][::-1][kk - 1], h[kk], h[kk - 1], deltat[kk - 1]) for kk in range(len(time))[1:]]
                    # list_S_ss_adj.append(sum(map(np.array,[sens_ss_indirect(list_AS[k][::-1][kk - 1], h[kk], h[kk - 1], deltat[kk - 1]) for kk in range(len(time))[1:]])))

                    # Local Sensitivity Coeffcients (for each grid element)
                    # list_S_K_adj.append(sum(map(np.array, list_lam_dAdk_h[k])))
                    # list_S_K33_adj.append(sum(map(np.array, list_lam_dAdk33_h[k])))
                    # list_S_ss_adj.append(sum(map(np.array, list_sens_ss_indirect)))
                    list_S_K_adj.append(convert_idmarray_2_gridarray(sum(map(np.array, list_lam_dAdk_h[k]))))
                    list_S_K33_adj.append(convert_idmarray_2_gridarray(sum(map(np.array, list_lam_dAdk33_h[k]))))
                    list_S_ss_adj.append(convert_idmarray_2_gridarray(sum(map(np.array, list_sens_ss_indirect))))
                if lines[i+j+1].startswith('SENSI'):
                    list_sens = []
                    list_sens_pb = []
                    for l in range(1, len(lines)):
                        if len(lines[i + j + l + 1].split()) == 1 and not lines[i + j + l + 1].split()[0] in list_of_SENSI_NAMES:
                            pbnmbr = int(lines[i + j + l + 1].split()[0])
                            list_sens.append(pbnmbr)
                            list_of_kji = []
                            list_of_lists_kji = []
                        elif len(lines[i + j + l + 1].split()) == 1 and lines[i + j + l + 1].split()[0] in list_of_SENSI_NAMES:
                            SENSCOEFNAME = lines[i + j + l + 1].split()[0]
                            list_of_kji = []
                            list_of_lists_kji = []
                        elif len(lines[i + j + l + 1].split()) == 3:
                            list_of_kji.append((int(lines[i + j + l + 1].split()[0]), int(lines[i + j + l + 1].split()[1]), int(lines[i + j + l + 1].split()[2])))
                        elif len(lines[i + j + l + 1].split()) == 0:
                            list_of_lists_kji.append(list_of_kji)
                            list_of_kji = []
                        if len(lines[i + j + l + 1].split()) == 0 and len(lines[i + j + l + 2].split()) == 1 and lines[i + j + l + 2].split()[0] in list_of_SENSI_NAMES:
                            NZONES = len(list_of_lists_kji)
                            list_sens.append([SENSCOEFNAME, NZONES, list_of_lists_kji])
                        if len(lines[i + j + l + 1].split()) == 0 and len(lines[i + j + l + 2].split()) == 1 and not lines[i + j + l + 2].split()[0] in list_of_SENSI_NAMES:
                            NZONES = len(list_of_lists_kji)
                            list_sens.append([SENSCOEFNAME, NZONES, list_of_lists_kji])
                            list_sens_pb.append(list_sens)
                            list_sens = []
                        if (len(lines[i + j + l + 1].split()) == 0 and len(lines[i + j + l + 2].split()) == 0):
                            NZONES = len(list_of_lists_kji)
                            list_sens.append([SENSCOEFNAME, NZONES, list_of_lists_kji])
                            list_sens_pb.append(list_sens)
                            break
                    print('list_sens_pb =', list_sens_pb)
                # Sensitivity of each PMs w.r.t each zone
                fo.write(stars)
                fo.write(TITLE_PM_SWH)
                fo.write(NBR_ASP % len(list_sens_pb))
                for k in range(NPROB):
                    fo.write(text_pbm %(k+1, list_regions[k][0]))
                    for kji in list_regions[k][1]:
                        fo.write("""                      ¦          %d    %d    %d          ¦\n""" % (kji[0], kji[1], kji[2]))
                    fo.write("""           ----------------------------------------------------------\n""")
                    fo.write("""                           Sensitivity Parameters:\n""")
                    for ii in range(1, len(list_sens_pb[k])):
                        if list_sens_pb[k][ii][0] == 'COND':
                            fo.write(text_K % (ii, list_sens_pb[k][ii][1]))
                        elif list_sens_pb[k][ii][0] == 'CONDV':
                            fo.write(text_Kz % (ii, list_sens_pb[k][ii][1]))
                        elif list_sens_pb[k][ii][0] == 'STOR':
                            fo.write(text_SS % (ii, list_sens_pb[k][ii][1]))
                    fo.write("""           ----------------------------------------------------------\n""")
                    fo.write(text_SC)
                    for jj in range(1, len(list_sens_pb[k])):
                        if list_sens_pb[k][jj][0] == 'COND':
                            fo.write(text_SC_K % jj)
                            list_SENS_COND = []
                            for item in list_sens_pb[k][jj][2]:
                                sens = 0.0
                                for kji in item:
                                    m = gwf.modelgrid.get_node(kji)[0]
                                    sens += list_S_K_adj[k][m]
                                list_SENS_COND.append(sens)
                            for kk in range(list_sens_pb[k][jj][1]):
                                fo.write(text_in_zone % ((kk + 1), list_SENS_COND[kk]))
                                for kji in list_sens_pb[k][jj][2][kk]:
                                    fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                                fo.write('\n')
                        elif list_sens_pb[k][jj][0] == 'CONDV':
                            fo.write(text_SC_Kz % jj)
                            list_SENS_CONDV = []
                            for item in list_sens_pb[k][jj][2]:
                                sens = 0.0
                                for kji in item:
                                    m = gwf.modelgrid.get_node(kji)[0]
                                    sens += list_S_K33_adj[k][m]
                                list_SENS_CONDV.append(sens)
                            for kk in range(list_sens_pb[k][jj][1]):
                                fo.write(text_in_zone % ((kk + 1), list_SENS_CONDV[kk]))
                                for kji in list_sens_pb[k][jj][2][kk]:
                                    fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                                fo.write('\n')
                        elif list_sens_pb[k][jj][0] == 'STOR':
                            fo.write(text_SC_SS % jj)
                            list_SENS_STORAGE = []
                            for item in list_sens_pb[k][jj][2]:
                                sens = 0.0
                                for kji in item:
                                    m = gwf.modelgrid.get_node(kji)[0]
                                    sens += list_S_ss_adj[k][m]
                                list_SENS_STORAGE.append(sens)
                            print('list_SENS_STORAGE = ', list_SENS_STORAGE)
                            for kk in range(list_sens_pb[k][jj][1]):
                                fo.write(text_in_zone % ((kk + 1), list_SENS_STORAGE[kk]))
                                for kji in list_sens_pb[k][jj][2][kk]:
                                    fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                                fo.write('\n')


            elif PMname == 'SWLSH':
                list_obs = []
                list_of_kji_LS = []
                list_of_weights_LS = []
                list_of_hobs_LS = []
                for j in range(2, len(lines)):
                    if len(lines[i+j].split()) == 1 and not lines[i+j].startswith('SENSI'):
                        nts = int(lines[i + j].split()[0])
                        list_of_kji_LS = []
                        list_of_weights_LS = []
                        list_of_hobs_LS = []
                    elif len(lines[i+j].split()) > 1:
                        list_of_kji_LS.append((int(lines[i + j].split()[0]), int(lines[i + j].split()[1]), int(lines[i + j].split()[2])))
                        list_of_weights_LS.append(float(lines[i + j].split()[3]))
                        list_of_hobs_LS.append(float(lines[i + j].split()[4]))
                    elif len(lines[i+j].split()) == 0:
                        list_obs.append([nts, list_of_kji_LS, list_of_weights_LS, list_of_hobs_LS])
                    if lines[i+j].startswith('SENSI'):
                        break
                # Solving the adjoint state problem for PM = 'SWLSH'
                for kkk in range(len(time))[1:]:
                    np.savetxt("head_k{0:04d}".format(kkk-1),h[kkk],fmt="%15.6E")
                list_AS_LS = SolveAdjointLS(len(IA) - 1, len(deltat), list_obs, h, reverse_deltat)
                # Local Sensitivity Coefficients (for each grid element)
                list_lam_dAdk_h_LS = [lam_dAdk_h(list_AS_LS[::-1][kkk-1], d_mat_k12, h[kkk]) for kkk in range(len(time))[1:]]
                list_lam_dAdk33_h_LS = [lam_dAdk_h(list_AS_LS[::-1][kkk-1], d_mat_k33, h[kkk]) for kkk in range(len(time))[1:]]
                # list_lam_dAdss_h_LS = [lam_dAdss_h(list_AS_LS[::-1][kkk-1], h[kkk], deltat[kkk-1]) for kkk in range(len(time))[1:]]
                # list_lam_drhsdss = [drhsdss(list_AS_LS[::-1][kkk-2], h[kkk], deltat[kkk-1]) for kkk in range(len(time))[1:]]
                list_sens_ss_indirect_LS = [sens_ss_indirect(list_AS_LS[::-1][kkk-1], h[kkk], h[kkk-1], deltat[kkk-1]) for kkk in range(len(time))[1:]]
                list_sens_HGHB_LS = [lam_drhs_dHGHB(list_AS_LS[::-1][kkk - 1]) for kkk in range(len(time))[1:]]
                list_sens_CGHB_indirect_LS = [sens_CGHB_indirect(list_AS_LS[::-1][kkk - 1], h[kkk]) for kkk in range(len(time))[1:]]
                # list_S_K_adj_LS = sum(map(np.array, list_lam_dAdk_h_LS))
                # list_S_K33_adj_LS = sum(map(np.array, list_lam_dAdk33_h_LS))
                # list_S_ss_adj_LS = sum(map(np.array, list_sens_ss_indirect_LS))
                list_S_K_adj_LS = convert_idmarray_2_gridarray(sum(map(np.array, list_lam_dAdk_h_LS)))
                list_S_K33_adj_LS = convert_idmarray_2_gridarray(sum(map(np.array, list_lam_dAdk33_h_LS)))
                list_S_ss_adj_LS = convert_idmarray_2_gridarray(sum(map(np.array, list_sens_ss_indirect_LS)))
                list_S_HGHB_adj_LS = convert_idmarray_2_gridarray(sum(map(np.array, list_sens_HGHB_LS)))
                list_S_CGHB_adj_LS = convert_idmarray_2_gridarray(sum(map(np.array, list_sens_CGHB_indirect_LS)))
                list_sens_QWEL_adj_LS = np.zeros((len(deltat), NODES))
                for nsp in dict_wel.keys():
                    for (k1, j1, i1) in [dict_wel[nsp][item][0] for item in range(len(dict_wel[nsp]))]:
                        list_sens_QWEL_adj_LS[nsp][list_lrc.index((k1, j1, i1))] = list_AS_LS[::-1][nsp][list_lrc.index((k1, j1, i1))]
                # list_sens1_QWEL_adj_LS = list(list(list_sens_QWEL_adj_LS[item]) for item in range(len(list_sens_QWEL_adj_LS)))
                list_S_QWEL_adj_LS = []
                for nsp in dict_wel.keys():
                    list_S_QWEL_adj_LS.append(convert_idmarray_2_gridarray(list_sens_QWEL_adj_LS[nsp]))
                list_S_RCH_adj_LS = []
                for nsp in range(len(deltat)):
                    list_S_RCH_adj_LS.append(convert_idmarray_2_gridarray(np.multiply(list_AS_LS[::-1][nsp], CELLAREA)))

                nlay,nrow,ncol = gwf.dis.nlay.data,gwf.dis.nrow.data,gwf.dis.ncol.data
                nper = sim.tdis.nper.data
                data_arrays = [list_S_K_adj_LS,list_S_K33_adj_LS,list_S_ss_adj_LS,list_S_HGHB_adj_LS,list_S_CGHB_adj_LS]
                tags = ["comp_sens_k11","comp_sens_k33","comp_sens_ss","comp_sens_hghb","comp_sens_cghb"]
                for data,tag in zip(data_arrays,tags):
                    arr = data.reshape(nlay,nrow,ncol)
                    for k in range(nlay):
                        fname = tag+"_k{0:03d}.dat".format(k)
                        np.savetxt(fname,arr[k,:,:],fmt="%15.6E")

                temporal_data_arrays = [list_AS_LS[1:],list_lam_dAdk_h_LS[::-1],list_lam_dAdk33_h_LS[::-1],list_sens_ss_indirect_LS[::-1]]
                temporal_tags = ["adjstates","sens_k","sens_k33","sens_ss"]

                for data,tag in zip(temporal_data_arrays,temporal_tags):
                    for kper in range(nper):
                        print(tag,kper,len(data))
                        arr = convert_idmarray_2_gridarray(data[kper]).reshape((nlay,nrow,ncol))
                        ttag = tag + "_kper{0:05d}".format(nper-kper-1)
                        for k in range(nlay):
                            fname = ttag+"_k{0:03d}.dat".format(k)
                            np.savetxt(fname,arr[k,:,:],fmt="%15.6E")

                for kper in range(nper):
                    arr = list_S_RCH_adj_LS[kper].reshape((nlay,nrow,ncol))
                    tag = "rech_kper{0:05d}".format(nper-kper-1)
                    for k in range(nlay):
                        fname = tag+"_k{0:03d}.dat".format(k)
                        np.savetxt(fname,arr[k,:,:],fmt="%15.6E")

                for kper in range(nper):
                    arr = list_S_QWEL_adj_LS[kper].reshape((nlay,nrow,ncol))
                    tag = "wel_kper{0:05d}".format(nper-kper-1)
                    for k in range(nlay):
                        fname = tag+"_k{0:03d}.dat".format(k)
                        np.savetxt(fname,arr[k,:,:],fmt="%15.6E")

                list_zones = []
                for k in range(1, len(lines)-i-j):
                    if len(lines[i+j+k].split()) == 1 and lines[i+j+k].split()[0] in list_of_SENSI_NAMES:
                        SENSCOEFNAME = lines[i + j + k].split()[0]
                        list_of_kji = []
                        list_of_lists_kji = []
                    elif len(lines[i + j + k].split()) == 3:
                        list_of_kji.append((int(lines[i + j + k].split()[0]), int(lines[i + j + k].split()[1]), int(lines[i + j + k].split()[2])))
                    elif len(lines[i + j + k].split()) == 0:
                        list_of_lists_kji.append(list_of_kji)
                        list_of_kji = []
                    if len(lines[i + j + k].split()) == 0 and len(lines[i + j + k + 1].split()) == 1 and lines[i + j + k + 1].split()[0] in list_of_SENSI_NAMES:
                        NZONES = len(list_of_lists_kji)
                        list_zones.append([SENSCOEFNAME, NZONES, list_of_lists_kji])
                    if (len(lines[i + j + k].split()) == 0 and len(lines[i + j + k + 1].split()) == 0):
                        NZONES = len(list_of_lists_kji)
                        list_zones.append([SENSCOEFNAME, NZONES, list_of_lists_kji])
                        break
                # Sensitivity coefficients
                fo.write(stars)
                fo.write(TITLE_PM_LS % len(list_zones))
                for k in range(len(list_zones)):
                    if list_zones[k][0] == 'COND':
                        fo.write(text_K % (k+1, list_zones[k][1]))
                    elif list_zones[k][0] == 'CONDV':
                        fo.write(text_Kz % (k+1, list_zones[k][1]))
                    elif list_zones[k][0] == 'STOR':
                        fo.write(text_SS % (k+1, list_zones[k][1]))
                    elif list_zones[k][0] == 'HGHB':
                        fo.write(text_HGHB % (k+1, list_zones[k][1]))
                    elif list_zones[k][0] == 'CGHB':
                        fo.write(text_CGHB % (k+1, list_zones[k][1]))
                fo.write(dashes)
                for k in range(len(list_zones)):
                    if list_zones[k][0] == 'COND':
                        fo.write(text_K1 % (k+1))
                        list_SENS_COND = []
                        for l in range(list_zones[k][1]):
                            sens = 0.0
                            for kji in list_zones[k][2][l]:
                                m = gwf.modelgrid.get_node(kji)[0]
                                print('!!!!!!!!!!!!!!!!!!! m = ', m)
                                sens += list_S_K_adj_LS[m]
                            list_SENS_COND.append(sens)
                        for l in range(list_zones[k][1]):
                            fo.write(text_in_zone % ((l + 1), list_SENS_COND[l]))
                            for kji in list_zones[k][2][l]:
                                fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                            fo.write('\n')
                    elif list_zones[k][0] == 'CONDV':
                        fo.write(text_Kz1 % (k+1))
                        list_SENS_CONDV = []
                        for l in range(list_zones[k][1]):
                            sens = 0.0
                            for kji in list_zones[k][2][l]:
                                m = gwf.modelgrid.get_node(kji)[0]
                                sens += list_S_K33_adj_LS[m]
                            list_SENS_CONDV.append(sens)
                        for l in range(list_zones[k][1]):
                            fo.write(text_in_zone % ((l + 1), list_SENS_CONDV[l]))
                            for kji in list_zones[k][2][l]:
                                fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                            fo.write('\n')
                    elif list_zones[k][0] == 'STOR':
                        fo.write(text_SS1 % (k+1))
                        list_SENS_STORAGE = []
                        for l in range(list_zones[k][1]):
                            sens = 0.0
                            for kji in list_zones[k][2][l]:
                                m = gwf.modelgrid.get_node(kji)[0]
                                sens += list_S_ss_adj_LS[m]
                            list_SENS_STORAGE.append(sens)
                        for l in range(list_zones[k][1]):
                            fo.write(text_in_zone % ((l + 1), list_SENS_STORAGE[l]))
                            for kji in list_zones[k][2][l]:
                                fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                            fo.write('\n')
                    elif list_zones[k][0] == 'HGHB':
                        fo.write(text_HGHB1 % (k+1))
                        list_SENS_HGHB = []
                        for l in range(list_zones[k][1]):
                            sens = 0.0
                            for kji in list_zones[k][2][l]:
                                m = gwf.modelgrid.get_node(kji)[0]
                                sens += list_S_HGHB_adj_LS[m]
                            list_SENS_HGHB.append(sens)
                        for l in range(list_zones[k][1]):
                            fo.write(text_in_zone % ((l + 1), list_SENS_HGHB[l]))
                            for kji in list_zones[k][2][l]:
                                fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                            fo.write('\n')
                    elif list_zones[k][0] == 'CGHB':
                        fo.write(text_CGHB1 % (k+1))
                        list_SENS_CGHB = []
                        for l in range(list_zones[k][1]):
                            sens = 0.0
                            for kji in list_zones[k][2][l]:
                                m = gwf.modelgrid.get_node(kji)[0]
                                sens += list_S_CGHB_adj_LS[m]
                            list_SENS_CGHB.append(sens)
                        for l in range(list_zones[k][1]):
                            fo.write(text_in_zone % ((l + 1), list_SENS_CGHB[l]))
                            for kji in list_zones[k][2][l]:
                                fo.write('%d   %d   %d\n' % (kji[0], kji[1], kji[2]))
                            fo.write('\n')

f.close()
fo.close()

epsilon = -1.575
print('{:.26f}'.format(1.0 + epsilon))
# print(list_S_K_adj_LS)
J = JLS(h, list_obs)
print('{:.40f}'.format(J))
print(J)
print((J - 12.0920734167638013190071433200500905513763) / epsilon)
print((J - 12.092073416763801) / epsilon)
# print(SENS_SWK_COND)

# -------------- Postprocessing
# L1 = 250.0 * 20
# L2 = 250.0 * 40
# x = np.linspace(0, L1, gwf.modelgrid.ncol)
# y = np.linspace(0, L2, gwf.modelgrid.nrow)
# y = y[::-1]
# array_S_K_adj_LS = np.array(list_S_K_adj_LS)
# S_K_adj_LS = np.reshape(array_S_K_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
#
# array_S_ss_adj_LS = np.array(list_S_ss_adj_LS)
# S_ss_adj_LS = np.reshape(array_S_ss_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# fig,ax = plt.subplots(1, 1)
# arr = S_K_adj_LS[0]
# arr[arr == -999.] = np.nan
# cb = ax.imshow(S_K_adj_LS[0])
# plt.colorbar(cb, ax=ax)
#
# -------------- Postprocessing for flow rates
L1 = 250.0 * 20
L2 = 250.0 * 40
x = np.linspace(0, L1, gwf.modelgrid.ncol)
y = np.linspace(0, L2, gwf.modelgrid.nrow)
y = y[::-1]
nsp = 0
array_S_QWEL_adj_LS = np.array(list_S_QWEL_adj_LS[nsp])
S_QWEL_adj_LS = np.reshape(array_S_QWEL_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))

array_S_K_adj_LS = np.array(list_S_K_adj_LS)
S_K_adj_LS = np.reshape(array_S_K_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))

QWEL_adj_LS = []
for nsp in range(len(deltat)):
    QWEL_adj_LS.append(np.reshape(np.array(list_S_QWEL_adj_LS[nsp]), (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol)))


# fig,ax = plt.subplots(1, 1)
# arr = S_QWEL_adj_LS[0]
# arr[arr == -999.] = np.nan
# cb = ax.imshow(S_QWEL_adj_LS[0])
# plt.colorbar(cb, ax=ax)
# plt.show()

# list_S_W1_AS = []
# list_S_W2_AS = []
# list_S_W3_AS = []
# list_S_W4_AS = []
# list_S_W5_AS = []
# list_S_W6_AS = []
# for nsp in dict_wel.keys():
#     list_S_W1_AS.append(list_S_QWEL_adj_LS[nsp][gwf.modelgrid.get_node((0, 8, 15))][0])
#     list_S_W2_AS.append(list_S_QWEL_adj_LS[nsp][gwf.modelgrid.get_node((0, 10, 12))][0])
#     list_S_W3_AS.append(list_S_QWEL_adj_LS[nsp][gwf.modelgrid.get_node((0, 19, 13))][0])
#     list_S_W4_AS.append(list_S_QWEL_adj_LS[nsp][gwf.modelgrid.get_node((0, 25, 9))][0])
#     list_S_W5_AS.append(list_S_QWEL_adj_LS[nsp][gwf.modelgrid.get_node((0, 28, 5))][0])
#     list_S_W6_AS.append(list_S_QWEL_adj_LS[nsp][gwf.modelgrid.get_node((0, 33, 11))][0])
#
# list_S_W1_per = [-0.12899028651941444, -3.352644060074475, -3.1858525720974233, -2.839158574160338, -1.7176668533647033]
# list_S_W2_per = [-0.12842885323253644, -3.275731987231416, -3.1144459074110995, -2.779582124683678, -1.6832728675826674]
# list_S_W3_per = [-0.11805203574704863, -2.7081197463135878, -2.58621799957472, -2.3168100555975246, -1.4339331057743532]
# list_S_W4_per = [-0.10610690225825067, -2.161670119604266, -2.071244377806169, -1.8570105784520838, -1.1675971990623664]
# list_S_W5_per = [-0.09622969671649378, -1.8884046713501355, -1.807216651220636, -1.624014042372774, -1.0163763677300257]
# list_S_W6_per = [-0.07294084492196631, -1.2245356937052378, -1.174771127054283, -1.0591242999506945, -0.6744576489523058]
#
# plt.plot(dict_wel.keys(), list_S_W1_AS, 'b-', dict_wel.keys(), list_S_W1_per, 'r-')
# plt.show()
# plt.plot(dict_wel.keys(), list_S_W2_AS, 'b-', dict_wel.keys(), list_S_W2_per, 'r-')
# plt.show()
# plt.plot(dict_wel.keys(), list_S_W3_AS, 'b-', dict_wel.keys(), list_S_W3_per, 'r-')
# plt.show()
# plt.plot(dict_wel.keys(), list_S_W4_AS, 'b-', dict_wel.keys(), list_S_W4_per, 'r-')
# plt.show()
# plt.plot(dict_wel.keys(), list_S_W5_AS, 'b-', dict_wel.keys(), list_S_W5_per, 'r-')
# plt.show()
# plt.plot(dict_wel.keys(), list_S_W6_AS, 'b-', dict_wel.keys(), list_S_W6_per, 'r-')
# plt.show()


# ##### CONDUCTIVITY
# list_S_K_per_LS = []
# with open('sens_per.DAT', 'r') as fp:
#     lines = fp.readlines()
#     for row in lines:
#         list_S_K_per_LS.append(float(row))
#
# array_S_K_per_LS = np.array(list_S_K_per_LS)
# array_S_ss_adj_LS = np.array(list_S_ss_adj_LS)
# S_K_per_LS = np.reshape(array_S_K_per_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# for i in range(gwf.modelgrid.nlay):
#     # plot_colorbar_arrays(S_K_adj_LS[i], S_K_per_LS[i], i+1, "Hydraulic conductivity")
#     plot_colorbar_arrays(S_K_adj_LS[i], S_K_per_LS[i], i + 1, "Hydraulic conductivity")
#     # contour_intervals = np.array([-0.004, -0.002, 0.0, 0.002, 0.004])
#     contour_intervals = np.array([-0.01, -0.001, 0.0003, 0.003, 0.01])
#     plot_colorbar_sensitivity(x, y, S_K_adj_LS[i], S_K_per_LS[i], i + 1, "Hydraulic conductivity", contour_intervals)
#
# ##### STORAGE COEFFICIENT
# list_S_ss_per_LS = []
# with open('sens_sto.DAT', 'r') as fp:
#     lines = fp.readlines()
#     for row in lines:
#         list_S_ss_per_LS.append(float(row))
#
# array_S_ss_per_LS = np.array(list_S_ss_per_LS)
# S_ss_per_LS = np.reshape(array_S_ss_per_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# S_ss_adj_LS = np.reshape(array_S_ss_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# for i in range(gwf.modelgrid.nlay):
#     plot_colorbar_arrays(S_ss_adj_LS[i], S_ss_per_LS[i], i + 1, "Storage coefficient")
#     contour_intervals = np.array([4.0e3, 5.0e3, 1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4])
#     plot_colorbar_sensitivity(x, y, S_ss_adj_LS[i], S_ss_per_LS[i], i + 1, "Storage coefficient", contour_intervals)
#
#
# ##### HGHB
array_S_HGHB_adj_LS = np.array(list_S_HGHB_adj_LS)
S_HGHB_adj_LS = np.reshape(array_S_HGHB_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# for i in range(gwf.modelgrid.nlay):
#     plot_colorbar_arrays(S_HGHB_adj_LS[i], S_HGHB_adj_LS[i], i + 1, "HGHB")

##### CGHB
# array_S_CGHB_adj_LS = np.array(list_S_CGHB_adj_LS)
# S_CGHB_adj_LS = np.reshape(array_S_CGHB_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# for i in range(gwf.modelgrid.nlay):
#     plot_colorbar_arrays(S_CGHB_adj_LS[i], S_CGHB_adj_LS[i], i + 1, "CGHB")

##### model
array_model = S_HGHB_adj_LS
for item in range(gwf.modelgrid.nrow):
    array_model[0][item][14] = 1.0
for (kk, jj, ii) in [list_ghb[item][0] for item in range(len(list_ghb))]:
    array_model[kk][jj][ii] = 2.0
for (kw, jw, iw) in [dict_wel[0][item][0] for item in range(len(dict_wel[0]))]:
    array_model[kw][jw][iw] = 3.0
for (ko, jo, io) in list_obs[0][1]:
    array_model[ko][jo][io] = 4.0

#plot_colorbar_arrays(array_model[0], array_model[0], 1, "model")

# # HGHB
# list_SENSITIVITY_HGHB_adj = [-0.05250082327769279, -0.07040794412328386, -0.13323025770515223, -0.36620150041616906, -0.16672358950375796, -0.17514838139596708, -0.3980402057607099,  -0.22583357106631438, -0.389557768210818,  -0.16444088099663287]
# list_SENSITIVITY_HGHB_per = [-0.05250072204505817, -0.0704078565267686,  -0.1332301894678236,  -0.36620072440433066, -0.16672347800846493, -0.17514828645469388, -0.39803947401787976, -0.22583346933795687, -0.3895571101518509, -0.16444091899104613]
# # CGHB
# list_SENSITIVITY_CGHB_adj = [0.00018212360697616656, 0.0002364194838702649,  0.0004246445111825009,  0.001097597558679414,  0.0004693947868245202,  0.00046833304916019114, 0.00104478318473894,   0.0006196862488358629, 0.0012084734185784929, 0.0005992582494902393]
# list_SENSITIVITY_CGHB_per = [0.0001821324678376511,  0.00023642960605181024, 0.00042466335210227565, 0.0010975968705346467, 0.00046940876582571685, 0.00046832740859973195, 0.0010447673202764107, 0.00061969102865029,   0.0012084763665955378, 0.0005992576593679846]

# list_S_ss_per_LS = []
# with open('sens_sto.DAT', 'r') as fp:
#     lines = fp.readlines()
#     for row in lines:
#         list_S_ss_per_LS.append(float(row))
#
# array_S_ss_per_LS = np.array(list_S_ss_per_LS)
# S_ss_per_LS = np.reshape(array_S_ss_per_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))

# ##### Recharge Rate
# S_RCH_adj_LS = []
# for nsp in range(len(deltat)):
#     S_RCH_adj_LS.append(np.reshape(list_S_RCH_adj_LS[nsp], (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol)))
#
# list_S_RCH_per_LS = []
# S_RCH_per_LS = []
# # for nsp in range(len(deltat)):
# for nsp in range(gwf.modeltime.nper):
#     mylist = []
#     with open('sens_rcha_%d.dat' %nsp, 'r') as fp:
#         lines = fp.readlines()
#         for row in lines:
#             mylist.append(float(row))
#     fp.close()
#     list_S_RCH_per_LS.append(np.array(mylist))
#     S_RCH_per_LS.append(np.reshape(convert_idmarray_2_gridarray(list_S_RCH_per_LS[nsp]), (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol)))

# for nsp in range(gwf.modeltime.nper):
#     plot_colorbar_arrays_RCH(S_RCH_adj_LS[nsp][0], S_RCH_per_LS[nsp][0], nsp + 1, "Recharge rate")
#     lst = np.arange(np.nanmin(S_RCH_adj_LS[nsp][0]), np.nanmax(S_RCH_adj_LS[nsp][0]), (np.nanmax(S_RCH_adj_LS[nsp][0]) - np.nanmin(S_RCH_adj_LS[nsp][0]))/11).tolist()
#     contour_intervals = np.array(lst)
#     plot_colorbar_sensitivity_RCH(S_RCH_adj_LS[nsp][0], S_RCH_per_LS[nsp][0], nsp + 1, "Recharge rate", contour_intervals)


#
# array_S_ss_per_LS = np.array(list_S_ss_per_LS)
# S_ss_per_LS = np.reshape(array_S_ss_per_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# S_ss_adj_LS = np.reshape(array_S_ss_adj_LS, (gwf.modelgrid.nlay, gwf.modelgrid.nrow, gwf.modelgrid.ncol))
# for i in range(gwf.modelgrid.nlay):
#     plot_colorbar_arrays(S_ss_adj_LS[i], S_ss_per_LS[i], i + 1, "Storage coefficient")
#     contour_intervals = np.array([4.0e3, 5.0e3, 1.0e4, 2.0e4, 3.0e4, 4.0e4, 5.0e4])
#     plot_colorbar_sensitivity(x, y, S_ss_adj_LS[i], S_ss_per_LS[i], i + 1, "Storage coefficient", contour_intervals)

# plot_colorbar_arrays(S_ss_adj_LS[0], S_ss_per_LS[0])

# -----------------------------
# plot_colorbar_sensitivity_Single(x, y, S_K_adj_LS[0])


# J = JSWH(list_regions[0][0], h, list_regions[0][1], list_regions[0][2], deltat[0])
# print('{:.40f}'.format(J))
# print((J - 714.2416168981606006127549335360527038574219) / epsilon)

# print('list_obs', list_obs)
# print(dFdhLS(len(IA) - 1, len(deltat), list_obs, h))
# print(list_lam_dAdk_h_LS)


# ##Sensitivities
# ### k
# d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k123 = d_amat_k()
# list_lam_dAdk_h = [lam_dAdk_h(list_AS[::-1][k-1], d_mat_k123, h[k]) for k in range(len(time))[1:]]
# list_S_adj = sum(map(np.array, list_lam_dAdk_h))
# print('dJdk', list_S_adj)
#
# ### ss
# # list_lam_dAdss_h = [lam_dAdss_h(list_AS[::-1][k-1], h[k], deltat[k-1]) for k in range(len(time))[1:]]
# # list1_Sss_adj = sum(map(np.array, list_lam_dAdss_h))
# # list_drhsdss = [drhsdss(list_AS[::-1][k-1], h_old[k], deltat[k-1]) for k in range(len(time))[1:]]
# # list2_Sss_adj = sum(map(np.array, list_drhsdss))
# # list_Sss_adj = [list1_Sss_adj[item] + list2_Sss_adj[item] for item in range(len(list1_Sss_adj))]
# # print('dJdss',list_Sss_adj)
#
# # J = h[len(deltat)][gwf.modelgrid.get_node((0, int(Nrow / 2), int(Ncol / 2)))[0]] / deltat[-1]
# # print('{:.40f}'.format(J))
# #
# print((J - 4.7500000000195914395817453623749315738678) / epsilon)
# J = JLS(h, "OBSMF6")
# # J = JSWH(1, h, list_of_kji, list_of_weights, deltat[0])
# print('{:.40f}'.format(J))
# print((J - 97.7303349562657217575178947299718856811523) / epsilon)
#
# #-----------------------------------------------------------------------------------------------------------------------
# # ## Post-Process Head Results
# # Grid
# x = np.linspace(0, L1, Ncol)
# y = np.linspace(0, L2, Nrow)
# y = y[::-1]
# # Adjoint states
# lam = []
# lam_3d = []
# vmin = []
# vmax = []
# for i in range(len(t)):
#     lam.append(list(list_AS[time[::-1].index(t[i])]))
#     lam[i] = np.array(lam[i])
#     lam_3d.append(np.reshape(lam[i], (Nlay, Nrow, Ncol)))
#     vmin.append(lam_3d[i].min())
#     vmax.append(lam_3d[i].max())
# plot_contour_list_Single(x, y, t, lam_3d)
# # plot_colorbar_list(x, y, t, lam_anal, lam_3d, vmin, vmax)
#
# array_S_adj = np.array(list_S_adj)
# S_adj = np.reshape(array_S_adj, (Nlay, Nrow, Ncol))
# plot_colorbar_sensitivity_Single(x, y, S_adj[0])