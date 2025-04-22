from freyberg6 import *


#*******************
def getlrc():
    list_lrc = []
    for m in range(gwf.modelgrid.nnodes):
        (km, jm, im) = gwf.modelgrid.get_lrc(m)[0]
        if gwf.modelgrid.idomain[km][jm][im] == 0:
            continue
        else:
            list_lrc.append((km, jm, im))
    return list_lrc

list_lrc = getlrc()

#*******************
def getnodesnumber():
    list_nodesnumber = []
    for m in range(gwf.modelgrid.nnodes):
        (km, jm, im) = gwf.modelgrid.get_lrc(m)[0]
        if gwf.modelgrid.idomain[km][jm][im] == 0:
            continue
        else:
            list_nodesnumber.append(m)
    return list_nodesnumber

print("list_nodesnumber = ", getnodesnumber())


# ------------------------------------------------------------------------------------------------------------------------
# def RightHandSide(mylist, Nlay, Nrow, Ncol, SS, A, TP, BT, XX, dt):
def RightHandSide(mylist, SS, A, TP, BT, XX, dt):
    list_rhs = []
    mylist_cell = []
    mylist_head = []
    for ii in range(len(mylist)):
        mylist_cell.append(mylist[ii][0])
        mylist_head.append(mylist[ii][1])
    ii = -1
    for k in range(Nlay):
        for j in range(Nrow):
            for i in range(Ncol):
                ii += 1
                # if (k, j, i) in [item[0] for item in mylist]:
                if (k, j, i) in mylist_cell:
                    list_rhs.append(mylist_head[mylist_cell.index((k, j, i))])
                else:
                    list_rhs.append(- SS[ii] * A[ii] * (TP[ii] - BT[ii]) * XX[ii] / dt)
    return list_rhs

# def dJdh(listheads, listtimes, i, j, k, t):
#     for ii in range(len(listheads) -1 ):
#         if t >= listtimes(ii) and t <= listtimes(ii + 1):
#            t1 = listtimes(ii)
#            t2 = listtimes(ii + 1)
#            w1 = (t2 - t) / (t2 - t1)
#            w2 = (t - t1) / (t2 - t1)
# ------------------------------------------------------------------------------------------------------------------------
def drhsdh(S, A, TP, BT, dt):
    # mylist = []
    myarray = np.zeros(len(S))
    for ii in range(len(S)):
        # mylist.append(- S[ii] * A[ii] * (TP[ii] - BT[ii]) / dt)
        myarray[ii] = - S[ii] * A[ii] * (TP[ii] - BT[ii]) / dt
    return myarray
# ------------------------------------------------------------------------------------------------------------------------
def dFdh(k, j, i, ntime, nts, ncells, dt):
    mylist = []
    m = gwf.modelgrid.get_node((k, j, i))[0]
    # m = 1
    for ii in range(nts):
        mylist.append(np.zeros(ncells))
        if ii == ntime:
            # mylist[ii][m] = 1.0
            # mylist[ii][m] = (1./dt) * (1 / (ncells))
            mylist[ii][m] = - 1. / dt
    return mylist
# ------------------------------------------------------------------------------------------------------------------------
def dFdhSWH(ncells, current_ts, nts, list_of_kji, list_of_weights, dt):
    # Sum of Weighted Head at time step number nts
    mylist = []
    for ii in range(nts):
        mylist.append(np.zeros(ncells))
        if ii == current_ts:
           iii = -1
           for (k, j, i) in list_of_kji:
               iii += 1
               m = gwf.modelgrid.get_node((k, j, i))[0]
               mylist[ii][m] = list_of_weights[iii]
    return mylist
# ------------------------------------------------------------------------------------------------------------------------
# def dFdhLS(ncells, nts, head, OBSFILE):
#     myarray = np.zeros(ncells)
#     # Weighted Least Square differences
#     with open(OBSFILE, 'r') as f:
#         obslines = f.readlines()
#         for i in range(len(obslines)):
#             if len(obslines[i].split()) == 2 and int(obslines[i][5:10]) == nts:
#                 for ii in range(ncells)[1:]:
#                     if len(obslines[i + ii].split()) != 0:
#                         km = int(obslines[i + ii][1:5])
#                         jm = int(obslines[i + ii][6:10])
#                         im = int(obslines[i + ii][11:15])
#                         w = float(obslines[i + ii][16:23])
#                         head_m = float(obslines[i + ii][24:45])
#                         m = gwf.modelgrid.get_node((km, jm, im))[0]
#                         myarray[m] = 2.0 * w * (head[nts+1][m] - head_m)
#                         # print(int(obslines[i + ii][5:10]))
#                     else:
#                         break
#     f.close()
#     return myarray

# def dFdhLS(ncells, nts, list_nbr_TS, list_of_kji, list_of_weights, head):
def dFdhLS(ncells, nts, list_obs, head):
    mylist = []
    # Weighted Least Square differences
    list_nbr_TS = [item[0] for item in list_obs]
    list_of_kji = [item[1] for item in list_obs]
    list_of_weights = [item[2] for item in list_obs]
    list_of_hobs = [item[3] for item in list_obs]
    for n in range(nts):
        myarray = np.zeros(ncells)
        if n in list_nbr_TS:
            # idx = list_nbr_TS[n]
            idx = list_nbr_TS.index(n)
            ii = -1
            for (k, j, i) in list_of_kji[idx]:
                ii += 1
                m = gwf.modelgrid.get_node((k, j, i))[0]
                mm = getnodesnumber().index(m)
                w = list_of_weights[idx][ii]
                head_obs = list_of_hobs[idx][ii]
                # print("m = ", m)
                # print("mm = ", mm)
                myarray[mm] = 2.0 * w * (head[n + 1][mm] - head_obs)
        mylist.append(myarray)
    return mylist
# ------------------------------------------------------------------------------------------------------------------------
def JSWH(current_ts, head, list_of_kji, list_of_weights, dt):
    # PM: Sum of Weighted Head at time step number nts
    Sum = 0.0
    iii = -1
    for (k, j, i) in list_of_kji:
        iii += 1
        m = gwf.modelgrid.get_node((k, j, i))[0]
        Sum += list_of_weights[iii] * head[current_ts+1][m]
    return Sum
# ------------------------------------------------------------------------------------------------------------------------
# def JLS(head, OBSFILE):
#     # PM: Weighted Least Square differences
#     Sum = 0.0
#     with open(OBSFILE, 'r') as f:
#         obslines = f.readlines()
#         for i in range(len(obslines)):
#             if len(obslines[i].split()) == 2:
#                 nts = int(obslines[i][5:10])
#             elif len(obslines[i].split()) == 6:
#                 km = int(obslines[i][1:5])
#                 jm = int(obslines[i][6:10])
#                 im = int(obslines[i][11:15])
#                 w = float(obslines[i][16:23])
#                 head_m = float(obslines[i][24:45])
#                 m = gwf.modelgrid.get_node((km, jm, im))[0]
#                 Sum += w * (head[nts+1][m] - head_m) ** 2
#     f.close()
#     return Sum

def JLS(head, list_obs):
    # PM: Weighted Least Square differences
    list_nbr_TS = [item[0] for item in list_obs]
    list_of_kji = [item[1] for item in list_obs]
    list_of_weights = [item[2] for item in list_obs]
    list_of_hobs = [item[3] for item in list_obs]
    Sum1 = 0.0
    for n in list_nbr_TS:
        # idx = list_nbr_TS[n]
        idx = list_nbr_TS.index(n)
        ii = -1
        for (k, j, i) in list_of_kji[idx]:
            ii += 1
            m = gwf.modelgrid.get_node((k, j, i))[0]
            print('m = ', m)
            mm = getnodesnumber().index(m)
            w = list_of_weights[idx][ii]
            head_obs = list_of_hobs[idx][ii]
            Sum1 += w * (head[n + 1][mm] - head_obs) ** 2.0
    return Sum1
# ------------------------------------------------------------------------------------------------------------------------
def multiplyarray(A1, A2):
    n = len(A1)
    product = np.zeros(n)
    for i in range(n):
        product[i] = A1[i] * A2[i]
    return product

def substructarray(A1, A2):
    # n = len(A1)
    mylist = [item1 - item2 for item1, item2 in zip(A1, A2)]
    return mylist
# ------------------------------------------------------------------------------------------------------------------------
def SolveAdjointHeadAtPoint(k, j, i, ntime, nts, ncells, rev_dt):
    lam = np.zeros(ncells)
    list_AS = [lam]
    l = list(range(len(rev_dt)))
    l = l[::-1]
    for kk in l:
        list_drhs = drhsdh(STORAGE, CELLAREA, CELLTOP, CELLBOT, rev_dt[kk])
        array_lam = lam
        # print('lam', list(lam))
        l1 =  multiplyarray(list_drhs, array_lam)
        l2 = dFdh(k, j, i, ntime, nts, ncells, rev_dt[::-1][ntime])[kk]
        # print(kk, list(l2))
        # l2 =  dFdh(k, j, i, ntime, nts, ncells)[kk] / rev_dt[kk]
        rhs1 = substructarray(l1, l2)
        A = csr_matrix((MAT[kk], JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
        A = csr_matrix(A)
        At = A.transpose()
        lam = spsolve(At, rhs1)
        # lam = spsolve(A, rhs1)
        list_AS.append(lam)
    return list_AS
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# def SolveAdjointLS(ncells, head, OBSFILE, rev_dt):
#     lam = np.zeros(ncells)
#     list_AS = [lam]
#     l = list(range(len(rev_dt)))
#     l = l[::-1]
#     print('-----------------', l)
#     for kk in l:
#         list_drhs = drhsdh(STORAGE, CELLAREA, CELLTOP, CELLBOT, rev_dt[kk])
#         array_lam = lam
#         # print('lam', list(lam))
#         l1 =  multiplyarray(list_drhs, array_lam)
#         l2 = - dFdhLS(ncells, kk, head, OBSFILE)
#         # print(kk, list(l2))
#         # l2 =  dFdh(k, j, i, ntime, nts, ncells)[kk] / rev_dt[kk]
#         rhs1 = substructarray(l1, l2)
#         A = csr_matrix((MAT[kk], JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
#         A = csr_matrix(A)
#         At = A.transpose()
#         lam = spsolve(At, rhs1)
#         # lam = spsolve(A, rhs1)
#         list_AS.append(lam)
#     return list_AS
def SolveAdjointSWH(ncells, current_ts, nts, list_of_kji, list_of_weights, rev_dt):
    lam = np.zeros(ncells)
    list_AS = [lam]
    l = list(range(len(rev_dt)))
    l = l[::-1]
    for kk in l:
        list_drhs = drhsdh(STORAGE, CELLAREA, CELLTOP, CELLBOT, rev_dt[len(rev_dt)-kk-1])
        array_lam = lam
        l1 =  multiplyarray(list_drhs, array_lam)
        l2 = dFdhSWH(ncells, current_ts, nts, list_of_kji, list_of_weights, rev_dt[::-1][current_ts])[kk]
        rhs1 = substructarray(l1, l2)
        A = csr_matrix((MAT[kk], JA_p.copy(), IA_p.copy()), shape=(len(IA) - 1, len(IA) - 1)).toarray()
        A = csr_matrix(A)
        At = A.transpose()
        lam = spsolve(At, rhs1)
        list_AS.append(lam)
    return list_AS

def SolveAdjointLS(ncells, nts, list_obs, head, rev_dt):
    lam = np.zeros(ncells)
    list_AS = [lam]
    l = list(range(len(rev_dt)))
    l = l[::-1]
    for kk in l:
        list_drhs = drhsdh(STORAGE, CELLAREA, CELLTOP, CELLBOT, rev_dt[len(rev_dt)-kk-1])
        array_lam = lam
        l1 =  multiplyarray(list_drhs, array_lam)
        l2 =  dFdhLS(ncells, nts, list_obs, head)[kk]
        rhs1 = substructarray(l1, l2)
        np.savetxt("rhs_kper{0:04d}.dat".format(kk),rhs1,fmt="%15.6E")
        np.savetxt("amat_kper{0:04d}.dat".format(kk), MAT[kk], fmt="%15.6E")
        A = csr_matrix((MAT[kk], JA_p.copy(), IA_p.copy()), shape=(len(IA) - 1, len(IA) - 1))
        A = csr_matrix(A)
        At = A.transpose()
        np.savetxt("amattodense_kper{0:04d}.dat".format(kk), At.todense(), fmt="%15.6E")
        lam = spsolve(At, rhs1)
        # lam = spsolve(A, rhs1)
        list_AS.append(lam)
    return list_AS
# ------------------------------------------------------------------------------------------------------------------------
def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
    return 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)


def d_amat_k():
    d_mat_k11 = np.zeros(len(MAT[0]))
    d_mat_k22 = np.zeros(len(MAT[0]))
    d_mat_k33 = np.zeros(len(MAT[0]))
    d_mat_k12 = np.zeros(len(MAT[0]))
    for nn in range(len(IA)-1):
        # (k, j, i) = gwf.modelgrid.get_lrc(nn)[0]
        # (k, j, i) = getlrc()[nn]
        (k, j, i) = list_lrc[nn]
        if (k, j, i) in list_ch:
            for ij in range(IAC[nn]-1):
                d_mat_k11[IA_p[nn] + ij] = 0.0
                d_mat_k22[IA_p[nn] + ij] = 0.0
                d_mat_k33[IA_p[nn] + ij] = 0.0
                d_mat_k12[IA_p[nn] + ij] = d_mat_k11[IA_p[nn] + ij] + d_mat_k22[IA_p[nn] + ij]
        else:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            ii = 1
            for nc in range(IAC[nn])[1:]:
                # (knc, jnc, inc) = gwf.modelgrid.get_lrc(JA_p[IA_p[nn]+nc])[0]
                # (knc, jnc, inc) = getlrc()[JA_p[IA_p[nn] + nc]]
                (knc, jnc, inc) = list_lrc[JA_p[IA_p[nn] + nc]]
                if knc == k - 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = derivative_conductance_k1(K33[nn], K33[JA_p[IA_p[nn]+nc]], CELLTOP[nn] - CELLBOT[nn], CELLTOP[JA_p[IA_p[nn]+nc]] - CELLBOT[JA_p[IA_p[nn]+nc]], DELR[i], DELC[j])
                    d_mat_k12[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] 
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif jnc == j - 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = derivative_conductance_k1(K22[nn], K22[JA_p[IA_p[nn]+nc]], DELC[j], DELC[j-1], DELR[i], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k12[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] 
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif inc == i - 1:
                    d_mat_k11[IA_p[nn] + ii] = derivative_conductance_k1(K11[nn], K11[JA_p[IA_p[nn]+nc]], DELR[i], DELR[i-1], DELC[j], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k12[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif inc == i + 1:
                    d_mat_k11[IA_p[nn] + ii] = derivative_conductance_k1(K11[nn], K11[JA_p[IA_p[nn]+nc]], DELR[i], DELR[i+1], DELC[j], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k12[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif jnc == j + 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = derivative_conductance_k1(K22[nn], K22[JA_p[IA_p[nn]+nc]], DELC[j], DELC[j+1], DELR[i], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k12[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif knc == k + 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = derivative_conductance_k1(K33[nn], K33[JA_p[IA_p[nn]+nc]], CELLTOP[nn] - CELLBOT[nn], CELLTOP[JA_p[IA_p[nn]+nc]] - CELLBOT[JA_p[IA_p[nn]+nc]], DELR[i], DELC[j])
                    d_mat_k12[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
            d_mat_k11[IA_p[nn]] = - sum1
            d_mat_k22[IA_p[nn]] = - sum2
            d_mat_k33[IA_p[nn]] = - sum3
            d_mat_k12[IA_p[nn]] = d_mat_k11[IA_p[nn]] + d_mat_k22[IA_p[nn]]
    return d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k12

def lam_dAdk_h(lam, dAdk, h):
    my_list = []
    for k in range(len(lam)):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(IAC[k]):
            sum1 += dAdk[IA_p[k] + j] * h[JA_p[IA_p[k] + j]]
        sum1 = lam[k] * sum1
        for j in list(range(IAC[k]))[1:]:
            # (kk, jj, ii) = gwf.modelgrid.get_lrc(JA_p[IA_p[k] + j])[0]
            # (kk, jj, ii) = getlrc()[JA_p[IA_p[k] + j]]
            #(kk, jj, ii) = list_lrc[JA_p[IA_p[k] + j]]
            #if (kk, jj, ii) in list_ch:
            #    sum2 += 0.0
            #else:
            sum2 += lam[JA_p[IA_p[k] + j]] * dAdk[IA_p[k] + j] * (h[k] - h[JA_p[IA_p[k] + j]])
        sum = sum1 + sum2
        my_list.append(sum)
    return my_list


def lam_dAdss_h(lam, hh, delt):
    mylist = []
    for nn in range(len(lam)):
        # (k, j, i) = gwf.modelgrid.get_lrc(nn)[0]
        # (k, j, i) = getlrc()[nn]
        #(k, j, i) = list_lrc[nn]
        #if (k, j, i) in list_ch:
        #    mylist.append(0.0)
        #else:
        mylist.append(- lam[nn] * hh[nn] * CELLAREA[nn] * (CELLTOP[nn] - CELLBOT[nn]) / delt)
    return mylist

def drhsdss(lam, hh, delt):
    mylist = []
    for nn in range(len(lam)):
        # (k, j, i) = gwf.modelgrid.get_lrc(nn)[0]
        # (k, j, i) = getlrc()[nn]
        #(k, j, i) = list_lrc[nn]
        #if (k, j, i) in list_ch:
        #    mylist.append(0.0)
        #else:
        mylist.append(- lam[nn] * CELLAREA[nn] * (CELLTOP[nn] - CELLBOT[nn]) * hh[nn] / delt)
    return mylist

def sens_ss_indirect(lam, hh, hhold, delt):
    mylist = [lam_dAdss_h(lam, hh, delt)[nn] - drhsdss(lam, hhold, delt)[nn] for nn in range(len(lam_dAdss_h(lam, hh, delt)))]
    return mylist

def lam_drhs_dHGHB(lam):
    # This function returns "- lambda * drhsdHGHB", where drhsdHGHB = -CGHB
    # drhsdHGHB = Derivative of RHS (w.r.t. HGHB)
    mylist = []
    list_kji_GHB = [list_ghb[m][0] for m in range(len(list_ghb))]
    for nn in range(len(lam)):
        (k, j, i) = list_lrc[nn]
        if (k, j, i) in list_kji_GHB:
            ii = list_kji_GHB.index((k, j, i))
            mylist.append(lam[nn] * list_ghb[ii][2])
        else:
            mylist.append(0.0)
    return mylist

def lam_dAdCGHB_h(lam, hh):
    # This function returns "lambda * dAdCGHB * h", where dAdCGHB = - 1
    # dAdCGHB = derivative of conductance matrix w.r.t. CGHB
    mylist = []
    list_kji_GHB = [list_ghb[m][0] for m in range(len(list_ghb))]
    for nn in range(len(lam)):
        (k, j, i) = list_lrc[nn]
        if (k, j, i) in list_kji_GHB:
            mylist.append(- lam[nn] * hh[nn])
        else:
            mylist.append(0.0)
    return mylist

def lam_drhs_dCGHB(lam):
    # This function returns "- lambda * drhsdCGHB", where drhsdHGHB = -HGHB
    # drhsdCGHB = Derivative of RHS (w.r.t. CGHB)
    mylist = []
    list_kji_GHB = [list_ghb[m][0] for m in range(len(list_ghb))]
    for nn in range(len(lam)):
        (k, j, i) = list_lrc[nn]
        if (k, j, i) in list_kji_GHB:
            ii = list_kji_GHB.index((k, j, i))
            mylist.append(lam[nn] * list_ghb[ii][1])
        else:
            mylist.append(0.0)
    return mylist

def sens_CGHB_indirect(lam, hh):
    mylist = [lam_dAdCGHB_h(lam, hh)[nn] + lam_drhs_dCGHB(lam)[nn] for nn in range(len(lam_dAdCGHB_h(lam, hh)))]
    return mylist

def lam_drhs_dQWEL(lam, nsp):
    # This function returns the sensitivity with respect to each flow rate at a well
    # at a given location and a given stress period which is equal to the adjoint state
    mylist = len(lam) * [0.0]
    if nsp in dict_wel.keys():
        list_kji_WEL = []
        for i in range(dict_wel[nsp].size):
            list_kji_WEL.append(dict_wel[nsp][i][0])
        for (k, j, i) in list_kji_WEL:
            # nn = gwf.modelgrid.get_node((k, j, i))[0]
            nn = list_lrc.index((k, j, i))
            mylist[nn] = lam[nn]
        return mylist
    else:
        return mylist

def convert_idmarray_2_gridarray(idmarray):
    gridlist = []
    i = -1
    for m in range(gwf.modelgrid.nnodes):
        (km, jm, im) = gwf.modelgrid.get_lrc(m)[0]
        # (km, jm, im) = getlrc()[m]
        # (km, jm, im) = list_lrc[m]
        if gwf.modelgrid.idomain[km][jm][im] == 0:
            gridlist.append(-999.0)
        else:
            i += 1
            gridlist.append(idmarray[i])
    return np.array(gridlist)

# def getlrc():
#     # list_lrc = []
#     # for m in range(gwf.modelgrid.nnodes):
#     #     (km, jm, im) = gwf.modelgrid.get_lrc(m)[0]
#     #     if gwf.modelgrid.idomain[km][jm][im] == 0:
#     #         continue
#     #     else:
#     #         list_lrc.append((km, jm, im))
#     # return list_lrc
#     list_lrc = []
#     for m in range(gwf.modelgrid.nnodes):
#         (km, jm, im) = gwf.modelgrid.get_lrc(m)[0]
#         if gwf.modelgrid.idomain[km][jm][im] == 1:
#             list_lrc.append((km, jm, im))
#     return list_lrc
