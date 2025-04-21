from GWF_2D_SS import *

# from GridData import *
# new_module = __import__(name)
# print('------------------', new_module)
# import importlib
# module = importlib.import_module(name, package=None)
# from module import *


def dFdh(k, j, i, ncells):
    mylist = np.zeros(ncells)
    m = gwf.modelgrid.get_node((k, j, i))[0]
    for ii in range(ncells):
        if ii == m:
            mylist[ii] = 1.0
    return mylist

def dFdh_avraged_head(ncells):
    mylist = np.zeros(ncells)
    for ii in range(ncells):
        (k, j, i) = gwf.modelgrid.get_lrc(ii)[0]
        if (k, j, i) in list_ch:
            mylist[ii] = 0.0
        else:
            mylist[ii] = CELLAREA[ii]
    return mylist

def J_averaged_head(ncells, X):
    sum = 0.0
    for ii in range(ncells):
        sum += X[ii] * CELLAREA[ii]
    return sum



def multiplyarray(A1, A2):
    n = len(A1)
    product = np.zeros(n)
    for i in range(n):
        product[i] = A1[i] * A2[i]
    return product

def substructarray(A1, A2):
    n = len(A1)
    mylist = [item1 - item2 for item1, item2 in zip(A1, A2)]
    return mylist


def SolveAdjointHeadAtPoint(k, j, i, ncells):
    rhs =  dFdh(k, j, i, ncells)
    A = csr_matrix((MAT[0], JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
    At = A.transpose()
    lam = spsolve(At, rhs)
    # lam = np.linalg.solve(At, rhs)
    return lam

def SolveAdjointAveragedHead(ncells):
    rhs =  dFdh_avraged_head(ncells)
    A = csr_matrix((MAT[0], JA_p, IA_p), shape=(len(IA) - 1, len(IA) - 1)).toarray()
    # At = A.transpose()
    lam = spsolve(A, rhs)
    # lam = spsolve(At, rhs)
    # lam = np.linalg.solve(At, rhs)
    return lam

def derivative_conductance_k1(k1, k2, w1, w2, d1, d2):
    return - 2.0 * w1 * d1 * d2 / ((w1 + w2 * k1 / k2) ** 2)

def d_amat_k():
    d_mat_k11 = np.zeros(len(MAT[0]))
    d_mat_k22 = np.zeros(len(MAT[0]))
    d_mat_k33 = np.zeros(len(MAT[0]))
    d_mat_k123 = np.zeros(len(MAT[0]))
    for nn in range(len(IA)-1):
        (k, j, i) = gwf.modelgrid.get_lrc(nn)[0]
        if (k, j, i) in list_ch:
            for ij in range(IAC[nn]-1):
                d_mat_k11[IA_p[nn] + ij] = 0.0
                d_mat_k22[IA_p[nn] + ij] = 0.0
                d_mat_k33[IA_p[nn] + ij] = 0.0
                d_mat_k123[IA_p[nn] + ij] = d_mat_k11[IA_p[nn] + ij] + d_mat_k22[IA_p[nn] + ij] + d_mat_k33[IA_p[nn] + ij]
        else:
            sum1 = 0.0
            sum2 = 0.0
            sum3 = 0.0
            ii = 1
            for nc in range(IAC[nn])[1:]:
                (knc, jnc, inc) = gwf.modelgrid.get_lrc(JA_p[IA_p[nn]+nc])[0]
                if knc == k - 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = derivative_conductance_k1(K33[nn], K33[JA_p[IA_p[nn]+nc]], CELLTOP[nn] - CELLBOT[nn], CELLTOP[JA_p[IA_p[nn]+nc]] - CELLBOT[JA_p[IA_p[nn]+nc]], DELR[i], DELC[j])
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif jnc == j - 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = derivative_conductance_k1(K22[nn], K22[JA_p[IA_p[nn]+nc]], DELC[j], DELC[j-1], DELR[i], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif inc == i - 1:
                    d_mat_k11[IA_p[nn] + ii] = derivative_conductance_k1(K11[nn], K11[JA_p[IA_p[nn]+nc]], DELR[i], DELR[i-1], DELC[j], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif inc == i + 1:
                    d_mat_k11[IA_p[nn] + ii] = derivative_conductance_k1(K11[nn], K11[JA_p[IA_p[nn]+nc]], DELR[i], DELR[i+1], DELC[j], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif jnc == j + 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = derivative_conductance_k1(K22[nn], K22[JA_p[IA_p[nn]+nc]], DELC[j], DELC[j+1], DELR[i], CELLTOP[nn] - CELLBOT[nn])
                    d_mat_k33[IA_p[nn] + ii] = 0.0
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
                    ii += 1
                elif knc == k + 1:
                    d_mat_k11[IA_p[nn] + ii] = 0.0
                    d_mat_k22[IA_p[nn] + ii] = 0.0
                    d_mat_k33[IA_p[nn] + ii] = derivative_conductance_k1(K33[nn], K33[JA_p[IA_p[nn]+nc]], CELLTOP[nn] - CELLBOT[nn], CELLTOP[JA_p[IA_p[nn]+nc]] - CELLBOT[JA_p[IA_p[nn]+nc]], DELR[i], DELC[j])
                    d_mat_k123[IA_p[nn] + ii] = d_mat_k11[IA_p[nn] + ii] + d_mat_k22[IA_p[nn] + ii] + d_mat_k33[IA_p[nn] + ii]
                    sum1 += d_mat_k11[IA_p[nn] + ii]
                    sum2 += d_mat_k22[IA_p[nn] + ii]
                    sum3 += d_mat_k33[IA_p[nn] + ii]
            d_mat_k11[IA_p[nn]] = - sum1
            d_mat_k22[IA_p[nn]] = - sum2
            d_mat_k33[IA_p[nn]] = - sum3
            d_mat_k123[IA_p[nn]] = d_mat_k11[IA_p[nn]] + d_mat_k22[IA_p[nn]] + d_mat_k33[IA_p[nn]]
    return d_mat_k11, d_mat_k22, d_mat_k33, d_mat_k123


def lam_dAdk_h(lam, dAdk, h):
    my_list = []
    for k in range(len(lam)):
        sum1 = 0.0
        sum2 = 0.0
        for j in range(IAC[k]):
            sum1 += dAdk[IA_p[k] + j] * h[JA_p[IA_p[k] + j]]
        sum1 = lam[k] * sum1
        for j in list(range(IAC[k]))[1:]:
            (kk, jj, ii) = gwf.modelgrid.get_lrc(JA_p[IA_p[k] + j])[0]
            if (kk, jj, ii) in list_ch:
                sum2 += 0.0
            else:
                sum2 += lam[JA_p[IA_p[k] + j]] * dAdk[IA_p[k] + j] * (h[k] - h[JA_p[IA_p[k] + j]])
        sum = sum1 + sum2
        my_list.append(sum)
    return my_list










