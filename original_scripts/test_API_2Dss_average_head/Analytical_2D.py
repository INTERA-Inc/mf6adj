import numpy as np
from GridData import *


def a(C, n):
    return (C * L1 ** 2 / (((n * np.pi) ** 3) * T)) * (((-1) ** n) * (2 - (n * np.pi) ** 2) - 2)

def b(C, n):
    return - (C * L1 ** 2 / (n * np.pi * T)) * (((-1) ** n) - 1)

def alpha(m):
    return m * np.pi / L1

def beta(n):
    return n * np.pi / L2

def omega_square(m, n):
    return (alpha(m)) ** 2 + (beta(n)) ** 2

def phi_s(x, y, xs, ys, M, N):
    listofmindices = [item + 1 for item in range(M)]
    Sum = 0.0
    a = [0.5]
    for i in [item + 1 for item in range(N)]:
        a.append(1.0)
    for n in range(N+1):
        for m in listofmindices:
            Sum = Sum + a[n] * np.sin(alpha(m) * x) * np.cos(beta(n) * y) * np.sin(alpha(m) * xs) * np.cos(beta(n) * ys) / omega_square(m, n)
    Sum = - (4 / (T*D)) * Sum
    return Sum

# def phi_s_Averaged_Head(x, y, C, N):
#     listofmindices = [item + 1 for item in range(N)]
#     Sum = 0.0
#     for n in listofmindices:
#             Sum = Sum + a(C, n) * (np.sinh(alpha(n) * y) - np.sinh(alpha(n) * (y - L2))) * np.sin(alpha(n) * x) / np.sinh(alpha(n) * L2) \
#                       + b(C, n) * np.sinh(beta(n) * x) * np.sin(beta(n) * y) / np.sinh(beta(n) * L1) - (C / (2 * T)) * (x ** 2)
#     return Sum

def phi_s_Averaged_Head(x, y, C, M, N):
    listofmindices = [item + 1 for item in range(M)]
    listofnindices = [item + 1 for item in range(N)]
    Sum = 0.0
    for m in listofmindices:
        for n in listofnindices:
            Sum = Sum + ((-1)**m - 1) * ((-1)**n - 1) * np.sin(alpha(m) * x) * np.sin(beta(n) * y) / (alpha(m) * beta(n) * omega_square(m, n))
    Sum = - (4 * C / (L1 * L2 * T)) * Sum
    return Sum

