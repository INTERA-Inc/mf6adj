import numpy as np

name = "GWF_2D_SS"
epsilon = 0.0e-05
tf = 1.0
h1 = 100
N = 11
L = 100.0
H = 1.0
k = 10.0
# sy = 0.01
# ss = 0.01
# sy = 0.001
# ss = 0.001
q = -300.0
T = k * H
L1 = L2 = L
D = L1 * L2
Nlay = 1
Nrow = Ncol = N
delrow = delcol = L / (N - 1)
bot = np.linspace(-H / Nlay, -H, Nlay) # botm