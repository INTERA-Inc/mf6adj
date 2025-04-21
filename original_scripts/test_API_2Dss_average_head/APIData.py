import os

import modflowapi
import numpy as np

dll = os.path.join('D:/','github','MFexe','libmf6.dll')
mf6api = modflowapi.ModflowApi(dll)
mf6api.initialize()
current_time = mf6api.get_current_time()
end_time = mf6api.get_end_time()
max_iter = mf6api.get_value(mf6api.get_var_address("MXITER", "SLN_1"))
JA = mf6api.get_value_ptr(mf6api.get_var_address("JA", "TEST_API_1D/CON"))
IA = mf6api.get_value_ptr(mf6api.get_var_address("IA", "TEST_API_1D/CON"))
JA_p = np.array([number - 1 for number in JA])
IA_p = np.array([number - 1 for number in IA])
inewton = mf6api.get_value_ptr(mf6api.get_var_address("INEWTON", "TEST_API_1D/NPF"))
idomain = mf6api.get_value_ptr(mf6api.get_var_address("IDOMAIN", "TEST_API_1D/DIS"))
# print('IA = ', IA)
# print('JA = ', JA)
# print('IA_p = ', IA_p)
# print('JA_p = ', JA_p)