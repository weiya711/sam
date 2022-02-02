import numpy as np

from .primitive import *

# x(i) = b(i)*c()
def vec_scalar_mul_u(v1_vals_arr, s2_vals_arr, out_vals_arr):
    # metadata
    v1_dim0 = len(v1_vals_arr) 
    s1_dim = 1

    v1_crds = rdScan(0, stop=v1_dim0, crd_only=True)
    s2_crds = rdScan(0, stop=s1_dim, repeat=v1_dim0, crd_only=True)
    v1_vals = valArr(v1_crds, v1_vals_arr)
    s2_vals = valArr(s2_crds, s2_vals_arr)
    vout_vals = mul(v1_vals, s2_vals)
    wrScan(vout_vals, out_vals_arr)
    
    

def vec_elem_mul(v1, v2):
    pass

def mat_vec_mul(m1, v2):
    pass

def mat_mul(m1, m2):
    pass

def tensor3_elem_mul(t3_1, t3_2):
    pass




