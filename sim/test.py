import numpy as np 
import pytest
import random 
from sim import vec_scalar_mul_u

@pytest.mark.parametrize("dim", [16])
@pytest.mark.parametrize("val_type", ["constant", "increment", "random"])
@pytest.mark.parametrize("s2", [2])
def test_ndarr_vec_scalar_mul_unc(dim, val_type, s2):
    v1 = []
    vout = []
    
    if val_type == "constant":
        v1 = [3]*dim
    elif val_type == "increment":
        v1 = [*range(0, dim)]
    else:
        v1 = [random.random() * N for x in range(dim)]

    v1_ndarr = np.ndarray(v1)
    s2_ndarr = np.ndarray([s2])
    vec_scalar_mul_u(v1_ndarr, s2_ndarr, vout)
    assert vout == s2*v1_ndarr

@pytest.mark.parametrize("dim", [16])
@pytest.mark.parametrize("val_type", ["constant", "increment", "random"])
def test_ndarr_vec_elem_mul(dim, val_type): 
    v1 = np.arange(0, dim, 1)
    v2 = np.arange(0, dim, 1)
    v_out = vec_elem_mull(v1, v2)
    assert v_out == v1 * v2
    
