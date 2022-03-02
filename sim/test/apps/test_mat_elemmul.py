import pytest
import random

from sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
from sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sim.src.joiner import Intersect2
from sim.src.compute import Multiply2
from sim.src.array import Array

from sim.test.test import TIMEOUT, check_arr, check_seg_arr

@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
@pytest.mark.parametrize("dim2", [4, 16, 32, 64])
def test_mat_elemmul_uu_uu_uu(dim1, dim2, debug_sim, max_val=1000, fill=0):
    size = dim1*dim2+1
    in_mat1 = [random.randint(0, max_val) for _ in range(dim1*dim2)]
    in_mat2 = [random.randint(0, max_val) for _ in range(dim1*dim2)]

    if debug_sim:
        print("Mat 1:", in_mat1)
        print("Mat 2:", in_mat2)

    assert (len(in_mat1) == len(in_mat2))
    print(len(in_mat1))

    gold_vec = [in_mat1[i] * in_mat2[i] for i in range(len(in_mat1))]

    rdscan_d1 = UncompressRdScan(dim=dim1, debug=debug_sim)
    rdscan_d2 = UncompressRdScan(dim=dim2, debug=debug_sim)

    val1 = Array(init_arr=in_mat1, debug=debug_sim)
    val2 = Array(init_arr=in_mat2, debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    wrscan = ValsWrScan(size=size, fill=fill, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            rdscan_d1.set_in_ref(in_ref.pop(0))
        rdscan_d1.update()

        rdscan_d2.set_in_ref(rdscan_d1.out_ref())
        rdscan_d2.update()

        val1.set_load(rdscan_d2.out_ref())
        val2.set_load(rdscan_d2.out_ref())
        val1.update()
        val2.update()

        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())
        mul.update()

        wrscan.set_input(mul.out_val())
        wrscan.update()
        print("Timestep", time, "\t Done --",
              "\tRdScan D1:", rdscan_d1.out_done(),
              "\tRdScan D2:", rdscan_d2.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())
        done = wrscan.out_done()
        time += 1

    check_arr(wrscan, gold_vec)
