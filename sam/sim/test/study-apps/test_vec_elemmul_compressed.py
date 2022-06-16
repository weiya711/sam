import pytest
import random

from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array

from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr

# @pytest.mark.vec
@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_unit_vec_elemmul_c_c_c(nnz, vecname, debug_sim, max_val=1000, size=1001, fill=0):
    assert(size > max_val)

    crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr2 = sorted(set(crd_arr2))
    seg_arr2 = [0, len(crd_arr2)]
    vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)] * vals_arr2[crd_arr2.index(i)] for i in gold_crd]

    if debug_sim:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)

    crdscan1 = CompressedCrdRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim)
    crdscan2 = CompressedCrdRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim)
    inter = Intersect2(debug=debug_sim)
    val1 = Array(init_arr=vals_arr1, debug=debug_sim)
    val2 = Array(init_arr=vals_arr2, debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    oval_wrscan = ValsWrScan(size=size, fill=fill)
    ocrd_wrscan = CompressWrScan(size=size, seg_size=size, fill=fill)

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0))
        crdscan1.update()
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0))
        crdscan2.update()
        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd())
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd())
        inter.update()
        val1.set_load(inter.out_ref1())
        val2.set_load(inter.out_ref2())
        val1.update()
        val2.update()
        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())
        mul.update()
        oval_wrscan.set_input(mul.out_val())
        oval_wrscan.update()
        ocrd_wrscan.set_input(inter.out_crd())
        ocrd_wrscan.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\tInter:", inter.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\tOutVal:", oval_wrscan.out_done(),
              "\tOutCrd:", ocrd_wrscan.out_done())
        done = ocrd_wrscan.out_done()
        time += 1

    check_arr(oval_wrscan, gold_vals)
    check_arr(ocrd_wrscan, gold_crd)
    check_seg_arr(ocrd_wrscan, gold_seg)
