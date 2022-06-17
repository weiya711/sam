import pytest
import random
import time
import os

from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.array import Array

from sam.sim.test.test import TIMEOUT, check_arr, check_seg_arr


@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("vectype", ["random", "runs", "blocks"])
def test_unit_vec_elemmul_u_u_u(samBench, vectype, debug_sim, dim1=2000, max_val=1000, fill=0):
    in_vec1 = [random.randint(0, max_val) for _ in range(dim1)]
    in_vec2 = [random.randint(0, max_val) for _ in range(dim1)]

    if debug_sim:
        print("VECTOR 1:", in_vec1)
        print("VECTOR 2:", in_vec2)

    assert (len(in_vec1) == len(in_vec2))

    gold_vec = [in_vec1[i] * in_vec2[i] for i in range(len(in_vec1))]

    rdscan = UncompressCrdRdScan(dim=dim1, debug=debug_sim)
    val1 = Array(init_arr=in_vec1, debug=debug_sim)
    val2 = Array(init_arr=in_vec2, debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    wrscan = ValsWrScan(size=dim1, fill=fill, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time_cnt = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref) > 0:
            rdscan.set_in_ref(in_ref.pop(0))
        rdscan.update()
        val1.set_load(rdscan.out_ref())
        val2.set_load(rdscan.out_ref())
        val1.update()
        val2.update()
        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())
        mul.update()
        wrscan.set_input(mul.out_val())
        wrscan.update()
        print("Timestep", time_cnt, "\t Done --", "\tRdScan:", rdscan.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())
        done = wrscan.out_done()
        time_cnt += 1

    check_arr(wrscan, gold_vec)

    def bench():
        time.sleep(0.0001)

    extra_info = dict()
    extra_info["cycles"] = time_cnt
    extra_info["vectype"] = vectype
    extra_info["format"] = "dense"

    samBench(bench, extra_info)
