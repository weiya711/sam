import pytest
import random
from sim.src.rd_scanner import UncompressRdScan
from sim.src.wr_scanner import UncompressWrScan
from sim.src.joiner import Intersect2
from sim.src.compute import Multiply2
from sim.src.array import Array
from sim.test.primitives.test_intersect import TIMEOUT

@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_vec_elemmul_unc(dim1, max_val=1000, size=100, fill=0):
    debug = True

    in_vec1 = [random.randint(0, max_val) for _ in range(dim1)]
    in_vec2 = [random.randint(0, max_val) for _ in range(dim1)]

    if debug:
        print("VECTOR 1:", in_vec1)
        print("VECTOR 2:", in_vec2)

    assert(len(in_vec1) == len(in_vec2))

    gold_vec = [in_vec1[i] * in_vec2[i] for i in range(len(in_vec1))]
    in_vec1 += ['S', 'D']
    in_vec2 += ['S', 'D']

    rdscan = UncompressRdScan(dim=dim1, debug=debug)
    val1 = Array(init_arr=in_vec1, debug=debug)
    val2 = Array(init_arr=in_vec2, debug=debug)
    mul = Multiply2(debug=debug)
    wrscan = UncompressWrScan(size=size, fill=fill, debug=debug)

    in_ref = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
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
        wrscan.set_val(mul.out_val())
        wrscan.update()
        print("Timestep", time, "\t Done --", "\tRdScan:", rdscan.out_done(), "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())
        done = wrscan.out_done()
        time += 1

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (wrscan.get_arr() == gold_vec + [fill]*(size-len(gold_vec)))

    # Assert the array stores only the values
    wrscan.resize_arr(len(gold_vec))
    assert (wrscan.get_arr() == gold_vec)