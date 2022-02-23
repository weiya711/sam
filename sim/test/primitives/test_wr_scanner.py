import pytest
import random
from sim.src.wr_scanner import UncompressWrScan
from sim.test.primitives.test_intersect import TIMEOUT

@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_arr_store_1d(dim1, max_val=1000, size=100, fill=0):
    in_val = [random.randint(0, max_val) for x in range(dim1)] + ['S', 'D']

    gold_val = in_val[:-2]

    wrscan = UncompressWrScan(size=size, fill=fill)

    done = False
    time = 0
    out_val = []
    count = 0
    while not done and time < TIMEOUT:
        if count < len(in_val):
            wrscan.set_val(in_val[count])
            count += 1
        wrscan.update()
        print("Timestep", time)
        done = wrscan.out_done()
        time += 1

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (wrscan.get_arr() == gold_val + [fill]*(size-len(gold_val)))

    # Assert the array stores only the values
    wrscan.resize_arr(len(gold_val))
    assert (wrscan.get_arr() == gold_val)
