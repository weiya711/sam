import pytest
import random
from sim.src.array import Array
from sim.test.test import TIMEOUT


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_arr_load_1d(dim1, debug_sim, max_val=1000):
    in_val = [x for x in range(dim1)] + ['S0', 'D']

    gold_val = [random.randint(0, max_val) for _ in range(dim1)]

    arr = Array(init_arr=gold_val, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    count = 0
    while not done and time < TIMEOUT:
        if count < len(in_val):
            arr.set_load(in_val[count])
            count += 1
        arr.update()
        print("Timestep", time, "\t Out:", arr.out_load())
        out_val.append(arr.out_load())
        done = arr.out_done()
        time += 1

    # Assert the array stores only the values
    assert (arr.get_arr() == gold_val)

    # Assert that the loaded streams also pass through tokens
    assert (out_val == gold_val + ['S0', 'D'])


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_arr_store_1d(dim1, debug_sim, max_val=1000):
    in_addr = [x for x in range(dim1)] + ['S0', 'D']
    in_val = [random.randint(0, max_val) for x in range(dim1)] + ['S0', 'D']
    assert (len(in_val) == len(in_addr))

    gold_val = in_val[:-2]

    arr = Array(debug=debug_sim)
    arr.resize(len(gold_val))

    done = False
    time = 0
    out_val = []
    count = 0
    while not done and time < TIMEOUT:
        if count < len(in_val):
            arr.set_store(in_addr[count], in_val[count])
            count += 1
        arr.update()
        print("Timestep", time, "\t Out:", arr.out_load())
        out_val.append(arr.out_load())
        done = arr.out_done()
        time += 1

    # Assert the array stores only the values
    assert (arr.get_arr() == gold_val)
