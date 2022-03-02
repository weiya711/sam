import pytest
import copy
import random
from sim.src.repeater import RepeatSigGen
from sim.src.base import remove_emptystr

from sim.test.test import TIMEOUT, gen_stream, dedup_adj


arr_dict1 = {'istream':[0, 1, 'S', 2, 'S', 3, 'S', 'S', 'D'],
             'gold': ['R', 'R', 'S', 'R', 'S', 'R', 'S', 'D']}
arr_dict2 = {'istream':[0, 1, 2, 'S', 0, 1, 2, 'S', 'S', 3, 'S', 'S', 4, 5, 'S', 'S', 'S', 'D'],
             'gold': ['R', 'R', 'R', 'S', 'R', 'R', 'R', 'S', 'R', 'S', 'R', 'R', 'S', 'D']}
@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2])
def test_repeat_gen_direct(arrs, debug_sim):
    in_stream = copy.deepcopy(arrs['istream'])
    gold = copy.deepcopy(arrs['gold'])

    repsig = RepeatSigGen(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_stream) > 0:
            repsig.set_istream(in_stream.pop(0))
        repsig.update()
        print("Timestep", time, "\t Done:", repsig.out_done(), "\t Repeat Sig:", repsig.out_repeat())
        out.append(repsig.out_repeat())
        done = repsig.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_repeat_gen_1d(dim, debug_sim, max_val=100):
    in_stream = [random.randint(0, max_val) for _ in range(dim)] + ['S', 'D']
    gold = ['R'] * dim + ['S', 'D']

    repsig = RepeatSigGen(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_stream) > 0:
            repsig.set_istream(in_stream.pop(0))
        repsig.update()
        print("Timestep", time, "\t Done:", repsig.out_done(), "\t Repeat Sig:", repsig.out_repeat())
        out.append(repsig.out_repeat())
        done = repsig.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)

@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_repeat_gen_1d(dim, debug_sim, max_val=100):
    in_stream = [random.randint(0, max_val) for x in range(dim) for y in range(dim)] + ['S', 'D']
    gold = ['R'] * dim + ['S', 'D']

    repsig = RepeatSigGen(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_stream) > 0:
            repsig.set_istream(in_stream.pop(0))
        repsig.update()
        print("Timestep", time, "\t Done:", repsig.out_done(), "\t Repeat Sig:", repsig.out_repeat())
        out.append(repsig.out_repeat())
        done = repsig.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)


@pytest.mark.parametrize("max", [4, 16, 32, 64])
@pytest.mark.parametrize("nd", [1, 2, 3, 4, 5])
def test_repeat_gen_nd(max, nd, debug_sim):
    in_stream = gen_stream(n=nd, max_val=max, max_nnz=max)

    if debug_sim:
        print("Input Stream:", in_stream)

    gold = dedup_adj(in_stream)
    gold = ['R' if isinstance(x, int) else x for x in gold]

    repsig = RepeatSigGen(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(in_stream) > 0:
            repsig.set_istream(in_stream.pop(0))
        repsig.update()
        print("Timestep", time, "\t Done:", repsig.out_done(), "\t Repeat Sig:", repsig.out_repeat())
        out.append(repsig.out_repeat())
        done = repsig.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)
