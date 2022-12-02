import pytest
from sam.sim.src.base import remove_emptystr
from sam.sim.src.compute import Add2, Multiply2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
from sam.sim.test.primitives.test_intersect import TIMEOUT
import numpy as np


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_max_1d(dim1, debug_sim):
    in1 = [x for x in range(dim1)] + ['S0', 'D']
    print(in1)
    in2 = dim1 / 2
    print(in2)
    # assert (len(in1) == len(in1))

    gold_val = np.maximum(np.arange(dim1), in2).tolist() + ['S0', 'D']

    max1 = Max(debug=debug_sim)

    done = False
    time = 0
    out_val = []
    max1.set_in2(in2)
    while not done and time < TIMEOUT:
        if len(in1) > 0:
            max1.set_in1(in1.pop(0))

        max1.update()

        out_val.append(max1.out_val())

        print("Timestep", time, "\t Out:", max1.out_val())

        done = max1.out_done()
        time += 1

    out_val = remove_emptystr(out_val)
    print("Ref:", gold_val)
    print("Out:", out_val)

    assert (out_val == gold_val)

@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_exp_1d(dim1, debug_sim):
    in1 = [x for x in range(dim1)] + ['S0', 'D']
    in2 = None
    # assert (len(in1) == len(in1))

    gold_val = np.exp(np.arange(dim1)).tolist() + ['S0', 'D']

    exp1 = Exp(debug=debug_sim)

    done = False
    time = 0
    out_val = []
    exp1.set_in2(in2)
    while not done and time < TIMEOUT:
        if len(in1) > 0:
            exp1.set_in1(in1.pop(0))

        exp1.update()

        out_val.append(exp1.out_val())

        print("Timestep", time, "\t Out:", exp1.out_val())

        done = exp1.out_done()
        time += 1

    out_val = remove_emptystr(out_val)
    print("Ref:", gold_val)
    print("Out:", out_val)

    assert (out_val == gold_val)


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_scalar_mult_1d(dim1, debug_sim):
    in1 = [x for x in range(dim1)] + ['S0', 'D']
    in2 = 4

    gold_val = np.arange(dim1) * in2 
    gold_val = gold_val.tolist() + ['S0', 'D']

    scal1 = ScalarMult(debug=debug_sim)

    done = False
    time = 0
    out_val = []
    scal1.set_in2(in2)
    while not done and time < TIMEOUT:
        if len(in1) > 0:
            scal1.set_in1(in1.pop(0))

        scal1.update()

        out_val.append(scal1.out_val())

        print("Timestep", time, "\t Out:", scal1.out_val())

        done = scal1.out_done()
        time += 1

    out_val = remove_emptystr(out_val)
    print("Ref:", gold_val)
    print("Out:", out_val)

    assert (out_val == gold_val)

