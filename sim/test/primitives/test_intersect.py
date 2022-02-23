import numpy as np
import pytest
import random
from sim.src.base import remove_emptystr
from sim.src.joiner import Intersect2

TIMEOUT = 100

def test_intersect_direct_2d():
    gold_crd = [0, 'S', 0, 1, 2, 'S', 'S', 'D']
    gold_ref1 = [0, 'S', 1, 2, 3, 'S', 'S', 'D']
    gold_ref2 = [0, 'S', 0, 1, 2, 'S', 'S', 'D']
    assert(len(gold_crd) == len(gold_ref1) and len(gold_crd) == len(gold_ref2))

    in_crd1 = [0, 'S', 0, 1, 2, 'S', 'S', 'D']
    in_ref1 = [0, 'S', 1, 2, 3, 'S', 'S', 'D']
    in_crd2 = [0, 1, 2, 'S', 0, 1, 2, 'S', 'S', 'D']
    in_ref2 = [0, 1, 2, 'S', 0, 1, 2, 'S', 'S', 'D']
    assert(len(in_crd1) == len(in_ref1))
    assert(len(in_crd2) == len(in_ref2))

    inter = Intersect2()

    done = False
    time = 0
    out_crd = []
    out_ref1 = []
    out_ref2 = []
    while not done and time < TIMEOUT:
        if len(in_crd1) > 0:
            inter.set_in1(in_ref1.pop(0), in_crd1.pop(0))
        if len(in_crd2) > 0:
            inter.set_in2(in_ref2.pop(0), in_crd2.pop(0))
        inter.update()
        print("Timestep", time, "\t Crd:", inter.out_crd(), "\t Ref1:", inter.out_ref1(), "\t Ref2:", inter.out_ref2())
        out_crd.append(inter.out_crd())
        out_ref1.append(inter.out_ref1())
        out_ref2.append(inter.out_ref2())
        done = inter.done
        time += 1

    out_crd = remove_emptystr(out_crd)
    out_ref1 = remove_emptystr(out_ref1)
    out_ref2 = remove_emptystr(out_ref2)

    assert (out_crd == gold_crd)
    assert (out_ref1 == gold_ref1)
    assert (out_ref2 == gold_ref2)

@pytest.mark.parametrize("in1", [4, 16, 32, 64])
def test_intersect_1d(in1):

    in_crd1 = [x for x in range(in1)]+['S', 'D']
    in_ref1 = [x for x in range(in1)]+['S', 'D']
    in_crd2 = [0, 2, 4, 15, 17, 25, 31, 32, 50, 63, 'S', 'D']
    in_ref2 = [x for x in range(10)] + ['S', 'D']
    assert(len(in_crd1) == len(in_ref1))
    assert(len(in_crd2) == len(in_ref2))

    gold_crd = [x for x in in_crd2[:-2] if x < in1] + ['S', 'D']
    gold_ref1 = gold_crd
    gold_ref2 = [x for x in range(len(gold_crd[:-2]))] + ['S', 'D']
    assert(len(gold_crd) == len(gold_ref1) and len(gold_crd) == len(gold_ref2))

    inter = Intersect2()

    done = False
    time = 0
    out_crd = []
    out_ref1 = []
    out_ref2 = []
    while not done and time < TIMEOUT:
        if len(in_crd1) > 0:
            inter.set_in1(in_ref1.pop(0), in_crd1.pop(0))
        if len(in_crd2) > 0:
            inter.set_in2(in_ref2.pop(0), in_crd2.pop(0))
        inter.update()
        print("Timestep", time, "\t Crd:", inter.out_crd(), "\t Ref1:", inter.out_ref1(), "\t Ref2:", inter.out_ref2())
        out_crd.append(inter.out_crd())
        out_ref1.append(inter.out_ref1())
        out_ref2.append(inter.out_ref2())
        done = inter.done
        time += 1

    out_crd = remove_emptystr(out_crd)
    out_ref1 = remove_emptystr(out_ref1)
    out_ref2 = remove_emptystr(out_ref2)

    assert (out_crd == gold_crd)
    assert (out_ref1 == gold_ref1)
    assert (out_ref2 == gold_ref2)
