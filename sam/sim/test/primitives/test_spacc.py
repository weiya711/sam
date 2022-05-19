import copy
import pytest

from sam.sim.src.accumulator import SparseAccumulator1
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'ocrd_in': [0, 0, 0, 2, 2, 2, 2, 2, 2, 'D'],
              'icrd_in': [0, 2, 3, 0, 2, 3, 0, 2, 3, 'D'],
              'val_in': [50, 5, 10, 40, 4, 8, -40, 33, 36, 'D'],
              'ocrd_gold': [0, 2, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
              'val_gold': [50, 5, 10, 'S0',  0, 37, 44, 'S1', 'D']}

@pytest.mark.parametrize("arrs", [arrs_dict1])
def test_spacc_direct(arrs, debug_sim):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])
    val = copy.deepcopy(arrs['val_in'])

    gold_ocrd = copy.deepcopy(arrs['ocrd_gold'])
    gold_icrd = copy.deepcopy(arrs['icrd_gold'])
    gold_val = copy.deepcopy(arrs['val_gold'])

    sa = SparseAccumulator1(val_stkn=True, debug=debug_sim)

    done = False
    time = 0
    out_ocrd = []
    out_icrd = []
    out_val = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            sa.set_inner_crdpt(icrd.pop(0))
        if len(ocrd) > 0:
            sa.set_outer_crdpt(ocrd.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))
        sa.update()
        print("Timestep", time, "\t Done:", sa.out_done())

        out_ocrd.append(sa.out_outer_crd())
        out_icrd.append(sa.out_inner_crd())
        out_val.append(sa.out_val())
        done = sa.out_done()
        time += 1

    out_ocrd = remove_emptystr(out_ocrd)
    out_icrd = remove_emptystr(out_icrd)
    out_val = remove_emptystr(out_val)

    if debug_sim:
        print("Outer Crd: ", out_ocrd)
        print("Inner Crd: ", out_icrd)
        print("Vals: ", out_val)

    assert (out_ocrd == gold_ocrd)
    assert (out_icrd == gold_icrd)
    assert (out_val == gold_val)
