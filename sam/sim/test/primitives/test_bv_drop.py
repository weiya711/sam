import copy
import pytest

from sam.sim.src.bitvector import BVDrop
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'obv_in': [0b1100, 'S0', 'D'],
              'ibv_in': [0b1000, 'S0', 'S1', 'D'],
              'obv_gold': [0b0100, 'S0', 'D'],
              'ibv_gold': [0b1000, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1])
def test_bv_drop_1d(arrs, debug_sim):
    ibv = copy.deepcopy(arrs['ibv_in'])
    obv = copy.deepcopy(arrs['obv_in'])

    gold_obv = copy.deepcopy(arrs['obv_gold'])

    bd = BVDrop(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(ibv) > 0:
            bd.set_inner_bv(ibv.pop(0))
        if len(obv) > 0:
            bd.set_outer_bv(obv.pop(0))
        bd.update()
        print("Timestep", time, "\t Done:", bd.out_done(), "\t Out:", bd.out_bv_outer())
        out.append(bd.out_bv_outer())
        done = bd.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold_obv)