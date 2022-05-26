import copy
import pytest

from sam.sim.src.bitvector import BVDrop
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT

arrs_dict1 = {'obv_in': [0b1100, 'S0', 'D'],
              'ibv_in': [0b1000, 'S0', 'S1', 'D'],
              'obv_gold': [0b0100, 'S0', 'D'],
              'ibv_gold': [0b1000, 'S1', 'D']}

arrs_dict2 = {'obv_in': [0b1111, 'S0', 'D'],
              'ibv_in': ['S0', 0b1000, 'S0', 'S0', 'S1', 'D'],
              'obv_gold': [0b0010, 'S0', 'D'],
              'ibv_gold': [0b1000, 'S1', 'D']}

arrs_dict3 = {'obv_in': [0b1111, 'S0', 0b0101, 'S1', 'D'],
              'ibv_in': ['S0', 0b1000, 'S0', 'S0', 'S1', 'S0', 'S2', 'D'],
              'obv_gold': [0b0010, 'S1', 'D'],
              'ibv_gold': [0b1000, 'S2', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3])
def test_bv_drop_nd(arrs, debug_sim):
    ibv = copy.deepcopy(arrs['ibv_in'])
    obv = copy.deepcopy(arrs['obv_in'])

    gold_obv = copy.deepcopy(arrs['obv_gold'])
    gold_ibv = copy.deepcopy(arrs['ibv_gold'])

    bd = BVDrop(debug=debug_sim)

    done = False
    time = 0
    out_obv = []
    out_ibv = []
    while not done and time < TIMEOUT:
        if len(ibv) > 0:
            item = ibv.pop(0)
            bd.set_inner_bv(item)
        if len(obv) > 0:
            bd.set_outer_bv(obv.pop(0))
        bd.update()

        out_obv.append(bd.out_bv_outer())
        out_ibv.append(bd.out_bv_inner())

        print("Timestep", time, "\t Done:", bd.out_done(), "\t Out:", bd.out_bv_outer())
        done = bd.out_done()
        time += 1

    out_obv = remove_emptystr(out_obv)
    out_ibv = remove_emptystr(out_ibv)

    assert (out_obv == gold_obv)
    assert (out_ibv == gold_ibv)
