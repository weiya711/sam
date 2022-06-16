import copy
import pytest

from functools import reduce

from sam.sim.src.bitvector import BVFixWidth
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT


def get_bv(crd):
    gold_bv = []
    temp = []
    for x in crd:
        if isinstance(x, int):
            temp.append(x)
        else:
            if temp:
                gold_bv.append(bin(reduce(lambda a, b: a | b, [0b1 << i for i in temp])))
                temp = []
            gold_bv.append(x)
    return gold_bv


arrs_dict1 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D']}
arrs_dict2 = {'crd_in': [0, 2, 3, 'S0', 'D']}
arrs_dict3 = {'crd_in': [0, 2, 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']}

arrs_dict4 = {'crd_in': [0, 2, 3, 'S0', 'D']}
arrs_dict5 = {'crd_in': [0, 'S0', 'D']}
arrs_dict6 = {'crd_in': [0, 2, 3, 'S1', 'D']}

arrs_dict7 = {'crd_in': [0, 2, 3, 'S0', 1, 3, 'S0', 0, 'S1', 'D']}
arrs_dict8 = {'crd_in': [0, 1, 'S0', 0, 1, 'S0', 0, 'S1', 'D']}
arrs_dict9 = {'crd_in': [0, 'S0', 2, 3, 'S1', 1, 'S0', 3, 'S1', 0, 'S2', 'D']}

arrs_dict10 = {'crd_in': [0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S1', 'D']}
arrs_dict11 = {'crd_in': [0, 1, 'S0', 0, 1, 'S0', 0, 1, 'S1', 'D']}
arrs_dict12 = {'crd_in': [0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S2', 'D']}

arrs_dict13 = {'crd_in': [0, 4, 8, 12, 16, 'S0', 'D']}
arrs_dict14 = {'crd_in': [0, 1, 2, 3, 4, 'S0', 'D']}
arrs_dict15 = {'crd_in': [0, 'S0', 4, 'S0', 8, 'S0', 12, 'S0', 16, 'S1', 'D']}

arrs_dict16 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D']}
arrs_dict17 = {'crd_in': [0, 1, 3, 4, 'S0', 'D']}
arrs_dict18 = {'crd_in': [0, 2, 'S0', 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']}


@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3, arrs_dict4, arrs_dict5, arrs_dict6,
                                  arrs_dict7, arrs_dict8, arrs_dict9, arrs_dict10, arrs_dict11, arrs_dict12,
                                  arrs_dict13, arrs_dict14, arrs_dict15, arrs_dict16, arrs_dict17, arrs_dict18])
def test_bvfw_direct(arrs, debug_sim):
    crd = copy.deepcopy(arrs['crd_in'])
    gold_bv = get_bv(crd)
    if debug_sim:
        print("Gold:", gold_bv)

    bv = BV(debug=debug_sim)

    done = False
    time = 0
    out_bv = []
    while not done and time < TIMEOUT:
        if len(crd) > 0:
            bv.set_in_crd(crd.pop(0))

        bv.update()

        out_bv.append(bv.out_bv())

        print("Timestep", time, "\t Done:", bv.out_done())

        done = bv.out_done()
        time += 1

    out_bv = remove_emptystr(out_bv)

    if debug_sim:
        print("BV: ", out_bv)

    assert (out_bv == gold_bv)


arrs_dict1 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D'],
              'ocrd_gold': [0, 2, 3, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']
              }

arrs_dict2 = {'crd_in': [0, 2, 3, 'S0', 'D'],
              'ocrd_gold': [0, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S1', 'D']
              }

arrs_dict3 = {'crd_in': [0, 2, 3, 'S0', 1, 3, 'S0', 0, 'S1', 'D'],
              'ocrd_gold': [0, 1, 'S0', 0, 1, 'S0', 0, 'S1', 'D'],
              'icrd_gold': [0, 'S0', 2, 3, 'S1', 1, 'S0', 3, 'S1', 0, 'S2', 'D']
              }

arrs_dict4 = {'crd_in': [0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S0', 0, 1, 2, 3, 'S1', 'D'],
              'ocrd_gold': [0, 1, 'S0', 0, 1, 'S0', 0, 1, 'S1', 'D'],
              'icrd_gold': [0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S1', 0, 1, 'S0', 2, 3, 'S2', 'D']
              }

arrs_dict5 = {'crd_in': [0, 4, 8, 12, 16, 'S0', 'D'],
              'ocrd_gold': [0, 1, 2, 3, 4, 'S0', 'D'],
              'icrd_gold': [0, 'S0', 4, 'S0', 8, 'S0', 12, 'S0', 16, 'S1', 'D']
              }

arrs_dict7 = {'crd_in': [0, 2, 3, 9, 11, 12, 'S0', 'D'],
              'ocrd_gold': [0, 1, 3, 4, 'S0', 'D'],
              'icrd_gold': [0, 2, 'S0', 3, 'S0', 9, 11, 'S0', 12, 'S1', 'D']
              }

