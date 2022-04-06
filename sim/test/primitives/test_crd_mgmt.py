import copy
import pytest

from sim.src.crd_manager import CrdDrop
from sim.src.base import remove_emptystr
from sim.test.test import TIMEOUT

arrs_dict1 = {'ocrd_in': [0, 1, 'S0', 'D'],
              'icrd_in': [1, 'S0', 'S1', 'D'],
              'gold': [0, 'S0', 'D']}

arrs_dict2 = {'ocrd_in': [0, 1, 'S0', 'D'],
              'icrd_in': [1, 'S0', 1, 'S1', 'D'],
              'gold': [0, 1, 'S0', 'D']}

arrs_dict3 = {'ocrd_in': [0, 1, 2, 3, 'S0', 'D'],
              'icrd_in': [1, 'S0', 1, 'S0', 'S0', 1, 'S1', 'D'],
              'gold': [0, 1, 3, 'S0', 'D']}
@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3])
def test_crd_drop_1d(arrs, debug_sim, max_val=1000):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])

    gold = copy.deepcopy(arrs['gold'])

    cd = CrdDrop(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            cd.set_inner_crd(icrd.pop(0))
        if len(ocrd) > 0:
            cd.set_outer_crd(ocrd.pop(0))
        cd.update()
        print("Timestep", time, "\t Done:", cd.out_done(), "\t Out:", cd.out_crd_outer())
        out.append(cd.out_crd_outer())
        done = cd.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)

arrs_dict0 = {"ocrd_in": ['', 0, 1, '', '', 'S0', 'D', '', '', '', ''],
              "icrd_in": ['', '', '', 1, '', '', 'S0', '', '', 'S1', 'D'],
              "gold": [0, 'S0', 'D']}
@pytest.mark.parametrize("arrs", [arrs_dict0])
def test_crd_drop_emptystr_1d(arrs, debug_sim, max_val=1000):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])

    gold = copy.deepcopy(arrs['gold'])

    cd = CrdDrop(debug=debug_sim)

    done = False
    time = 0
    out = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            cd.set_inner_crd(icrd.pop(0))
        if len(ocrd) > 0:
            cd.set_outer_crd(ocrd.pop(0))
        cd.update()
        print("Timestep", time, "\t Done:", cd.out_done(), "\t Out:", cd.out_crd_outer())
        out.append(cd.out_crd_outer())
        done = cd.out_done()
        time += 1

    out = remove_emptystr(out)
    assert (out == gold)