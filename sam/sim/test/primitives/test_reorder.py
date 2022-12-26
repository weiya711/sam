import copy
import pytest

from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.reorder import Reorder_and_split
from sam.sim.src.base import remove_emptystr


######################################
# Compressed Read Scanner Unit Tests #
######################################

arr_dict = {"seg": [0, 3, 6, 9, 11], "crd": [1, 3, 4, 0, 1, 2, 0, 4, 5, 3, 5], "in_ref": [0, 1, 2, 3, 'S0', 'D'], "in_crd": [0, 1, 4, 6, 'S0', 'D'],
            "out_crd_k": [1, 3, 'S0', 0, 1, 2, 'S0', 0, 'S0', 3, 'S1', 0, 'S0', 0, 1, 'S0', 1, 'S2', 'D'],
            "out_ref_k": [0, 1, 'S0', 3, 4, 5, 'S0', 6, 'S0', 9, 'S1', 2, 'S0', 7, 8, 'S0', 10, 'S2', 'D'],
            "out_crd_i": [0, 1, 4, 6, 'S0', 0, 4, 6, 'S1', 'D'],
            "out_ref_i": [0, 1, 2, 3, 'S0', 0, 2, 3, 'S1', 'D']}


arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S0', 'D'], "in_crd": [0, 2, 3, 'S0', 'D'],
             "out_crd_k": [0, 'S1', 2, 'S0', 2, 'S0',  2,  'S1', 'D'], "out_ref_k": [0, 'S0', 1, 'S0', 2, 'S0', 3, 'S1', 'D'],
             "out_crd_i": [0, 'S0',  0, 2, 3, 'S1', 'D'], "out_ref_i": [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D']}



arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D'],
             "out_crd": [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 'S1', 2, 3, 'S2', 'D'],
             "out_ref": [0, 1, 2, 'S0', 0, 1, 2, 'S1', 3, 'S1', 4, 5, 'S2', 'D']}
arr_dict3 = {"seg": [0, 4], "crd": [0, 1, 2, 3], "in_ref": [0, 'D'],
             "out_crd": [0, 1, 2, 3, 'S0', 'D'], "out_ref": [0, 1, 2, 3, 'S0', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict]) #, arr_dict2, arr_dict3])
def test_reorder_direct(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd_k"]
    gold_ref = arrs["out_ref_k"]
    gold_crd_i = arrs["out_crd_i"]
    gold_ref_i = arrs["out_ref_i"]

    assert (len(gold_crd) == len(gold_ref))

    crdscan = Reorder_and_split(seg_arr=seg_arr, crd_arr=crd_arr, limit=10, sf=4, debug=debug_sim)

    in_ref = copy.deepcopy(arrs["in_ref"])
    in_crd = copy.deepcopy(arrs["in_crd"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    out_crd_i = []
    out_ref_i = []
    out_crd_k_out = []
    out_ref_k_out = []
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            crdscan.set_input(in_ref.pop(0), in_crd.pop(0))
            
        crdscan.update()
        if crdscan.out_crd_k() != "":
            out_crd.append(crdscan.out_crd_k())
            out_ref.append(crdscan.out_ref_k())
        if crdscan.out_crd_i() != "":
            out_crd_i.append(crdscan.out_crd_i())
            out_ref_i.append(crdscan.out_ref_i())
        if crdscan.out_ref_k_outer() != "":
            out_crd_k_out.append(crdscan.out_crd_k_outer())
            out_ref_k_out.append(crdscan.out_ref_k_outer())


        print("Timestep", time, "\t Crd:", crdscan.out_crd_i(), "\t Ref:", crdscan.out_ref_i(), "\t Crd:", crdscan.out_crd_k(), "\t Ref:", crdscan.out_ref_k())
        print("______________________________________________________________________")
        done = crdscan.done
        time += 1
        if time > 10000:
            break
    print("Done and time: ", done, time)
    print(out_crd)
    print(gold_crd)
    print(out_ref)
    print(gold_ref)

    print(out_crd_i)
    print(gold_crd_i)
    print(out_ref_i)
    print(gold_ref_i)

    print(out_crd_k_out)
    print(out_ref_k_out)


    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)

