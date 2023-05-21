import copy
import pytest

from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.reorder import ReorderAndSplit, RepeatedTokenDropper
from sam.sim.src.base import remove_emptystr


######################################
# Compressed Read Scanner Unit Tests #
######################################

arr_dict = {"split_factor": 4, "seg": [0, 3, 6, 9, 11], "crd": [1, 3, 4, 0, 1, 2, 0, 4, 5, 3, 5],
            "in_ref": [0, 1, 2, 3, 'S0', 'D'], "in_crd": [0, 1, 4, 6, 'S0', 'D'],
            "out_crd_k": [1, 3, 'S0', 0, 1, 2, 'S0', 0, 'S0', 3, 'S1', 0, 'S0', 0, 1, 'S0', 1, 'S2', 'D'],
            "out_ref_k": [0, 1, 'S0', 3, 4, 5, 'S0', 6, 'S0', 9, 'S1', 2, 'S0', 7, 8, 'S0', 10, 'S2', 'D'],
            "out_crd_i": [0, 1, 4, 6, 'S0', 0, 4, 6, 'S1', 'D'],
            "out_ref_i": [0, 1, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
            "out_crd_k_outer": [0, 1, 'S0', 'D'],
            "out_ref_k_outer": [0, 1, 'S0', 'D']
            }


arr_dict2 = {"split_factor": 2, "seg": [0, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             "crd": [1, 2, 0, 1, 0, 2, 1, 1, 0, 2, 0, 2, 4, 2, 0],
             "in_ref": [0, 1, 2, 3, 4, 5, 6, 'S0', 7, 8, 9, 10, 11, 'S1', 'D'],
             "in_crd": [0, 1, 4, 5, 12, 15, 20, 'S0', 0, 1, 4, 12, 20, 'S1', 'D'],
             "out_crd_k": [1, 'S0', 0, 1, 'S0', 0, 'S0', 1, 'S0', 1, 'S0', 0, 'S1', 0, 'S0',
                           0, 'S0', 0, 'S2', 0, 'S0', 0, 'S1', 0, 'S0', 0, 'S1', 0, 'S3', 'D'],
             "out_ref_k": [0, 'S0', 2, 3, 'S0', 4, 'S0', 6, 'S0', 7, 'S0', 8, 'S1', 1, 'S0', 5, 'S0',
                           9, 'S2', 10, 'S0', 14, 'S1', 11, 'S0', 13, 'S1', 12, 'S3', 'D'],
             # "out_ref_k": [0, 'S0', 1, 'S0', 3, 4, 5, 'S0', 6, 'S0', 9, 'S1', 2, 'S0', 7, 8, 'S0', 10, 'S2', 'D'],
             "out_crd_i": [0, 1, 4, 5, 12, 15, 'S0', 0, 4, 20, 'S1', 0, 20, 'S0', 1, 12, 'S0', 4, 'S2', 'D'],
             "out_ref_i": [0, 1, 2, 3, 4, 5, 'S0', 0, 2, 6, 'S1', 7, 11, 'S0', 8, 10, 'S0', 9, 'S2', 'D'],
             "out_crd_k_outer": [0, 1, 'S0', 0, 1, 2, 'S1', 'D'],
             "out_ref_k_outer": [0, 1, 'S0', 2, 3, 4, 'S1', 'D']
             }

arr_dict3 = {"split_factor": 1, "seg": [0, 1], "crd": [12], "in_ref": [0, 'S0', 'D'], "in_crd": [19, 'S0', 'D'],
             "out_crd_k": [0, 'S2', 'D'],
             "out_ref_k": [0, 'S2', 'D'],
             # "out_ref_k": [0, 'S0', 1, 'S0', 3, 4, 5, 'S0', 6, 'S0', 9, 'S1', 2, 'S0', 7, 8, 'S0', 10, 'S2', 'D'],
             "out_crd_i": [19, 'S1', 'D'],
             "out_ref_i": [0, 'S1', 'D'],
             "out_crd_k_outer": [12, 'S0', 'D'],
             "out_ref_k_outer": [0, 'S0', 'D']
             }

arr_dict4 = {"split_factor": 3, "seg": [0, 1, 2, 3, 4, 5, 6], "crd": [0, 1, 2, 3, 4, 5],
             "in_ref": [0, 1, 2, 'S0', 3, 4, 5, 'S1', 'D'], "in_crd": [0, 1, 2, 'S0', 3, 4, 5, 'S1', 'D'],
             "out_crd_k": [0, 'S0', 1, 'S0', 2, 'S2', 0, 'S0', 1, 'S0', 2, 'S3', 'D'],
             "out_ref_k": [0, 'S0', 1, 'S0', 2, 'S2', 3, 'S0', 4, 'S0', 5, 'S3', 'D'],
             # "out_ref_k": [0, 'S0', 1, 'S0', 3, 4, 5, 'S0', 6, 'S0', 9, 'S1', 2, 'S0', 7, 8, 'S0', 10, 'S2', 'D'],
             "out_crd_i": [0, 1, 2, 'S1', 3, 4, 5, 'S2', 'D'],
             "out_ref_i": [0, 1, 2, 'S1', 3, 4, 5, 'S2', 'D'],
             "out_crd_k_outer": [0, 'S0', 1, 'S1', 'D'],
             "out_ref_k_outer": [0, 'S0', 1, 'S1', 'D']
             }


@pytest.mark.parametrize("arrs", [arr_dict2, arr_dict4, arr_dict2, arr_dict3])
def test_reorder_direct(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]
    gold_crd = arrs["out_crd_k"]
    gold_ref = arrs["out_ref_k"]
    gold_crd_i = arrs["out_crd_i"]
    gold_ref_i = arrs["out_ref_i"]
    gold_crd_k_out = arrs["out_crd_k_outer"]
    gold_ref_k_out = arrs["out_ref_k_outer"]
    sf = arrs["split_factor"]

    assert (len(gold_crd) == len(gold_ref))
    crdscan = ReorderAndSplit(seg_arr=seg_arr, crd_arr=crd_arr, limit=10, sf=sf, debug=debug_sim)
    crd_k = RepeatedTokenDropper(name="crdk")
    ref_k = RepeatedTokenDropper(name="refk")
    crd_i = RepeatedTokenDropper(name="crdi")
    ref_i = RepeatedTokenDropper(name="refi")
    crd_k_out = RepeatedTokenDropper(name="crdkout")
    ref_k_out = RepeatedTokenDropper(name="refkout")

    temp_arr = []
    t_arr_k = []
    t_arr_i = []
    t_arr_k_out = []
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
        crd_k.add_token(crdscan.out_crd_k())
        ref_k.add_token(crdscan.out_ref_k())
        crd_i.add_token(crdscan.out_crd_i())
        ref_i.add_token(crdscan.out_ref_i())
        crd_k_out.add_token(crdscan.out_crd_k_outer())
        ref_k_out.add_token(crdscan.out_ref_k_outer())
        crdscan.update()
        crd_k.update()
        ref_k.update()
        crd_i.update()
        ref_i.update()
        crd_k_out.update()
        ref_k_out.update()
        if crd_k.get_token() != "":
            out_crd.append(crd_k.get_token())
            out_ref.append(ref_k.get_token())
        if crd_i.get_token() != "":
            out_crd_i.append(crd_i.get_token())
            out_ref_i.append(ref_i.get_token())
        if crd_k_out.get_token() != "":
            out_crd_k_out.append(crd_k_out.get_token())
        if ref_k_out.get_token() != "":
            out_ref_k_out.append(ref_k_out.get_token())
        t_arr_k.append(crdscan.out_crd_k())
        t_arr_i.append(crdscan.out_crd_i())
        t_arr_k_out.append(crdscan.out_crd_k())
        temp_arr.append(crdscan.out_ref_k_outer())
        print("Timestep", time, "\t k_out_crd:", crdscan.out_crd_k_outer(),
              "\t k_out_ref:", crdscan.out_ref_k_outer(), "\t Crd:",
              crdscan.out_crd_i(), "\t Ref:", crdscan.out_ref_i(), "\t Crd:",
              crdscan.out_crd_k(), "\t Ref:", crdscan.out_ref_k())
        done = crd_k.done and ref_k.done and crd_i.done and ref_i.done and crd_k_out.done and ref_k_out.done
        time += 1
        if time > 10000:
            break
    print("Done and time: ", done, time)
    print("Out Crd val (k): ", out_crd)
    print("Gold Crd val (k): ", gold_crd)
    print(out_ref)
    print(gold_ref)
    print("Out Crd Val (i) ", out_crd_i)
    print("Gold Crd Val (i)", gold_crd_i)
    print("outer crd: ", out_crd_k_out)
    print(out_ref_k_out)
    print("crd1 ", out_crd)
    print("crd2 ", gold_crd)
    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)
    assert (out_ref_i == gold_ref_i)
    assert (out_crd_i == gold_crd_i)
    assert (out_crd_k_out == gold_crd_k_out)
    assert (out_ref_k_out == gold_ref_k_out)
