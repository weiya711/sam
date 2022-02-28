import numpy as np
import pytest
import random
from sim.src.rd_scanner import UncompressRdScan, CompressedRdScan


########################################
# Uncompressed Read Scanner Unit Tests #
########################################
@pytest.mark.parametrize("dim", [1, 2, 4, 16])
def test_rd_scan_uncompress_1d(dim, debug_sim):
    gold_crd = [x for x in range(dim)]
    gold_crd.append('S')
    gold_crd.append('D')
    gold_ref = gold_crd
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))
        urs.update()
        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())
        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())
        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


@pytest.mark.parametrize("dim", [1, 2, 4])
def test_rd_scan_uncompress_rd_scan_2d(dim, debug_sim):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S']) * 4 + ['S', 'D']
    gold_ref = cnt + ['S'] + [dim + x for x in cnt] + ['S'] + [2 * dim + x for x in cnt] + ['S'] + \
               [3 * dim + x for x in cnt] + ['S'] + ['S', 'D']
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 1, 2, 3, 'S', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))
        urs.update()
        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())
        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())
        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


def test_rd_scan_uncompress_3d(debug_sim, dim=4):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S']) * 3 + ['S'] + (cnt + ['S']) * 2 + ['S'] + (cnt + ['S']) * 2 + ['S'] + ['S', 'D']
    gold_ref = cnt + ['S'] + [dim + x for x in cnt] + ['S'] + [2 * dim + x for x in cnt] + ['S'] + ['S'] + \
               [3 * dim + x for x in cnt] + ['S'] + [4 * dim + x for x in cnt] + ['S', 'S'] + \
               [5 * dim + x for x in cnt] + ['S'] + \
               [6 * dim + x for x in cnt] + ['S'] + ['S'] + ['S', 'D']
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 1, 2, 'S', 3, 4, 'S', 5, 6, 'S', 'S', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))
        urs.update()
        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())
        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())
        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


######################################
# Compressed Read Scanner Unit Tests #
######################################

def test_rd_scan_comp_direct_1d(debug_sim):
    seg_arr = [0, 3]
    crd_arr = [0, 1, 3]

    gold_crd = crd_arr + ['S', 'D']
    gold_ref = [x for x in range(len(crd_arr))] + ['S', 'D']

    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))
        crdscan.update()
        print("Timestep", time, "\t Crd:", crdscan.out_crd(), "\t Ref:", crdscan.out_ref())
        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())
        done = crdscan.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S', 'D'],
             "out_crd": [0, 2, 'S', 2, 'S', 2, 'S', 'S', 'D'], "out_ref": [0, 1, 'S', 2, 'S', 3, 'S', 'S', 'D']}
arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 0, 'S', 1, 'S', 2, 'S', 'S', 'D'],
             "out_crd": [0, 2, 3, 'S', 0, 2, 3, 'S', 'S', 0, 'S', 'S', 2, 3, 'S', 'S', 'S', 'D'],
             "out_ref": [0, 1, 2, 'S', 0, 1, 2, 'S', 'S', 3, 'S', 'S', 4, 5, 'S', 'S', 'S', 'D']}

@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2])
def test_rd_scan_comp_direct(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = arrs["in_ref"]
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            crdscan.set_in_ref(in_ref.pop(0))
        crdscan.update()
        print("Timestep", time, "\t Crd:", crdscan.out_crd(), "\t Ref:", crdscan.out_ref())
        out_crd.append(crdscan.out_crd())
        out_ref.append(crdscan.out_ref())
        done = crdscan.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)
