import pytest
import copy

import pytest

from sam.sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
from sam.sim.test.test import TIMEOUT


########################################
# Uncompressed Read Scanner Unit Tests #
########################################
@pytest.mark.parametrize("dim", [1, 2, 4, 16, 32, 100])
def test_rd_scan_u_direct_1d(dim, debug_sim):
    gold_crd = [x for x in range(dim)]
    gold_crd.append('S0')
    gold_crd.append('D')
    gold_ref = gold_crd
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
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


@pytest.mark.parametrize("dim", [1, 2, 4, 16, 32, 100])
def test_rd_scan_u_direct_2d(dim, debug_sim):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S0']) * 3 + cnt + ['S1', 'D']
    gold_ref = cnt + ['S0'] + [dim + x for x in cnt] + ['S0'] + [2 * dim + x for x in cnt] + ['S0'] + \
        [3 * dim + x for x in cnt] + ['S1', 'D']
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 1, 2, 3, 'S0', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
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


@pytest.mark.parametrize("dim", [1, 2, 4, 16, 32, 100])
def test_rd_scan_u_direct_3d(dim, debug_sim):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S0']) * 2 + cnt + ['S1'] + (cnt + ['S0']) + cnt + ['S1'] + (cnt + ['S0']) + cnt + ['S2', 'D']
    gold_ref = cnt + ['S0'] + [dim + x for x in cnt] + ['S0'] + [2 * dim + x for x in cnt] + ['S1'] + \
        [3 * dim + x for x in cnt] + ['S0'] + [4 * dim + x for x in cnt] + ['S1'] + \
        [5 * dim + x for x in cnt] + ['S0'] + \
        [6 * dim + x for x in cnt] + ['S2', 'D']
    assert (len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim=dim, debug=debug_sim)

    in_ref = [0, 1, 2, 'S0', 3, 4, 'S0', 5, 6, 'S1', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
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

def test_rd_scan_c_direct_1d(debug_sim):
    seg_arr = [0, 3]
    crd_arr = [0, 1, 3]

    gold_crd = crd_arr + ['S0', 'D']
    gold_ref = [x for x in range(len(crd_arr))] + ['S0', 'D']

    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
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


arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S0', 'D'],
             "out_crd": [0, 2, 'S0', 2, 'S0', 2, 'S1', 'D'], "out_ref": [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D']}
arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D'],
             "out_crd": [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 'S1', 2, 3, 'S2', 'D'],
             "out_ref": [0, 1, 2, 'S0', 0, 1, 2, 'S1', 3, 'S1', 4, 5, 'S2', 'D']}
arr_dict3 = {"seg": [0, 4], "crd": [0, 1, 2, 3], "in_ref": [0, 'D'],
             "out_crd": [0, 1, 2, 3, 'S0', 'D'], "out_ref": [0, 1, 2, 3, 'S0', 'D']}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2, arr_dict3])
def test_rd_scan_c_direct_nd(arrs, debug_sim):
    seg_arr = arrs["seg"]
    crd_arr = arrs["crd"]

    gold_crd = arrs["out_crd"]
    gold_ref = arrs["out_ref"]
    assert (len(gold_crd) == len(gold_ref))

    crdscan = CompressedRdScan(seg_arr=seg_arr, crd_arr=crd_arr, debug=debug_sim)

    in_ref = copy.deepcopy(arrs["in_ref"])
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done and time < TIMEOUT:
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
