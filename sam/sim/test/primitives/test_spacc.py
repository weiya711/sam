import copy
import pytest
import random
import numpy as np

from sam.sim.src.accumulator import SparseAccumulator1, SpAcc1New
from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.src.array import Array
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT, gen_n_comp_arrs, gen_val_arr, get_point_list, \
    convert_point_tuple, convert_ndarr_point_tuple, convert_point_tuple_ndarr, check_point_tuple

# Old sparse accumulator
arrs_dict1 = {'ocrd_in': [0, 2, 'S0', 2, 'S1', 'D'],
              'icrd_in': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 2, 3, 'S2', 'D'],
              'val_in': [50, 5, 10, 'S0', 40, 4, 8, 'S1', -40, 33, 36, 'S2', 'D'],
              'ocrd_gold': [0, 2, 'S0', 'D'],
              'icrd_gold': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
              'val_gold': [50, 5, 10, 'S0', 0, 37, 44, 'S1', 'D']}

arrs_dict2 = {
    'ocrd_in': [0, 'S0', 1, 'S0', 2, 'S0', 3, 'S0', 4, 'S0', 5, 'S0', 6, 'S0', 7, 'S0', 8, 'S0', 9, 'S0', 10, 'S0',
                11, 'S0', 12, 'S0', 13, 'S0', 14, 'S0', 15, 'S0', 16, 'S0', 17, 'S0', 18, 'S0', 19, 'S0', 20, 'S0',
                21, 'S0', 22, 'S0', 23, 'S0', 24, 'S0', 25, 'S0', 26, 'S0', 27, 'S0', 28, 'S0', 29, 'S0', 30, 'S0',
                31, 'S0', 32, 'S0', 33, 'S0', 34, 'S0', 35, 'S0', 36, 'S0', 37, 'S0', 38, 'S0', 39, 'S0', 40, 'S0',
                41, 'S0', 42, 'S0', 43, 'S0', 44, 'S0', 45, 'S0', 46, 'S0', 47, 'S0', 48, 'S0', 49, 'S0', 50, 'S0',
                51, 'S0', 52, 'S0', 53, 'S0', 54, 'S0', 55, 'S0', 56, 'S0', 57, 'S0', 58, 'S0', 59, 'S0', 60, 'S0',
                61, 'S0', 62, 'S0', 63, 'S0', 64, 'S0', 65, 'S1', 'D'],
    'icrd_in': [65, 'S1', 0, 'S1', 1, 'S1', 2, 'S1', 3, 'S1', 4, 'S1', 5, 'S1', 6, 'S1', 7, 'S1', 8, 'S1', 9, 'S1',
                10, 'S1', 11, 'S1', 12, 'S1', 13, 'S1', 14, 'S1', 15, 'S1', 16, 'S1', 17, 'S1', 18, 'S1', 19, 'S1',
                20, 'S1', 21, 'S1', 22, 'S1', 23, 'S1', 24, 'S1', 25, 'S1', 26, 'S1', 27, 'S1', 28, 'S1', 29, 'S1', 30,
                'S1', 31, 'S1', 32, 'S1', 33, 'S1', 34, 'S1', 35, 'S1', 36, 'S1', 37, 'S1', 38, 'S1', 39, 'S1', 40,
                'S1', 41, 'S1', 42, 'S1', 43, 'S1', 44, 'S1', 45, 'S1', 46, 'S1', 47, 'S1', 48, 'S1', 49, 'S1', 50,
                'S1', 51, 'S1', 52, 'S1', 53, 'S1', 54, 'S1', 55, 'S1', 56, 'S1', 57, 'S1', 58, 'S1', 59, 'S1', 60,
                'S1', 61, 'S1', 62, 'S1', 63, 'S1', 64, 'S2', 'D'],
    'val_in': [0.18427716102, 'S1', 0.18427716102, 'S1', 0.18427716102, 'S1', 0.275991475966, 'S1', 0.275991475966,
               'S1', 0.275991475966, 'S1', 0.275991475966, 'S1', 0.275991475966, 'S1', 0.275991475966, 'S1',
               0.18427716102, 'S1', 0.18427716102, 'S1', 0.18427716102, 'S1', 0.34565714691, 'S1', 0.34565714691, 'S1',
               0.34565714691, 'S1', 0.1704767152044, 'S1', 0.1704767152044, 'S1', 0.1704767152044, 'S1',
               0.1704767152044, 'S1', 0.1704767152044, 'S1', 0.1704767152044, 'S1', 0.34565714691, 'S1', 0.34565714691,
               'S1', 0.34565714691, 'S1', 0.1234664378214, 'S1', 0.1234664378214, 'S1', 0.1234664378214, 'S1',
               0.282616682952, 'S1', 0.282616682952, 'S1', 0.282616682952, 'S1', 0.282616682952, 'S1', 0.282616682952,
               'S1', 0.282616682952, 'S1', 0.1234664378214, 'S1', 0.1234664378214, 'S1', 0.1234664378214, 'S1',
               0.250853276904, 'S1', 0.250853276904, 'S1', 0.250853276904, 'S1', 0.1066417854742, 'S1', 0.1066417854742,
               'S1', 0.1066417854742, 'S1', 0.1066417854742, 'S1', 0.1066417854742, 'S1', 0.1066417854742, 'S1',
               0.250853276904, 'S1', 0.250853276904, 'S1', 0.250853276904, 'S1', 0.0463412200974, 'S1', 0.0463412200974,
               'S1', 0.0463412200974, 'S1', 0.061186376815, 'S1', 0.061186376815, 'S1', 0.061186376815, 'S1',
               0.1297137398738, 'S1', 0.1297137398738, 'S1', 0.1297137398738, 'S1', 0.1297137398738, 'S1',
               0.1297137398738, 'S1', 0.1297137398738, 'S1', 0.061186376815, 'S1', 0.061186376815, 'S1', 0.061186376815,
               'S1', 0.03949387733, 'S1', 0.03949387733, 'S1', 0.03949387733, 'S2', 'D'],
    'ocrd_gold': list(range(0, 66)) + ['S0', 'D'],
    'icrd_gold': [0, 'S0', 1, 'S0', 2, 'S0', 3, 'S0', 4, 'S0', 5, 'S0', 6, 'S0', 7, 'S0', 8, 'S0', 9, 'S0', 10, 'S0',
                  11, 'S0', 12, 'S0', 13, 'S0', 14, 'S0', 15, 'S0', 16, 'S0', 17, 'S0', 18, 'S0', 19, 'S0', 20, 'S0',
                  21, 'S0', 22, 'S0', 23, 'S0', 24, 'S0', 25, 'S0', 26, 'S0', 27, 'S0', 28, 'S0', 29, 'S0', 30, 'S0',
                  31, 'S0', 32, 'S0', 33, 'S0', 34, 'S0', 35, 'S0', 36, 'S0', 37, 'S0', 38, 'S0', 39, 'S0', 40, 'S0',
                  41, 'S0', 42, 'S0', 43, 'S0', 44, 'S0', 45, 'S0', 46, 'S0', 47, 'S0', 48, 'S0', 49, 'S0', 50, 'S0',
                  51, 'S0', 52, 'S0', 53, 'S0', 54, 'S0', 55, 'S0', 56, 'S0', 57, 'S0', 58, 'S0', 59, 'S0', 60, 'S0',
                  61, 'S0', 62, 'S0', 63, 'S0', 64, 'S0', 65, 'S1', 'D'],
    'val_gold': [0.18427716102, 'S0', 0.18427716102, 'S0', 0.275991475966, 'S0', 0.275991475966, 'S0', 0.275991475966,
                 'S0',
                 0.275991475966, 'S0', 0.275991475966, 'S0', 0.275991475966, 'S0', 0.18427716102, 'S0', 0.18427716102,
                 'S0',
                 0.18427716102, 'S0', 0.34565714691, 'S0', 0.34565714691, 'S0', 0.34565714691, 'S0', 0.1704767152044,
                 'S0', 0.1704767152044,
                 'S0', 0.1704767152044, 'S0', 0.1704767152044, 'S0', 0.1704767152044, 'S0', 0.1704767152044, 'S0',
                 0.34565714691, 'S0',
                 0.34565714691, 'S0', 0.34565714691, 'S0', 0.1234664378214, 'S0', 0.1234664378214, 'S0',
                 0.1234664378214,
                 'S0', 0.282616682952, 'S0', 0.282616682952, 'S0', 0.282616682952, 'S0', 0.282616682952, 'S0',
                 0.282616682952,
                 'S0', 0.282616682952, 'S0', 0.1234664378214, 'S0', 0.1234664378214, 'S0', 0.1234664378214, 'S0',
                 0.250853276904,
                 'S0', 0.250853276904, 'S0', 0.250853276904, 'S0', 0.1066417854742, 'S0', 0.1066417854742, 'S0',
                 0.1066417854742,
                 'S0', 0.1066417854742, 'S0', 0.1066417854742, 'S0', 0.1066417854742, 'S0', 0.250853276904, 'S0',
                 0.250853276904,
                 'S0', 0.250853276904, 'S0', 0.0463412200974, 'S0', 0.0463412200974, 'S0', 0.0463412200974, 'S0',
                 0.061186376815,
                 'S0', 0.061186376815, 'S0', 0.061186376815, 'S0', 0.1297137398738, 'S0', 0.1297137398738, 'S0',
                 0.1297137398738,
                 'S0', 0.1297137398738, 'S0', 0.1297137398738, 'S0', 0.1297137398738, 'S0', 0.061186376815, 'S0',
                 0.061186376815,
                 'S0', 0.061186376815, 'S0', 0.03949387733, 'S0', 0.03949387733, 'S0', 0.03949387733, 'S0',
                 0.18427716102, 'S1', 'D']
}

@pytest.mark.skip("Old sparse accumulator 1 test")
@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2])
def test_spacc1_direct(arrs, debug_sim):
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
            sa.set_in_crd0(icrd.pop(0))
        if len(ocrd) > 0:
            sa.set_in_crd1(ocrd.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))

        sa.update()

        out_ocrd.append(sa.out_crd1())
        out_icrd.append(sa.out_crd0())
        out_val.append(sa.out_val())

        print("Timestep", time, "\t Done:", sa.out_done())

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


arrs_dict1 = {'ocrd_in': [0, 2, 'S0', 2, 'S1', 'D'],
              'icrd_in': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 2, 3, 'S2', 'D'],
              'val_in': [50, 5, 10, 'S0', 40, 4, 8, 'S1', -40, 33, 36, 'S2', 'D'],
              'icrd_gold': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
              'val_gold': [90, 9, 18, 'S0', -40, 33, 36, 'S1', 'D']}


# New sparse accumulator
@pytest.mark.parametrize("arrs", [arrs_dict1])
def test_spacc1new_direct(arrs, debug_sim):
    icrd = copy.deepcopy(arrs['icrd_in'])
    ocrd = copy.deepcopy(arrs['ocrd_in'])
    val = copy.deepcopy(arrs['val_in'])

    gold_icrd = copy.deepcopy(arrs['icrd_gold'])
    gold_val = copy.deepcopy(arrs['val_gold'])

    sa = SpAcc1New(valtype=int, val_stkn=True, debug=debug_sim)

    done = False
    time = 0
    out_icrd = []
    out_val = []
    while not done and time < TIMEOUT:
        if len(icrd) > 0:
            sa.set_in_crd0(icrd.pop(0))
        if len(ocrd) > 0:
            sa.set_in_crd1(ocrd.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))

        sa.update()

        out_icrd.append(sa.out_crd0())
        out_val.append(sa.out_val())

        print("Timestep", time, "\t Done:", sa.out_done())

        done = sa.out_done()
        time += 1

    out_icrd = remove_emptystr(out_icrd)
    out_val = remove_emptystr(out_val)

    if debug_sim:
        print("Inner Crd: ", out_icrd)
        print("Vals: ", out_val)

    assert (out_icrd == gold_icrd)
    assert (out_val == gold_val)


@pytest.mark.parametrize("dim", [2 ** x for x in range(2, 11, 2)])
def test_spacc1new_rand(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))

    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    gold_nd = np.sum(nd1, 0)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)
    sa = SpAcc1New(valtype=int, val_stkn=True, debug=debug_sim)

    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=dim, fill=fill, debug=debug_sim)

    done = False
    time = 0
    in_ref_B = [0, 'D']
    out_rdscan_B1 = []
    out_rdscan_B2 = []
    out_val_B = []
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())
        val_B.set_load(rdscan_B2.out_ref())

        # Inject random empty strings
        out_rdscan_B1.append(rdscan_B1.out_crd())
        out_rdscan_B2.append(rdscan_B2.out_crd())
        out_val_B.append(val_B.out_load())
        if random.random() < 0.2:
            out_rdscan_B1.append("")
        if random.random() < 0.2:
            out_rdscan_B2.append("")
        if random.random() < 0.2:
            out_val_B.append("")

        sa.set_in_crd1(out_rdscan_B1.pop(0))
        sa.set_in_crd0(out_rdscan_B2.pop(0))
        sa.set_val(out_val_B.pop(0))
        vals_X.set_input(sa.out_val())
        wrscan_X1.set_input(sa.out_crd0())

        rdscan_B1.update()
        rdscan_B2.update()
        val_B.update()
        sa.update()
        vals_X.update()
        wrscan_X1.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan B1:", rdscan_B1.out_done(),
              "\tSpAcc1New:", sa.out_done(),
              "\tArr:", val_B.out_done(),
              "\tWrScan:", vals_X.out_done(),
              "\tWrScan X1:", wrscan_X1.out_done(),
              )

        done = wrscan_X1.out_done()
        time += 1

    wrscan_X1.autosize()
    vals_X.autosize()

    out_crds = [wrscan_X1.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print("Input", in_mat_segs1, in_mat_crds1, in_mat_vals1)
        print("X seg", out_segs)
        print("X crd", out_crds)
        print("X val", out_val)
        print("Gold np", gold_nd)
        print("Gold Tuple", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))


arrs_dict1 = {'crd2_in': [0, 1, 'S0', 'D'],
              'crd1_in': [0, 2, 'S0', 2, 'S1', 'D'],
              'crd0_in': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 2, 3, 'S2', 'D'],
              'val_in': [50, 5, 10, 'S0', 40, 4, 8, 'S1', -40, 33, 36, 'S2', 'D'],
              'crd1_gold': [0, 2, 'S0', 'D'],
              'crd0_gold': [0, 2, 3, 'S0', 0, 2, 3, 'S1', 'D'],
              'val_gold': [50, 5, 10, 'S0', 0, 37, 44, 'S1', 'D']}


# New sparse accumulator
@pytest.mark.parametrize("arrs", [arrs_dict1])
def test_spacc2new_direct(arrs, debug_sim):
    crd2 = copy.deepcopy(arrs['crd2_in'])
    crd1 = copy.deepcopy(arrs['crd1_in'])
    crd0 = copy.deepcopy(arrs['crd0_in'])
    val = copy.deepcopy(arrs['val_in'])

    gold_crd1 = copy.deepcopy(arrs['crd1_gold'])
    gold_crd0 = copy.deepcopy(arrs['crd0_gold'])
    gold_val = copy.deepcopy(arrs['val_gold'])

    sa = SpAcc2New(valtype=int, val_stkn=True, debug=debug_sim)

    done = False
    time = 0
    out_icrd = []
    out_val = []
    while not done and time < TIMEOUT:
        if len(crd0) > 0:
            sa.set_in_crd0(crd0.pop(0))
        if len(crd1) > 0:
            sa.set_in_crd1(crd1.pop(0))
        if len(crd2) > 0:
            sa.set_in_crd2(crd2.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))

        sa.update()

        out_icrd.append(sa.out_crd0())
        out_val.append(sa.out_val())

        print("Timestep", time, "\t Done:", sa.out_done())

        done = sa.out_done()
        time += 1

    out_icrd = remove_emptystr(out_icrd)
    out_val = remove_emptystr(out_val)

    if debug_sim:
        print("Inner Crd: ", out_icrd)
        print("Vals: ", out_val)

    assert (out_icrd == gold_icrd)
    assert (out_val == gold_val)