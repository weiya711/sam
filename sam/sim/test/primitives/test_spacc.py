import copy
import pytest
import random
import numpy as np

from sam.sim.src.accumulator import SpAcc1, SpAcc2
from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.src.array import Array
from sam.sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sam.sim.src.base import remove_emptystr
from sam.sim.test.test import TIMEOUT, gen_n_comp_arrs, gen_val_arr, get_point_list, \
    convert_point_tuple, convert_ndarr_point_tuple, convert_point_tuple_ndarr, check_point_tuple

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

    sa = SpAcc1(valtype=int, val_stkn=True, debug=debug_sim)

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
    sa = SpAcc1(valtype=int, val_stkn=True, debug=debug_sim)

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

# [[0, 1], [0, 4], [0, 2, 3, 4, 5]] [[0], [0, 1, 2, 3], [0, 2, 3, 1, 3]] [-60, 85, 314, 241, -887]
arrs_dict2 = {'crd2_in': [0, 'S0', 'D'],
              'crd1_in': [0, 1, 2, 3, 'S1', 'D'],
              'crd0_in': [0, 2, 'S0', 3, 'S0', 1, 'S0', 3, 'S2', 'D'],
              'val_in': [-60, 85, 'S0', 314, 'S0', 241, 'S0', -887, 'S2', 'D'],
              'crd1_gold': [0, 1, 2, 3, 'S0', 'D'],
              'crd0_gold': [0, 2, 'S0', 3, 'S0', 1, 'S0', 3, 'S1', 'D'],
              'val_gold': [-60, 85, 'S0', 314, 'S0', 241, 'S0', -887, 'S1', 'D']}

# [[0, 1], [0, 3], [0, 4, 6, 8]] [[1], [0, 1, 2], [0, 1, 2, 3, 1, 3, 0, 2]] [637, 210, -847, 358, 162, 687, 95, -91]
arrs_dict3 = {'crd2_in': [1, 'S0', 'D'],
              'crd1_in': [0, 1, 2, 'S1', 'D'],
              'crd0_in': [0, 1, 2, 3, 'S0', 1, 3, 'S0', 0, 2, 'S2', 'D'],
              'val_in': [637, 210, -847, 358, 'S0', 162, 687, 'S0', 95, -91, 'S2', 'D'],
              'crd1_gold': [0, 1, 2, 'S0', 'D'],
              'crd0_gold': [0, 1, 2, 3, 'S0', 1, 3, 'S0', 0, 2, 'S1', 'D'],
              'val_gold': [637, 210, -847, 358, 'S0', 162, 687, 'S0', 95, -91, 'S1', 'D']}


# New sparse accumulator
@pytest.mark.parametrize("arrs", [arrs_dict1, arrs_dict2, arrs_dict3])
def test_spacc2new_direct(arrs, debug_sim):
    crd2 = copy.deepcopy(arrs['crd2_in'])
    crd1 = copy.deepcopy(arrs['crd1_in'])
    crd0 = copy.deepcopy(arrs['crd0_in'])
    val = copy.deepcopy(arrs['val_in'])

    gold_crd1 = copy.deepcopy(arrs['crd1_gold'])
    gold_crd0 = copy.deepcopy(arrs['crd0_gold'])
    gold_val = copy.deepcopy(arrs['val_gold'])

    sa = SpAcc2(valtype=int, val_stkn=True, debug=debug_sim)

    done = False
    time = 0
    out_crd1 = []
    out_crd0 = []
    out_val = []
    while not done and time < TIMEOUT:
        if len(crd2) > 0:
            sa.set_in_crd2(crd2.pop(0))
        if len(crd1) > 0:
            sa.set_in_crd1(crd1.pop(0))
        if len(crd0) > 0:
            sa.set_in_crd0(crd0.pop(0))
        if len(val) > 0:
            sa.set_val(val.pop(0))

        sa.update()

        out_crd1.append(sa.out_crd1())
        out_crd0.append(sa.out_crd0())
        out_val.append(sa.out_val())

        print("Timestep", time, "\t Done:", sa.out_done())

        done = sa.out_done()
        time += 1

    out_crd1 = remove_emptystr(out_crd1)
    out_crd0 = remove_emptystr(out_crd0)
    out_val = remove_emptystr(out_val)

    if debug_sim:
        print("Crd1: ", out_crd1)
        print("Crd0: ", out_crd0)
        print("Vals: ", out_val)

    assert (out_crd1 == gold_crd1)
    assert (out_crd0 == gold_crd0)
    assert (out_val == gold_val)


@pytest.mark.parametrize("dim", [2 ** x for x in range(1, 5, 1)])
def test_spacc2new_rand(dim, debug_sim, max_val=1000, fill=0):
    np.random.seed(0)
    random.seed(0)

    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(3, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))

    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    gold_nd = np.sum(nd1, 0)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)
    rdscan_B3 = CompressedCrdRdScan(crd_arr=in_mat_crds1[2], seg_arr=in_mat_segs1[2], debug=debug_sim)

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)
    sa = SpAcc2(valtype=int, val_stkn=True, debug=debug_sim)

    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=dim, fill=fill, debug=debug_sim)
    wrscan_X2 = CompressWrScan(seg_size=dim+1, size=dim*dim, fill=fill, debug=debug_sim)

    done = False
    time = 0
    in_ref_B = [0, 'D']
    out_rdscan_B1 = []
    out_rdscan_B2 = []
    out_rdscan_B3 = []
    out_val_B = []
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())
        rdscan_B3.set_in_ref(rdscan_B2.out_ref())
        val_B.set_load(rdscan_B3.out_ref())

        # Inject random empty strings
        out_rdscan_B1.append(rdscan_B1.out_crd())
        out_rdscan_B2.append(rdscan_B2.out_crd())
        out_rdscan_B3.append(rdscan_B3.out_crd())
        out_val_B.append(val_B.out_load())

        # Inject random delay
        if random.random() < 0.2:
            out_rdscan_B1.append("")
        if random.random() < 0.2:
            out_rdscan_B2.append("")
        if random.random() < 0.2:
            out_val_B.append("")

        sa.set_in_crd2(out_rdscan_B1.pop(0))
        sa.set_in_crd1(out_rdscan_B2.pop(0))
        sa.set_in_crd0(out_rdscan_B3.pop(0))
        sa.set_val(out_val_B.pop(0))

        vals_X.set_input(sa.out_val())
        wrscan_X1.set_input(sa.out_crd1())
        wrscan_X2.set_input(sa.out_crd0())

        rdscan_B1.update()
        rdscan_B2.update()
        rdscan_B3.update()
        val_B.update()
        sa.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan B1:", rdscan_B1.out_done(),
              "\tRdScan B2:", rdscan_B2.out_done(),
              "\tRdScan B3:", rdscan_B3.out_done(),
              "\tSpAcc1New:", sa.out_done(),
              "\tArr:", val_B.out_done(),
              "\tWrScan:", vals_X.out_done(),
              "\tWrScan X1:", wrscan_X1.out_done(),
              "\tWrScan X2:", wrscan_X2.out_done(),
              )

        done = wrscan_X2.out_done() and wrscan_X1.out_done() and vals_X.out_done()
        time += 1

    wrscan_X1.autosize()
    wrscan_X2.autosize()
    vals_X.autosize()

    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print("Input", in_mat_segs1, in_mat_crds1, in_mat_vals1)
        print(nd1)
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