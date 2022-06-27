import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

arr_dict1 = {"vi_seg": [0, 2],
             "vi_crd": [1, 2],
             "vi_vals": [1, 2],
             "vj_seg": [0, 2],
             "vj_crd": [1, 2],
             "vj_vals": [5, 6],
             "mi_seg": [0, 2],
             "mi_crd": [0, 2],
             "mj_seg": [0, 1, 2],
             "mj_crd": [1, 1],
             "m_vals": [3, 4],
             "gold_seg": [0, 3],
             "gold_crd": [0, 1, 2],
             "gold_vals": [-15, 1, -18]}

arr_dict2 = {"vi_seg": [0, 5],
             "vi_crd": [1, 2, 4, 8, 15],
             "vi_vals": [1, 1, 1, 1, 1],
             "vj_seg": [0, 4],
             "vj_crd": [0, 3, 9, 15],
             "vj_vals": [2, 2, 2, 2],
             "mi_seg": [0, 16],
             "mi_crd": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
             "mj_seg": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32],
             "mj_crd": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
                        14, 14, 15, 15, 1],
             "m_vals": [-3] * 32}


@pytest.mark.parametrize("arrs", [arr_dict1, arr_dict2])
def test_unit_mat_residual(samBench, arrs, check_gold, debug_sim, fill=0):
    C_shape = (16, 16)

    C_seg0 = copy.deepcopy(arrs["mi_seg"])
    C_crd0 = copy.deepcopy(arrs["mi_crd"])
    C_seg1 = copy.deepcopy(arrs["mj_seg"])
    C_crd1 = copy.deepcopy(arrs["mj_crd"])
    C_vals = copy.deepcopy(arrs["m_vals"])

    b_shape = [C_shape[0]]
    b_seg0 = copy.deepcopy(arrs["vi_seg"])
    b_crd0 = copy.deepcopy(arrs["vi_crd"])
    b_vals = copy.deepcopy(arrs["vi_vals"])

    d_shape = [C_shape[1]]
    d_seg0 = copy.deepcopy(arrs["vj_seg"])
    d_crd0 = copy.deepcopy(arrs["vj_crd"])
    d_vals = copy.deepcopy(arrs["vj_vals"])

    fiberlookup_bi_17 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim)
    fiberlookup_Ci_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    unioni_16 = Union2(debug=debug_sim)
    fiberlookup_Cj_11 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=b_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_14 = RepeatSigGen(debug=debug_sim)
    repeat_di_13 = Repeat(debug=debug_sim)
    fiberlookup_dj_12 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim)
    intersectj_10 = Intersect2(debug=debug_sim)
    unionj1 = Union2(debug=debug_sim)
    unionj2 = Union2(debug=debug_sim)
    repsiggen_j_9 = RepeatSigGen(debug=debug_sim)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim)
    arrayvals_d_7 = Array(init_arr=d_vals, debug=debug_sim)
    repeat_bj_8 = Repeat(debug=debug_sim, union=True)
    mul_5 = Multiply2(debug=debug_sim)
    arrayvals_b_4 = Array(init_arr=b_vals, debug=debug_sim)
    add_3 = Add2(debug=debug_sim, neg2=True)
    reduce_2 = Reduce(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * b_shape[0], fill=fill, debug=debug_sim)
    in_ref_b = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_d = [0, 'D']
    done = False
    time_cnt = 0

    temp = []
    temp1 = []
    temp2 = []
    temp3 = []
    temp4 = []
    temp5 = []
    temp6 = []
    temp7 = []
    temp8 = []
    temp9 = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_b) > 0:
            fiberlookup_bi_17.set_in_ref(in_ref_b.pop(0))
        fiberlookup_bi_17.update()

        if len(in_ref_C) > 0:
            fiberlookup_Ci_18.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci_18.update()

        unioni_16.set_in1(fiberlookup_bi_17.out_ref(), fiberlookup_bi_17.out_crd())
        unioni_16.set_in2(fiberlookup_Ci_18.out_ref(), fiberlookup_Ci_18.out_crd())
        unioni_16.update()

        fiberlookup_Cj_11.set_in_ref(unioni_16.out_ref2())
        fiberlookup_Cj_11.update()

        fiberwrite_x0_1.set_input(unioni_16.out_crd())
        fiberwrite_x0_1.update()

        repsiggen_i_14.set_istream(unioni_16.out_crd())
        repsiggen_i_14.update()

        if len(in_ref_d) > 0:
            repeat_di_13.set_in_ref(in_ref_d.pop(0))
        repeat_di_13.set_in_repsig(repsiggen_i_14.out_repsig())
        repeat_di_13.update()

        fiberlookup_dj_12.set_in_ref(repeat_di_13.out_ref())
        fiberlookup_dj_12.update()

        temp.append(fiberlookup_Cj_11.out_crd())
        temp1.append(fiberlookup_dj_12.out_crd())
        temp2.append(fiberlookup_Cj_11.out_ref())
        temp3.append(fiberlookup_dj_12.out_ref())
        intersectj_10.set_in1(fiberlookup_dj_12.out_ref(), fiberlookup_dj_12.out_crd())
        intersectj_10.set_in2(fiberlookup_Cj_11.out_ref(), fiberlookup_Cj_11.out_crd())
        intersectj_10.update()

        repsiggen_j_9.set_istream(intersectj_10.out_crd())
        repsiggen_j_9.update()

        arrayvals_C_6.set_load(intersectj_10.out_ref2())
        arrayvals_C_6.update()

        arrayvals_d_7.set_load(intersectj_10.out_ref1())
        arrayvals_d_7.update()

        repeat_bj_8.set_in_ref(unioni_16.out_ref1())
        repeat_bj_8.set_in_repsig(repsiggen_j_9.out_repsig())
        repeat_bj_8.update()

        # unionj1.set_in1(repeat_bj_8.out_ref())
        # unionj1.set_in2(intersectj_10.out_crd(), intersectj_10.out_ref1())
        # unionj1.update()
        #
        # unionj2.set_in1(repeat_bj_8.out_ref())
        # unionj2.set_in2(intersectj_10.out_crd(), intersectj_10.out_ref2())
        # unionj2.update()

        # temp7.append(unioni_16.out_crd())
        # temp6.append(unioni_16.out_ref1())
        # temp8.append(unioni_16.out_ref2())
        # temp5.append(repeat_bj_8.out_ref())
        # temp4.append(arrayvals_b_4.out_val())
        # temp9.append(repsiggen_j_9.out_repsig())
        # print("union crd", remove_emptystr(temp7))
        # print("Ci ref", remove_emptystr(temp8))
        # print("bi ref", remove_emptystr(temp6))
        # print("Cj", remove_emptystr(temp))
        # print(remove_emptystr(temp2))
        # print("dj", remove_emptystr(temp1))
        # print(remove_emptystr(temp3))
        # print("bj rep", remove_emptystr(temp5))
        # print("rsg", remove_emptystr(temp9))
        # print()

        arrayvals_b_4.set_load(repeat_bj_8.out_ref())
        arrayvals_b_4.update()

        mul_5.set_in1(arrayvals_C_6.out_val())
        mul_5.set_in2(arrayvals_d_7.out_val())
        mul_5.update()

        add_3.set_in1(arrayvals_b_4.out_val())
        add_3.set_in2(mul_5.out_val())
        add_3.update()

        reduce_2.set_in_val(add_3.out_val())
        reduce_2.update()

        fiberwrite_xvals_0.set_input(reduce_2.out_val())
        fiberwrite_xvals_0.update()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()

    if "gold_seg" in arrs.keys():
        gold_seg = copy.deepcopy(arrs["gold_seg"])
        gold_crd = copy.deepcopy(arrs["gold_crd"])
        gold_vals = copy.deepcopy(arrs["gold_vals"])

        assert out_crds[0] == gold_crd
        assert out_segs[0] == gold_seg
        assert out_vals == gold_vals

    else:
        B_scipy = scipy.sparse.csr_matrix((C_vals, C_crd1, C_seg1), shape=C_shape)
        b_nd = np.zeros(b_shape)
        d_nd = np.zeros(d_shape)

        for i in range(len(b_crd0)):
            val = b_vals[i]
            crd = b_crd0[i]
            b_nd[crd] = val

        for i in range(len(d_crd0)):
            val = d_vals[i]
            crd = d_crd0[i]
            d_nd[crd] = val

        gold_nd = b_nd - B_scipy @ d_nd

        gold_tup = convert_ndarr_point_tuple(gold_nd)
        if debug_sim:
            print("Out segs:", out_segs)
            print("Out crds:", out_crds)
            print("Out vals:", out_vals)
            print("Dense Vec1:\n", b_nd)
            print("Dense Mat1:\n", B_scipy.transpose().toarray())
            print("Dense Vec2:\n", d_nd)
            print("Dense Gold:", gold_nd)
            print("Gold:", gold_tup)

        if not out_vals:
            assert out_vals == gold_tup
        elif not gold_tup:
            assert all([v == 0 for v in out_vals])
        else:
            out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_vals))
            out_tup = remove_zeros(out_tup)
            assert (check_point_tuple(out_tup, gold_tup))
