import pytest
import time
import scipy.sparse
import numpy as np
from sam.sim.src.compression import ValDropper
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2 #, Divider2
from sam.sim.src.crd_masker import RandomDropout, LowerTriangular, UpperTriangular, Diagonal, Tril
from sam.sim.src.unary_alu import Max
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
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_dropout(samBench, frosttname, check_gold, report_stats, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor3_dropout")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename)

    B2_seg_filename = os.path.join(B_dirname, "tensor_B_mode_2_seg")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "tensor_B_mode_2_crd")
    B_crd2 = read_inputs(B2_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    # pytest.set_trace()

    fiberlookup_Bi_14 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)

    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=2 * len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_11 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)

    fiberwrite_X1_2 = CompressWrScan(seg_size=2 * len(B_crd0) + 1, size=2 * len(B_crd1), fill=fill,
                                     debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)

    fiberwrite_X2_1 = CompressWrScan(seg_size=2 * len(B_crd1) + 1, size=len(B_vals) * 2, fill=fill,
                                     debug=debug_sim, statistics=report_stats)
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    crd = CrdHold(debug=debug_sim)
    # trilower = LowerTriangular(debug_sim=debug_sim)
    drop_prob = 0.4
    dropout = RandomDropout(dimension=2, drop_probability=drop_prob, debug=debug_sim)
    prob = np.random.rand(*B_shape)
    tril = Tril(debug=debug_sim, statistics=report_stats)
    dropout.set_prob(prob, drop_prob)

    # drop_1 = Compression(debug=debug_sim, statistics=report_stats)
    # drop_1 = CrdDrop(debug=debug_sim, statistics=report_stats)
    drop_1 = CrdDrop(debug=debug_sim, statistics=report_stats)
    # drop_3 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    # in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    if debug_sim:
        print("blocks done")

    arr_vals = []
    out_arr = []
    out1_arr = []
    input_arr = []
    val_arr = []

    dropin = []
    dropin1 = []
    drop_arr = []
    drop1_arr = []
    out_drop = []
    last_i = 0
    # max_1.set_in2(0)
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_14.set_in_ref(in_ref_B.pop(0))

        fiberlookup_Bj_11.set_in_ref(fiberlookup_Bi_14.out_ref())

        fiberlookup_Bk_8.set_in_ref(fiberlookup_Bj_11.out_ref())

        # max_1.set_in1(arrayvals_B_5.out_load())
        crd.set_outer_crd(fiberlookup_Bj_11.out_crd())
        crd.set_inner_crd(fiberlookup_Bk_8.out_crd())

        val_arr.append(crd.out_crd_outer())
        # out_arr.append(fiberlookup_Bj_11.out_crd())

        # print("input j:", remove_emptystr(out_arr))
        # print("input j crdhold:", remove_emptystr(val_arr))

        i = fiberlookup_Bi_14.out_crd()
        if i != "" and not is_stkn(i) and i != 'D':
            last_i = int(i)

        arrayvals_B_5.set_load(fiberlookup_Bk_8.out_ref())

        # dropout.set_inner_crd(crd.out_crd_inner())
        # dropout.set_outer_crd(crd.out_crd_outer())
        # dropout.set_inner_ref(arrayvals_B_5.out_val())
        tril.set_inner_crd(fiberlookup_Bk_8.out_crd())
        tril.set_outer_crd(fiberlookup_Bj_11.out_crd())
        tril.set_inner_ref(arrayvals_B_5.out_val())

        # out_drop.append(dropout.out_crd(1))
        # out_arr.append(dropout.out_crd(0))
        # print("out j:", remove_emptystr(out_drop))
        # print("out k:", remove_emptystr(out_arr))

        # arrayvals_B_5.set_load(dropout.out_ref())

        # print(arrayvals_B_5.out_load())
        # print(max_1.out_val())
        # fiberwrite_Xvals_0.set_input(arrayvals_B_5.out_val())
        fiberwrite_Xvals_0.set_input(tril.out_ref())

        drop_1.set_outer_crd(fiberlookup_Bi_14.out_crd())
        drop_1.set_inner_crd(dropout.out_crd(1))


        # fiberwrite_X0_3.set_input(drop_1.out_crd_outer())
        fiberwrite_X0_3.set_input(fiberlookup_Bi_14.out_crd())
        # fiberwrite_X1_2.set_input(fiberlookup_Bj_11.out_crd())
        fiberwrite_X1_2.set_input(tril.out_crd_outer())
        fiberwrite_X2_1.set_input(tril.out_crd_inner())
        # fiberwrite_X1_2.set_input(dropout.out_crd(1))
        # fiberwrite_X2_1.set_input(dropout.out_crd(0))

        # input_arr.append(fiberlookup_Bj_11.out_crd())
        # arr_vals.append(fiberlookup_Bk_8.out_crd())
        # input_arr.append(dropout.out_crd(1))
        # arr_vals.append(dropout.out_crd(0))
        # print("random outer", remove_emptystr(input_arr))
        # print("random inner", remove_emptystr(arr_vals))

        # fiberwrite_Xvals_0.set_input(max_1.out_val())
        # fiberwrite_Xvals_0.set_input(drop_1.out_crd_inner())

        fiberlookup_Bi_14.update()
        fiberlookup_Bj_11.update()
        fiberlookup_Bk_8.update()
        crd.update()
        arrayvals_B_5.update()
        dropout.update()
        tril.update()
        fiberwrite_Xvals_0.update()
        drop_1.update()
        fiberwrite_X0_3.update()
        fiberwrite_X1_2.update()
        fiberwrite_X2_1.update()
        # val_arr.append(fiberwrite_Xvals_0.get_arr())

        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and \
            fiberwrite_Xvals_0.out_done()
        # print(drop_1.out_done())
        # print(drop_2.out_done())
        # print(drop_3.out_done())
        # done = drop_1.out_done() and drop_2.out_done() and drop_3.out_done()
        # done = max_1.out_done()

        time_cnt += 1

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    # print(arr_vals)
    # print("Values:")
    print("segs:", out_segs)
    print("crds:", out_crds)
    print(out_vals)

    pytest.set_trace()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    # extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    # extra_info["tensor_C/nnz"] = len(C_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    # sample_dict = unioni_13.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["unioni_13" + "/" + k] = sample_dict[k]

    # sample_dict = unionj_10.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["unionj_10" + "/" + k] = sample_dict[k]

    # sample_dict = unionk_7.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["unionk_7" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    # sample_dict = arrayvals_C_6.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["arrayvals_C_6" + "/" + k] = sample_dict[k]

    # sample_dict = add_4.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["add_4" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_14" + "/" + k] = sample_dict[k]

    # sample_dict = fiberlookup_Ci_15.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberlookup_Ci_15" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_11" + "/" + k] = sample_dict[k]

    # sample_dict = fiberlookup_Cj_12.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberlookup_Cj_12" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_8" + "/" + k] = sample_dict[k]

    # sample_dict = fiberlookup_Ck_9.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberlookup_Ck_9" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        cast = 0
        scalar = 0
        check_gold_tensor3_dropout(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, prob, drop_prob, dropout.random_dropped, scalar, "sss012")
    samBench(bench, extra_info)
