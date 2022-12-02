import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
from sam.sim.src.unary_alu import Max, Exp
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce, MaxReduce
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
def test_tensor3_relu(samBench, frosttname, check_gold, report_stats, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "orig", "ssss0123")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B2_seg_filename = os.path.join(B_dirname, "B2_seg.txt")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "B2_crd.txt")
    B_crd2 = read_inputs(B2_crd_filename)

    B3_seg_filename = os.path.join(B_dirname, "B3_seg.txt")
    B_seg3 = read_inputs(B3_seg_filename)
    B3_crd_filename = os.path.join(B_dirname, "B3_crd.txt")
    B_crd3 = read_inputs(B3_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    fiberlookup_Bi = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)

    fiberwrite_X0 = CompressWrScan(seg_size=2, size=2 * len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)

    fiberwrite_X1 = CompressWrScan(seg_size=2 * len(B_crd0) + 1, size=2 * len(B_crd1), fill=fill,
                                     debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)

    fiberwrite_X2 = CompressWrScan(seg_size=2 * len(B_crd1) + 1, size=len(B_crd2) * 2, fill=fill,
                                     debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats)

    fiberwrite_X3 = CompressWrScan(seg_size=2 * len(B_crd2) + 1, size=len(B_vals) * 2, fill=fill,
                                     debug=debug_sim, statistics=report_stats)
    arrayvals_B = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    exp_1 = Exp(debug=debug_sim, statistics=report_stats)
    reduce = Reduce(debug=debug_sim, statistics=report_stats)
    div = Divide2(debug=debug_sim, statistics=report_stats)
    # drop_1 = CrdDrop(debug=debug_sim, statistics=report_stats)
    # drop_2 = CrdDrop(debug=debug_sim, statistics=report_stats)
    # drop_3 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    # in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    if debug_sim:
        print("blocks done")

    # exp_1.set_in2(0)
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))
        fiberwrite_X0.set_input(fiberlookup_Bi.out_crd())

        fiberlookup_Bj.set_in_ref(fiberlookup_Bi.out_ref())
        fiberwrite_X1.set_input(fiberlookup_Bj.out_crd())

        fiberlookup_Bk.set_in_ref(fiberlookup_Bj.out_ref())
        fiberwrite_X2.set_input(fiberlookup_Bk.out_crd())

        fiberlookup_Bl.set_in_ref(fiberlookup_Bl.out_ref())
        fiberwrite_X3.set_input(fiberlookup_Bl.out_crd())

        arrayvals_B.set_load(fiberlookup_Bl.out_ref())

        exp_1.set_in1(arrayvals_B.out_load())
        # print(arrayvals_B_5.out_load())
        # print(max_1.out_val())
        reduce.set_in_val(exp_1.out_val())

        div.set_in1(exp_1.out_val())
        div.set_in2(reduce.out_val())

        fiberwrite_Xvals.set_input(exp_1.out_val())
        # fiberwrite_Xvals_0.set_input(drop_1.out_crd_inner())

        fiberlookup_Bi.update()
        fiberlookup_Bj.update()
        fiberlookup_Bk.update()
        fiberlookup_Bl.update()
        arrayvals_B.update()
        exp_1.update()
        reduce.update()
        div.update()
        # drop_2.update()
        # drop_3.update()

        fiberwrite_Xvals.update()
        fiberwrite_X0.update()
        fiberwrite_X1.update()
        fiberwrite_X2.update()
        fiberwrite_X3.update()

        done = fiberwrite_X0.out_done() and fiberwrite_X1.out_done() and fiberwrite_X2.out_done() and \
            fiberwrite_Xvals.out_done()

    fiberwrite_X0.autosize()
    fiberwrite_X1.autosize()
    fiberwrite_X2.autosize()
    fiberwrite_Xvals.autosize()

    out_crds = [fiberwrite_X0.get_arr(), fiberwrite_X1.get_arr(), fiberwrite_X2.get_arr()]
    out_segs = [fiberwrite_X0.get_seg_arr(), fiberwrite_X1.get_seg_arr(), fiberwrite_X2.get_seg_arr()]
    out_vals = fiberwrite_Xvals.get_arr()

    print(out_vals)

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
        check_gold_tensor3_elemadd(frosttname, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
