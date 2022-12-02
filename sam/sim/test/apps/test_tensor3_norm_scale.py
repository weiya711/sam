import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
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


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
def test_tensor3_norm_scale(samBench, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, , "orig", "s0")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    fiberlookup_Bi_17 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    repsiggen_i_15 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bi_14 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    intersectj_11 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_10 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    repsiggen_k_8 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ck_7 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_C_5 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_17.set_in_ref(in_ref_B.pop(0))
        fiberwrite_X0_3.set_input(fiberlookup_Bi_17.out_crd())
        fiberlookup_Bj_13.set_in_ref(fiberlookup_Bi_17.out_ref())
        repsiggen_i_15.set_istream(fiberlookup_Bi_17.out_crd())
        repeat_Bi_14.set_in_repsig(repsiggen_i_15.out_repsig())
        fiberlookup_Cj_12.set_in_ref(repeat_Bi_14.out_ref())
        intersectj_11.set_in1(fiberlookup_Bj_13.out_ref(), fiberlookup_Bj_13.out_crd())
        intersectj_11.set_in2(fiberlookup_Cj_12.out_ref(), fiberlookup_Cj_12.out_crd())
        fiberwrite_X1_2.set_input(intersectj_11.out_crd())
        fiberlookup_Bk_10.set_in_ref(intersectj_11.out_ref1())
        fiberwrite_X2_1.set_input(fiberlookup_Bk_10.out_crd())
        arrayvals_B_6.set_load(fiberlookup_Bk_10.out_ref())
        repsiggen_k_8.set_istream(fiberlookup_Bk_10.out_crd())
        repeat_Ck_7.set_in_repsig(repsiggen_k_8.out_repsig())
        arrayvals_C_5.set_load(repeat_Ck_7.out_ref())
        mul_4.set_in1(arrayvals_B_6.out_val())
        mul_4.set_in2(arrayvals_C_5.out_val())
        fiberwrite_Xvals_0.set_input(mul_4.out_val())
        fiberlookup_Bi_17.update()

        fiberwrite_X0_3.update()
        fiberlookup_Bj_13.update()
        repsiggen_i_15.update()
        repeat_Bi_14.update()
        fiberlookup_Cj_12.update()
        intersectj_11.update()
        fiberwrite_X1_2.update()
        fiberlookup_Bk_10.update()
        fiberwrite_X2_1.update()
        arrayvals_B_6.update()
        repsiggen_k_8.update()
        repeat_Ck_7.update()
        arrayvals_C_5.update()
        mul_4.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = 
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    sample_dict = fiberlookup_Bi_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_17" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bi_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bi_14" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_13" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_12" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_10" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ck_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ck_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_6" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_norm_scale(, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
