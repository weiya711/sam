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
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_norm_scale(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    b_dirname = os.path.join(formatted_dir, frosttname, "tensor3_norm_scale")
    b_shape_filename = os.path.join(b_dirname, "tensor_b_mode_shape")
    b_shape = read_inputs(b_shape_filename)

    b0_seg_filename = os.path.join(b_dirname, "tensor_b_mode_0_seg")
    b_seg0 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "tensor_b_mode_0_crd")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "tensor_b_mode_vals")
    b_vals = read_inputs(b_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, frosttname, "tensor3_norm_scale")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename)

    C2_seg_filename = os.path.join(C_dirname, "tensor_C_mode_2_seg")
    C_seg2 = read_inputs(C2_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "tensor_C_mode_2_crd")
    C_crd2 = read_inputs(C2_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)


    fiberlookup_Ci_17 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj_13 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_i_15 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_bi_14 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_bj_12 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim, statistics=report_stats)
    intersectj_11 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=C_shape[0] + 1, size=C_shape[0] * b_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck_10 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=C_shape[0] * b_shape[0] + 1, size=C_shape[0] * b_shape[0] * C_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_k_8 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_bk_7 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_b_5 = Array(init_arr=b_vals, debug=debug_sim, statistics=report_stats)
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[0] * b_shape[0] * C_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_C = [0, 'D']
    in_ref_b = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_17.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Cj_13.set_in_ref(fiberlookup_Ci_17.out_ref())
        fiberwrite_X0_3.set_input(fiberlookup_Ci_17.out_crd())
        repsiggen_i_15.set_istream(fiberlookup_Ci_17.out_crd())
        if len(in_ref_b) > 0:
            repeat_bi_14.set_in_ref(in_ref_b.pop(0))
        repeat_bi_14.set_in_repsig(repsiggen_i_15.out_repsig())
        fiberlookup_bj_12.set_in_ref(repeat_bi_14.out_ref())
        intersectj_11.set_in1(fiberlookup_bj_12.out_ref(), fiberlookup_bj_12.out_crd())
        intersectj_11.set_in2(fiberlookup_Cj_13.out_ref(), fiberlookup_Cj_13.out_crd())
        fiberwrite_X1_2.set_input(intersectj_11.out_crd())
        fiberlookup_Ck_10.set_in_ref(intersectj_11.out_ref2())
        arrayvals_C_6.set_load(fiberlookup_Ck_10.out_ref())
        fiberwrite_X2_1.set_input(fiberlookup_Ck_10.out_crd())
        repsiggen_k_8.set_istream(fiberlookup_Ck_10.out_crd())
        repeat_bk_7.set_in_ref(intersectj_11.out_ref1())
        repeat_bk_7.set_in_repsig(repsiggen_k_8.out_repsig())
        arrayvals_b_5.set_load(repeat_bk_7.out_ref())
        mul_4.set_in1(arrayvals_b_5.out_val())
        mul_4.set_in2(arrayvals_C_6.out_val())
        fiberwrite_Xvals_0.set_input(mul_4.out_val())
        fiberlookup_Ci_17.update()

        fiberlookup_Cj_13.update()
        fiberwrite_X0_3.update()
        repsiggen_i_15.update()
        repeat_bi_14.update()
        fiberlookup_bj_12.update()
        intersectj_11.update()
        fiberwrite_X1_2.update()
        fiberlookup_Ck_10.update()
        arrayvals_C_6.update()
        fiberwrite_X2_1.update()
        repsiggen_k_8.update()
        repeat_bk_7.update()
        arrayvals_b_5.update()
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
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_b_shape"] = b_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberlookup_Ci_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_bi_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_bi_14" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_bj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_bj_12" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_bk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_bk_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_b_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_b_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_10" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_13" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_norm_scale(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
