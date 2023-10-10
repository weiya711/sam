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
from sam.sim.src.compression import ValDropper
from sam.sim.src.unary_alu import Max
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv
cwd = os.getcwd()
formatted_dir = os.getenv('CUSTOM_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_matmul_ijk_crddrop(samBench, ssname, cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "matmul_ijk_crddrop")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname,  "tensor_B_mode_0_seg" )
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd" )
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg" )
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd" )
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, ssname, "matmul_ijk_crddrop")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname,  "tensor_C_mode_0_seg" )
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd" )
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg" )
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd" )
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)


    fiberlookup_Bi_17 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    repsiggen_i_15 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ci_14 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj_13 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck_9 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    repsiggen_j_11 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bj_10 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    intersectk_7 = Intersect2(debug=debug_sim, statistics=report_stats)
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    mul_4 = Multiply2(debug=debug_sim, statistics=report_stats)
    crddrop_19 = CrdDrop(debug=debug_sim, statistics=report_stats)
    reduce_3 = Reduce(debug=debug_sim, statistics=report_stats)
    crddrop_18 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_17.set_in_ref(in_ref_B.pop(0))
        repsiggen_i_15.set_istream(fiberlookup_Bi_17.out_crd())
        if len(in_ref_C) > 0:
            repeat_Ci_14.set_in_ref(in_ref_C.pop(0))
        repeat_Ci_14.set_in_repsig(repsiggen_i_15.out_repsig())
        fiberlookup_Cj_13.set_in_ref(repeat_Ci_14.out_ref())
        fiberlookup_Ck_9.set_in_ref(fiberlookup_Cj_13.out_ref())
        repsiggen_j_11.set_istream(fiberlookup_Cj_13.out_crd())
        repeat_Bj_10.set_in_ref(fiberlookup_Bi_17.out_ref())
        repeat_Bj_10.set_in_repsig(repsiggen_j_11.out_repsig())
        fiberlookup_Bk_8.set_in_ref(repeat_Bj_10.out_ref())
        intersectk_7.set_in1(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd())
        intersectk_7.set_in2(fiberlookup_Ck_9.out_ref(), fiberlookup_Ck_9.out_crd())
        arrayvals_B_5.set_load(intersectk_7.out_ref1())
        arrayvals_C_6.set_load(intersectk_7.out_ref2())
        mul_4.set_in1(arrayvals_B_5.out_val())
        mul_4.set_in2(arrayvals_C_6.out_val())
        crddrop_19.set_outer_crd(fiberlookup_Cj_13.out_crd())
        crddrop_19.set_inner_crd(mul_4.out_val())
        reduce_3.set_in_val(crddrop_19.out_crd_inner())
        crddrop_18.set_outer_crd(fiberlookup_Bi_17.out_crd())
        crddrop_18.set_inner_crd(crddrop_19.out_crd_outer())
        fiberwrite_Xvals_0.set_input(reduce_3.out_val())
        fiberwrite_X1_1.set_input(crddrop_18.out_crd_inner())
        fiberwrite_X0_2.set_input(crddrop_18.out_crd_outer())
        fiberlookup_Bi_17.update()

        repsiggen_i_15.update()
        repeat_Ci_14.update()
        fiberlookup_Cj_13.update()
        fiberlookup_Ck_9.update()
        repsiggen_j_11.update()
        repeat_Bj_10.update()
        fiberlookup_Bk_8.update()
        intersectk_7.update()
        arrayvals_B_5.update()
        arrayvals_C_6.update()
        mul_4.update()
        crddrop_19.update()
        reduce_3.update()
        crddrop_18.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X1_1.update()
        fiberwrite_X0_2.update()

        done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print("out segs", out_segs)
    print("out crds", out_crds)
    print("out_vals", out_vals)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberlookup_Bi_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ci_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_14" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_13" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_10" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_8" + "_" + k] = sample_dict[k]

    sample_dict = intersectk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "_" + k] = sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_9" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_18" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_19" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_matmul(ssname, debug_sim, cast, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
