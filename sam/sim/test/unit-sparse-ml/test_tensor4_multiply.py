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
def test_tensor4_multiply(samBench, frosttname, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "orig", "ssss0213")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "B2_seg.txt")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B2_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B2_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd2 = read_inputs(B2_crd_filename)

    B3_seg_filename = os.path.join(B_dirname, "B3_seg.txt")
    B_seg3 = read_inputs(B3_seg_filename)
    B3_crd_filename = os.path.join(B_dirname, "B3_crd.txt")
    B_crd3 = read_inputs(B3_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, frosttname, "other", "ssss0213")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "C2_seg.txt")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "C2_crd.txt")
    C_crd1 = read_inputs(C1_crd_filename)

    C2_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
    C_seg2 = read_inputs(C2_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
    C_crd2 = read_inputs(C2_crd_filename)

    C3_seg_filename = os.path.join(C_dirname, "C3_seg.txt")
    C_seg3 = read_inputs(C3_seg_filename)
    C3_crd_filename = os.path.join(C_dirname, "C3_crd.txt")
    C_crd3 = read_inputs(C3_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_25 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ci_26 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    intersecti_24 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_22 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj_23 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats)
    intersectj_21 = Intersect2(debug=debug_sim, statistics=report_stats)
    crddrop_9 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_20 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_3 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_2 = CompressWrScan(seg_size=B_shape[0] * B_shape[2] + 1, size=B_shape[0] * B_shape[2] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_k_18 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ck_17 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Cl_16 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cm_12 = CompressedCrdRdScan(crd_arr=C_crd3, seg_arr=C_seg3, debug=debug_sim, statistics=report_stats)
    fiberwrite_X3_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[2] * B_shape[1] + 1, size=B_shape[0] * B_shape[2] * B_shape[1] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_l_14 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bl_13 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bm_11 = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats)
    intersectm_10 = Intersect2(debug=debug_sim, statistics=report_stats)
    arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    mul_6 = Multiply2(debug=debug_sim, statistics=report_stats)
    reduce_5 = Reduce(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[2] * B_shape[1] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0
    test_stream = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_25.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_C) > 0:
            fiberlookup_Ci_26.set_in_ref(in_ref_C.pop(0))
        intersecti_24.set_in1(fiberlookup_Bi_25.out_ref(), fiberlookup_Bi_25.out_crd())
        intersecti_24.set_in2(fiberlookup_Ci_26.out_ref(), fiberlookup_Ci_26.out_crd())
        fiberlookup_Bj_22.set_in_ref(intersecti_24.out_ref1())
        fiberlookup_Cj_23.set_in_ref(intersecti_24.out_ref2())
        intersectj_21.set_in1(fiberlookup_Bj_22.out_ref(), fiberlookup_Bj_22.out_crd())
        intersectj_21.set_in2(fiberlookup_Cj_23.out_ref(), fiberlookup_Cj_23.out_crd())
        crddrop_9.set_outer_crd(intersecti_24.out_crd())
        crddrop_9.set_inner_crd(intersectj_21.out_crd())
        # pytest.set_trace()
        test_stream.append(intersectj_21.out_ref1())
        fiberlookup_Bk_20.set_in_ref(intersectj_21.out_ref1())
        fiberwrite_X2_2.set_input(fiberlookup_Bk_20.out_crd())
        repsiggen_k_18.set_istream(fiberlookup_Bk_20.out_crd())
        repeat_Ck_17.set_in_ref(intersectj_21.out_ref2())
        repeat_Ck_17.set_in_repsig(repsiggen_k_18.out_repsig())
        fiberlookup_Cl_16.set_in_ref(repeat_Ck_17.out_ref())
        fiberlookup_Cm_12.set_in_ref(fiberlookup_Cl_16.out_ref())
        fiberwrite_X3_1.set_input(fiberlookup_Cl_16.out_crd())
        repsiggen_l_14.set_istream(fiberlookup_Cl_16.out_crd())
        repeat_Bl_13.set_in_ref(fiberlookup_Bk_20.out_ref())
        repeat_Bl_13.set_in_repsig(repsiggen_l_14.out_repsig())
        fiberlookup_Bm_11.set_in_ref(repeat_Bl_13.out_ref())
        intersectm_10.set_in1(fiberlookup_Bm_11.out_ref(), fiberlookup_Bm_11.out_crd())
        intersectm_10.set_in2(fiberlookup_Cm_12.out_ref(), fiberlookup_Cm_12.out_crd())
        arrayvals_B_7.set_load(intersectm_10.out_ref1())
        arrayvals_C_8.set_load(intersectm_10.out_ref2())
        mul_6.set_in1(arrayvals_B_7.out_val())
        mul_6.set_in2(arrayvals_C_8.out_val())
        reduce_5.set_in_val(mul_6.out_val())
        fiberwrite_Xvals_0.set_input(reduce_5.out_val())
        fiberwrite_X0_4.set_input(crddrop_9.out_crd_outer())
        fiberwrite_X1_3.set_input(crddrop_9.out_crd_inner())
        fiberlookup_Bi_25.update()

        fiberlookup_Ci_26.update()

        intersecti_24.update()
        fiberlookup_Bj_22.update()
        fiberlookup_Cj_23.update()
        intersectj_21.update()
        crddrop_9.update()
        fiberlookup_Bk_20.update()
        fiberwrite_X2_2.update()
        repsiggen_k_18.update()
        repeat_Ck_17.update()
        fiberlookup_Cl_16.update()
        fiberlookup_Cm_12.update()
        fiberwrite_X3_1.update()
        repsiggen_l_14.update()
        repeat_Bl_13.update()
        fiberlookup_Bm_11.update()
        intersectm_10.update()
        arrayvals_B_7.update()
        arrayvals_C_8.update()
        mul_6.update()
        reduce_5.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X0_4.update()
        fiberwrite_X1_3.update()

        done = fiberwrite_X0_4.out_done() and fiberwrite_X1_3.out_done() and fiberwrite_X2_2.out_done() and fiberwrite_X3_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_4.autosize()
    fiberwrite_X1_3.autosize()
    fiberwrite_X2_2.autosize()
    fiberwrite_X3_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_4.get_arr(), fiberwrite_X1_3.get_arr(), fiberwrite_X2_2.get_arr(), fiberwrite_X3_1.get_arr()]
    out_segs = [fiberwrite_X0_4.get_seg_arr(), fiberwrite_X1_3.get_seg_arr(), fiberwrite_X2_2.get_seg_arr(), fiberwrite_X3_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print(out_crds)
    print(out_segs)
    print(out_vals)

    def bench():
        time.sleep(0.01)

    print(test_stream)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberlookup_Bi_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_25" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_24" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_9" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_22" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_21" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_20" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ck_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ck_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cl_16" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bl_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bl_13" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bm_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bm_11" + "_" + k] = sample_dict[k]

    sample_dict = intersectm_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectm_10" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_7" + "_" + k] = sample_dict[k]

    sample_dict = reduce_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_8" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cm_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cm_12" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_23" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_26" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        # check_gold_tensor4_multiply(frosttname, debug_sim, out_crds, out_segs, out_vals, "ssss0123")
        check_gold_tensor4_mulQK(frosttname, debug_sim, out_crds, out_segs, out_vals, "ssss0123")
    samBench(bench, extra_info)
