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
def test_tensor3_linear(samBench, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, , "orig", "ss01")
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

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, , "other", "sss012")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
    C_crd1 = read_inputs(C1_crd_filename)

    C2_seg_filename = os.path.join(C_dirname, "C2_seg.txt")
    C_seg2 = read_inputs(C2_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "C2_crd.txt")
    C_crd2 = read_inputs(C2_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Ci_27 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj_23 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    repsiggen_i_25 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck_19 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats)
    crdhold_9 = CrdHold(debug=debug_sim, statistics=report_stats)
    repsiggen_j_21 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bi_24 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Bj_20 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_18 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    intersectk_17 = Intersect2(debug=debug_sim, statistics=report_stats)
    crdhold_8 = CrdHold(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl_16 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    crdhold_6 = CrdHold(debug=debug_sim, statistics=report_stats)
    arrayvals_B_11 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    crdhold_7 = CrdHold(debug=debug_sim, statistics=report_stats)
    repsiggen_l_14 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    crdhold_5 = CrdHold(debug=debug_sim, statistics=report_stats)
    repeat_Cl_13 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_C_12 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    mul_10 = Multiply2(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4 = SparseAccumulator1(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4_drop_val = StknDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[0] * C_shape[1] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=C_shape[0] * C_shape[1] + 1, size=C_shape[0] * C_shape[1] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=C_shape[0] + 1, size=C_shape[0] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_C = [0, 'D']
    in_ref_B = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_27.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Cj_23.set_in_ref(fiberlookup_Ci_27.out_ref())
        crdhold_9.set_outer_crd(fiberlookup_Ci_27.out_crd())
        crdhold_9.set_inner_crd(fiberlookup_Cj_23.out_crd())
        repsiggen_i_25.set_istream(fiberlookup_Ci_27.out_crd())
        if len(in_ref_B) > 0:
            repeat_Bi_24.set_in_ref(in_ref_B.pop(0))
        repeat_Bi_24.set_in_repsig(repsiggen_i_25.out_repsig())
        fiberlookup_Ck_19.set_in_ref(fiberlookup_Cj_23.out_ref())
        repsiggen_j_21.set_istream(fiberlookup_Cj_23.out_crd())
        repeat_Bj_20.set_in_ref(repeat_Bi_24.out_ref())
        repeat_Bj_20.set_in_repsig(repsiggen_j_21.out_repsig())
        fiberlookup_Bk_18.set_in_ref(repeat_Bj_20.out_ref())
        intersectk_17.set_in1(fiberlookup_Bk_18.out_ref(), fiberlookup_Bk_18.out_crd())
        intersectk_17.set_in2(fiberlookup_Ck_19.out_ref(), fiberlookup_Ck_19.out_crd())
        crdhold_8.set_outer_crd(crdhold_9.out_crd_outer())
        crdhold_8.set_inner_crd(intersectk_17.out_crd())
        fiberlookup_Bl_16.set_in_ref(intersectk_17.out_ref1())
        arrayvals_B_11.set_load(fiberlookup_Bl_16.out_ref())
        crdhold_7.set_outer_crd(crdhold_8.out_crd_outer())
        crdhold_7.set_inner_crd(fiberlookup_Bl_16.out_crd())
        repsiggen_l_14.set_istream(fiberlookup_Bl_16.out_crd())
        repeat_Cl_13.set_in_ref(intersectk_17.out_ref2())
        repeat_Cl_13.set_in_repsig(repsiggen_l_14.out_repsig())
        arrayvals_C_12.set_load(repeat_Cl_13.out_ref())
        mul_10.set_in1(arrayvals_C_12.out_val())
        mul_10.set_in2(arrayvals_B_11.out_val())
        crdhold_6.set_outer_crd(crdhold_9.out_crd_inner())
        crdhold_6.set_inner_crd(crdhold_8.out_crd_inner())
        crdhold_5.set_inner_crd(crdhold_7.out_crd_inner())
        crdhold_5.set_outer_crd(crdhold_6.out_crd_outer())
        spaccumulator1_4_drop_crd_outer.set_in_stream(crdhold_7.out_crd_outer())
        spaccumulator1_4_drop_crd_outer.set_in_stream(crdhold_5.out_crd_outer())
        spaccumulator1_4_drop_crd_inner.set_in_stream(crdhold_5.out_crd_inner())
        spaccumulator1_4_drop_crd_inner.set_in_stream(crdhold_6.out_crd_inner())
        spaccumulator1_4_drop_val.set_in_stream(mul_10.out_val())
        spaccumulator1_4.set_crd_outer(spaccumulator1_4_drop_crd_outer.out_val())
        spaccumulator1_4.set_crd_outer(spaccumulator1_4_drop_crd_outer.out_val())
        spaccumulator1_4.set_crd_inner(spaccumulator1_4_drop_crd_inner.out_val())
        spaccumulator1_4.set_crd_inner(spaccumulator1_4_drop_crd_inner.out_val())
        spaccumulator1_4.set_val(spaccumulator1_4_drop_val.out_val())
        fiberwrite_Xvals_0.set_input(spaccumulator1_4.out_val())
        fiberwrite_X2_1.set_input(spaccumulator1_4.out_crd_inner())
        fiberwrite_X1_2.set_input(spaccumulator1_4.out_crd_outer())
        fiberwrite_X0_3.set_input(spaccumulator1_4.out_crd_outer())
        fiberlookup_Ci_27.update()

        fiberlookup_Cj_23.update()
        crdhold_9.update()
        repsiggen_i_25.update()
        repeat_Bi_24.update()
        fiberlookup_Ck_19.update()
        repsiggen_j_21.update()
        repeat_Bj_20.update()
        fiberlookup_Bk_18.update()
        intersectk_17.update()
        crdhold_8.update()
        fiberlookup_Bl_16.update()
        arrayvals_B_11.update()
        crdhold_7.update()
        repsiggen_l_14.update()
        repeat_Cl_13.update()
        arrayvals_C_12.update()
        mul_10.update()
        crdhold_6.update()
        crdhold_5.update()
        spaccumulator1_4.update()
        spaccumulator1_4.update()
        spaccumulator1_4.update()
        spaccumulator1_4.update()
        spaccumulator1_4.update()
        spaccumulator1_4.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X2_1.update()
        fiberwrite_X1_2.update()
        fiberwrite_X0_3.update()

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
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberlookup_Ci_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_27" + "_" + k] = sample_dict[k]

    sample_dict = spaccumulator1_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bi_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bi_24" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bj_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_20" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_18" + "_" + k] = sample_dict[k]

    sample_dict = intersectk_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_16" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Cl_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cl_13" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_12" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_23" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_19" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_linear(, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
