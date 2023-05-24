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
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor4_softmax_multiply2(samBench, frosttname, cast, check_gold, debug_sim, backpressure, depth, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor4_softmax_multiply2")
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

    B3_seg_filename = os.path.join(B_dirname, "tensor_B_mode_3_seg")
    B_seg3 = read_inputs(B3_seg_filename)
    B3_crd_filename = os.path.join(B_dirname, "tensor_B_mode_3_crd")
    B_crd3 = read_inputs(B3_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, frosttname, "tensor4_softmax_multiply2")
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

    C3_seg_filename = os.path.join(C_dirname, "tensor_C_mode_3_seg")
    C_seg3 = read_inputs(C3_seg_filename)
    C3_crd_filename = os.path.join(C_dirname, "tensor_C_mode_3_crd")
    C_crd3 = read_inputs(C3_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)


    fiberlookup_Bi_34 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_35 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti_33 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_31 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cj_32 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj_30 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crddrop_18 = CrdDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_29 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_14 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bl_24 = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_k_27 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_13 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Ck_26 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_10 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cl_25 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectl_23 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_12 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cm_22 = CompressedCrdRdScan(crd_arr=C_crd3, seg_arr=C_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_9 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_17 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_11 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_m_20 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_7 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_8 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bm_19 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_6 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_16 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_15 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5 = SparseAccumulator1(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5_drop_val = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2] * C_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X3_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] * B_shape[2] + 1, size=B_shape[0] * B_shape[1] * B_shape[2] * C_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_3 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    crd0 = []
    crd1 = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_34.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_C) > 0:
            fiberlookup_Ci_35.set_in_ref(in_ref_C.pop(0))
        intersecti_33.set_in1(fiberlookup_Bi_34.out_ref(), fiberlookup_Bi_34.out_crd())
        intersecti_33.set_in2(fiberlookup_Ci_35.out_ref(), fiberlookup_Ci_35.out_crd())
        fiberlookup_Bj_31.set_in_ref(intersecti_33.out_ref1())
        fiberlookup_Cj_32.set_in_ref(intersecti_33.out_ref2())
        intersectj_30.set_in1(fiberlookup_Bj_31.out_ref(), fiberlookup_Bj_31.out_crd())
        intersectj_30.set_in2(fiberlookup_Cj_32.out_ref(), fiberlookup_Cj_32.out_crd())
        crddrop_18.set_outer_crd(intersecti_33.out_crd())
        crddrop_18.set_inner_crd(intersectj_30.out_crd())
        fiberlookup_Bk_29.set_in_ref(intersectj_30.out_ref1())
        fiberlookup_Bl_24.set_in_ref(fiberlookup_Bk_29.out_ref())
        repsiggen_k_27.set_istream(fiberlookup_Bk_29.out_crd())
        repeat_Ck_26.set_in_ref(intersectj_30.out_ref2())
        repeat_Ck_26.set_in_repsig(repsiggen_k_27.out_repsig())
        fiberlookup_Cl_25.set_in_ref(repeat_Ck_26.out_ref())
        intersectl_23.set_in1(fiberlookup_Cl_25.out_ref(), fiberlookup_Cl_25.out_crd())
        intersectl_23.set_in2(fiberlookup_Bl_24.out_ref(), fiberlookup_Bl_24.out_crd())
        fiberlookup_Cm_22.set_in_ref(intersectl_23.out_ref1())
        arrayvals_C_17.set_load(fiberlookup_Cm_22.out_ref())
        repsiggen_m_20.set_istream(fiberlookup_Cm_22.out_crd())
        repeat_Bm_19.set_in_ref(intersectl_23.out_ref2())
        repeat_Bm_19.set_in_repsig(repsiggen_m_20.out_repsig())
        crdhold_14.set_outer_crd(crddrop_18.out_crd_outer())
        crdhold_14.set_inner_crd(crddrop_18.out_crd_inner())
        crd0.append(crddrop_18.out_crd_inner())
        crd1.append(crddrop_18.out_crd_outer())

        print("outer:", remove_emptystr(crd1))
        print("inner:", remove_emptystr(crd0))

        crdhold_13.set_outer_crd(crdhold_14.out_crd_outer())
        crdhold_13.set_inner_crd(fiberlookup_Bk_29.out_crd())
        crdhold_10.set_outer_crd(crdhold_14.out_crd_inner())
        crdhold_10.set_inner_crd(crdhold_13.out_crd_inner())
        crdhold_12.set_outer_crd(crdhold_13.out_crd_outer())
        crdhold_12.set_inner_crd(intersectl_23.out_crd())
        crdhold_11.set_outer_crd(crdhold_12.out_crd_outer())
        crdhold_11.set_inner_crd(fiberlookup_Cm_22.out_crd())
        crdhold_9.set_inner_crd(crdhold_12.out_crd_inner())
        crdhold_9.set_outer_crd(crdhold_10.out_crd_outer())
        crdhold_8.set_inner_crd(crdhold_11.out_crd_inner())
        crdhold_8.set_outer_crd(crdhold_9.out_crd_outer())
        crdhold_7.set_inner_crd(crdhold_9.out_crd_inner())
        crdhold_7.set_outer_crd(crdhold_10.out_crd_inner())
        crdhold_6.set_inner_crd(crdhold_8.out_crd_inner())
        crdhold_6.set_outer_crd(crdhold_7.out_crd_outer())
        arrayvals_B_16.set_load(repeat_Bm_19.out_ref())
        mul_15.set_in1(arrayvals_B_16.out_val())
        mul_15.set_in2(arrayvals_C_17.out_val())
        spaccumulator1_5_drop_crd_outer.set_in_stream(crdhold_6.out_crd_outer())
        spaccumulator1_5_drop_crd_inner.set_in_stream(crdhold_6.out_crd_inner())
        spaccumulator1_5_drop_val.set_in_stream(mul_15.out_val())
        spaccumulator1_5.set_crd_outer(spaccumulator1_5_drop_crd_outer.out_val())
        spaccumulator1_5.set_crd_outer(spaccumulator1_5_drop_crd_outer.out_val())
        spaccumulator1_5.set_crd_outer(spaccumulator1_5_drop_crd_outer.out_val())
        spaccumulator1_5.set_crd_inner(spaccumulator1_5_drop_crd_inner.out_val())
        spaccumulator1_5.set_crd_inner(spaccumulator1_5_drop_crd_inner.out_val())
        spaccumulator1_5.set_val(spaccumulator1_5_drop_val.out_val())
        fiberwrite_Xvals_0.set_input(spaccumulator1_5.out_val())
        fiberwrite_X3_1.set_input(spaccumulator1_5.out_crd_inner())
        fiberwrite_X1_2.set_input(spaccumulator1_5.out_crd_outer())
        fiberwrite_X2_3.set_input(spaccumulator1_5.out_crd_outer())
        fiberwrite_X0_4.set_input(spaccumulator1_5.out_crd_outer())
        fiberlookup_Bi_34.update()

        fiberlookup_Ci_35.update()

        intersecti_33.update()
        fiberlookup_Bj_31.update()
        fiberlookup_Cj_32.update()
        intersectj_30.update()
        crddrop_18.update()
        fiberlookup_Bk_29.update()
        fiberlookup_Bl_24.update()
        repsiggen_k_27.update()
        repeat_Ck_26.update()
        fiberlookup_Cl_25.update()
        intersectl_23.update()
        fiberlookup_Cm_22.update()
        arrayvals_C_17.update()
        repsiggen_m_20.update()
        repeat_Bm_19.update()
        crdhold_14.update()
        crdhold_13.update()
        crdhold_10.update()
        crdhold_12.update()
        crdhold_11.update()
        crdhold_9.update()
        crdhold_8.update()
        crdhold_7.update()
        crdhold_6.update()
        arrayvals_B_16.update()
        mul_15.update()
        spaccumulator1_5_drop_crd_outer.update()
        spaccumulator1_5_drop_crd_outer.update()
        spaccumulator1_5_drop_crd_outer.update()
        spaccumulator1_5_drop_crd_inner.update()
        spaccumulator1_5_drop_crd_inner.update()
        spaccumulator1_5_drop_val.update()
        spaccumulator1_5.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X3_1.update()
        fiberwrite_X1_2.update()
        fiberwrite_X2_3.update()
        fiberwrite_X0_4.update()

        done = fiberwrite_X0_4.out_done() and fiberwrite_X2_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X3_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_4.autosize()
    fiberwrite_X2_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X3_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_4.get_arr(), fiberwrite_X2_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X3_1.get_arr()]
    out_segs = [fiberwrite_X0_4.get_seg_arr(), fiberwrite_X2_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X3_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print("segs:", out_segs)
    print("crds:", out_crds)
    print("vals:", out_vals)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberlookup_Bi_34.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_34" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_33.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_33" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_18" + "_" + k] = sample_dict[k]

    sample_dict = spaccumulator1_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_31.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_31" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_30.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_30" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_29.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_29" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ck_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ck_26" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cl_25" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_23" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bm_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bm_19" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_16" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cm_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cm_22" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_24" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_32.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_32" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_35.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_35" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_softmax_multiply2(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "ssss0213")
    samBench(bench, extra_info)
