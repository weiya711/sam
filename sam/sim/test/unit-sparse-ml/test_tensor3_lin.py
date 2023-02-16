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
def test_tensor3_lin(samBench, , cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, , "tensor3_lin")
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

    C_dirname = os.path.join(formatted_dir, , "tensor3_lin")
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

    D_dirname = os.path.join(formatted_dir, , "tensor3_lin")
    D_shape_filename = os.path.join(D_dirname, "tensor_D_mode_shape")
    D_shape = read_inputs(D_shape_filename)

    D0_seg_filename = os.path.join(D_dirname, "tensor_D_mode_0_seg")
    D_seg0 = read_inputs(D0_seg_filename)
    D0_crd_filename = os.path.join(D_dirname, "tensor_D_mode_0_crd")
    D_crd0 = read_inputs(D0_crd_filename)

    D_vals_filename = os.path.join(D_dirname, "tensor_D_mode_vals")
    D_vals = read_inputs(D_vals_filename, float)


    fiberlookup_Ci_38 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    repsiggen_i_36 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bi_33 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Di_34 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_31 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Dj_32 = CompressedCrdRdScan(crd_arr=D_crd0, seg_arr=D_seg0, debug=debug_sim, statistics=report_stats)
    unionj_30 = Union2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl_25 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    crdhold_9 = CrdHold(debug=debug_sim, statistics=report_stats)
    repsiggen_j_28 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Cj_27 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Cl_26 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    intersectl_24 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck_20 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats)
    crdhold_8 = CrdHold(debug=debug_sim, statistics=report_stats)
    repsiggen_l_22 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    arrayvals_C_13 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    repsiggen_k_18 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    crdhold_7 = CrdHold(debug=debug_sim, statistics=report_stats)
    crdhold_6 = CrdHold(debug=debug_sim, statistics=report_stats)
    repeat_Dl_21 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Bk_15 = Repeat(debug=debug_sim, statistics=report_stats)
    crdhold_5 = CrdHold(debug=debug_sim, statistics=report_stats)
    repeat_Dk_16 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_B_12 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_D_14 = Array(init_arr=D_vals, debug=debug_sim, statistics=report_stats)
    mul_11 = Multiply2(debug=debug_sim, statistics=report_stats)
    add_10 = Add2(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4 = SparseAccumulator1(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats)
    spaccumulator1_4_drop_val = StknDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[0] * B_shape[0] * C_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=C_shape[0] * B_shape[0] + 1, size=C_shape[0] * B_shape[0] * C_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=C_shape[0] + 1, size=C_shape[0] * B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_C = [0, 'D']
    in_ref_B = [0, 'D']
    in_ref_D = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_38.set_in_ref(in_ref_C.pop(0))
        repsiggen_i_36.set_istream(fiberlookup_Ci_38.out_crd())
        if len(in_ref_B) > 0:
            repeat_Bi_33.set_in_ref(in_ref_B.pop(0))
        repeat_Bi_33.set_in_repsig(repsiggen_i_36.out_repsig())
        if len(in_ref_D) > 0:
            repeat_Di_34.set_in_ref(in_ref_D.pop(0))
        repeat_Di_34.set_in_repsig(repsiggen_i_36.out_repsig())
        fiberlookup_Bj_31.set_in_ref(repeat_Bi_33.out_ref())
        fiberlookup_Dj_32.set_in_ref(repeat_Di_34.out_ref())
        unionj_30.set_in1(fiberlookup_Bj_31.out_ref(), fiberlookup_Bj_31.out_crd())
        unionj_30.set_in2(fiberlookup_Dj_32.out_ref(), fiberlookup_Dj_32.out_crd())
        crdhold_9.set_outer_crd(fiberlookup_Ci_38.out_crd())
        crdhold_9.set_inner_crd(unionj_30.out_crd())
        fiberlookup_Bl_25.set_in_ref(unionj_30.out_ref1())
        repsiggen_j_28.set_istream(unionj_30.out_crd())
        repeat_Cj_27.set_in_ref(fiberlookup_Ci_38.out_ref())
        repeat_Cj_27.set_in_repsig(repsiggen_j_28.out_repsig())
        fiberlookup_Cl_26.set_in_ref(repeat_Cj_27.out_ref())
        intersectl_24.set_in1(fiberlookup_Cl_26.out_ref(), fiberlookup_Cl_26.out_crd())
        intersectl_24.set_in2(fiberlookup_Bl_25.out_ref(), fiberlookup_Bl_25.out_crd())
        fiberlookup_Ck_20.set_in_ref(intersectl_24.out_ref1())
        crdhold_8.set_outer_crd(crdhold_9.out_crd_outer())
        crdhold_8.set_inner_crd(intersectl_24.out_crd())
        repsiggen_l_22.set_istream(intersectl_24.out_crd())
        repeat_Dl_21.set_in_ref(unionj_30.out_ref2())
        repeat_Dl_21.set_in_repsig(repsiggen_l_22.out_repsig())
        arrayvals_C_13.set_load(fiberlookup_Ck_20.out_ref())
        crdhold_7.set_outer_crd(crdhold_8.out_crd_outer())
        crdhold_7.set_inner_crd(fiberlookup_Ck_20.out_crd())
        repsiggen_k_18.set_istream(fiberlookup_Ck_20.out_crd())
        repeat_Bk_15.set_in_ref(intersectl_24.out_ref2())
        repeat_Bk_15.set_in_repsig(repsiggen_k_18.out_repsig())
        repeat_Dk_16.set_in_ref(repeat_Dl_21.out_ref())
        repeat_Dk_16.set_in_repsig(repsiggen_k_18.out_repsig())
        crdhold_6.set_outer_crd(crdhold_9.out_crd_inner())
        crdhold_6.set_inner_crd(crdhold_8.out_crd_inner())
        crdhold_5.set_inner_crd(crdhold_7.out_crd_inner())
        crdhold_5.set_outer_crd(crdhold_6.out_crd_outer())
        arrayvals_D_14.set_load(repeat_Dk_16.out_ref())
        arrayvals_B_12.set_load(repeat_Bk_15.out_ref())
        mul_11.set_in1(arrayvals_B_12.out_val())
        mul_11.set_in2(arrayvals_C_13.out_val())
        add_10.set_in1(arrayvals_D_14.out_val())
        add_10.set_in2(mul_11.out_val())
        spaccumulator1_4_drop_crd_outer.set_in_stream(crdhold_7.out_crd_outer())
        spaccumulator1_4_drop_crd_outer.set_in_stream(crdhold_5.out_crd_outer())
        spaccumulator1_4_drop_crd_inner.set_in_stream(crdhold_5.out_crd_inner())
        spaccumulator1_4_drop_crd_inner.set_in_stream(crdhold_6.out_crd_inner())
        spaccumulator1_4_drop_val.set_in_stream(add_10.out_val())
        spaccumulator1_4.set_crd_outer(spaccumulator1_4_drop_crd_outer.out_val())
        spaccumulator1_4.set_crd_outer(spaccumulator1_4_drop_crd_outer.out_val())
        spaccumulator1_4.set_crd_inner(spaccumulator1_4_drop_crd_inner.out_val())
        spaccumulator1_4.set_crd_inner(spaccumulator1_4_drop_crd_inner.out_val())
        spaccumulator1_4.set_val(spaccumulator1_4_drop_val.out_val())
        fiberwrite_Xvals_0.set_input(spaccumulator1_4.out_val())
        fiberwrite_X2_1.set_input(spaccumulator1_4.out_crd_inner())
        fiberwrite_X1_2.set_input(spaccumulator1_4.out_crd_outer())
        fiberwrite_X0_3.set_input(spaccumulator1_4.out_crd_outer())
        fiberlookup_Ci_38.update()

        repsiggen_i_36.update()
        repeat_Bi_33.update()
        repeat_Di_34.update()
        fiberlookup_Bj_31.update()
        fiberlookup_Dj_32.update()
        unionj_30.update()
        crdhold_9.update()
        fiberlookup_Bl_25.update()
        repsiggen_j_28.update()
        repeat_Cj_27.update()
        fiberlookup_Cl_26.update()
        intersectl_24.update()
        fiberlookup_Ck_20.update()
        crdhold_8.update()
        repsiggen_l_22.update()
        repeat_Dl_21.update()
        arrayvals_C_13.update()
        crdhold_7.update()
        repsiggen_k_18.update()
        repeat_Bk_15.update()
        repeat_Dk_16.update()
        crdhold_6.update()
        crdhold_5.update()
        arrayvals_D_14.update()
        arrayvals_B_12.update()
        mul_11.update()
        add_10.update()
        spaccumulator1_4_drop_crd_outer.update()
        spaccumulator1_4_drop_crd_outer.update()
        spaccumulator1_4_drop_crd_inner.update()
        spaccumulator1_4_drop_crd_inner.update()
        spaccumulator1_4_drop_val.update()
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
    extra_info["tensor_D_shape"] = D_shape
    sample_dict = fiberlookup_Ci_38.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_38" + "_" + k] = sample_dict[k]

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

    sample_dict = repeat_Bi_33.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bi_33" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_31.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_31" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Cj_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cj_27" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cl_26" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_24" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Dl_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Dl_21" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Dk_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Dk_16" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_D_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_D_14" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bk_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bk_15" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_12" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_20" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_13" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_25" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Di_34.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Di_34" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Dj_32.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Dj_32" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_lin(, debug_sim, cast, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
