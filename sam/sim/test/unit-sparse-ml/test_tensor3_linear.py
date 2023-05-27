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
def test_tensor3_linear(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor3_linear")
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

    C_dirname = os.path.join(formatted_dir, frosttname, "tensor3_linear")
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

    D_dirname = os.path.join(formatted_dir, frosttname, "tensor3_linear")
    D_shape_filename = os.path.join(D_dirname, "tensor_D_mode_shape")
    D_shape = read_inputs(D_shape_filename)

    D_vals_filename = os.path.join(D_dirname, "tensor_D_mode_vals")
    D_vals = read_inputs(D_vals_filename, float)


    fiberlookup_Ci_32 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck_26 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_i_30 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_2 = CompressWrScan(seg_size=C_shape[0] + 1, size=C_shape[0] * C_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_k_24 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bi_27 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Di_28 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Bk_21 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Dk_22 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_19 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Dj_20 = UncompressCrdRdScan(dim=D_shape[0], debug=debug_sim, statistics=report_stats)
    unionj_18 = Union2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_1 = CompressWrScan(seg_size=C_shape[0] * C_shape[2] + 1, size=C_shape[0] * C_shape[2] * B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Cj_15 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Cl_14 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    intersectl_12 = Intersect2(debug=debug_sim, statistics=report_stats)
    repsiggen_l_11 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    repeat_Dl_10 = Repeat(debug=debug_sim, statistics=report_stats)
    mul_6 = Multiply2(debug=debug_sim, statistics=report_stats)
    arrayvals_D_9 = Array(init_arr=D_vals, debug=debug_sim, statistics=report_stats)
    add_5 = Add2(debug=debug_sim, statistics=report_stats)
    reduce_4 = Reduce(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[0] * C_shape[2] * B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_C = [0, 'D']
    in_ref_B = [0, 'D']
    in_ref_D = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_32.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ck_26.set_in_ref(fiberlookup_Ci_32.out_ref())
        fiberwrite_X0_3.set_input(fiberlookup_Ci_32.out_crd())
        repsiggen_i_30.set_istream(fiberlookup_Ci_32.out_crd())
        if len(in_ref_B) > 0:
            repeat_Bi_27.set_in_ref(in_ref_B.pop(0))
        repeat_Bi_27.set_in_repsig(repsiggen_i_30.out_repsig())
        if len(in_ref_D) > 0:
            repeat_Di_28.set_in_ref(in_ref_D.pop(0))
        repeat_Di_28.set_in_repsig(repsiggen_i_30.out_repsig())
        fiberwrite_X2_2.set_input(fiberlookup_Ck_26.out_crd())
        repsiggen_k_24.set_istream(fiberlookup_Ck_26.out_crd())
        repeat_Bk_21.set_in_ref(repeat_Bi_27.out_ref())
        repeat_Bk_21.set_in_repsig(repsiggen_k_24.out_repsig())
        repeat_Dk_22.set_in_ref(repeat_Di_28.out_ref())
        repeat_Dk_22.set_in_repsig(repsiggen_k_24.out_repsig())
        fiberlookup_Bj_19.set_in_ref(repeat_Bk_21.out_ref())
        fiberlookup_Dj_20.set_in_ref(repeat_Dk_22.out_ref())
        unionj_18.set_in1(fiberlookup_Bj_19.out_ref(), fiberlookup_Bj_19.out_crd())
        unionj_18.set_in2(fiberlookup_Dj_20.out_ref(), fiberlookup_Dj_20.out_crd())
        fiberlookup_Bl_13.set_in_ref(unionj_18.out_ref1())
        fiberwrite_X1_1.set_input(unionj_18.out_crd())
        repsiggen_j_16.set_istream(unionj_18.out_crd())
        repeat_Cj_15.set_in_repsig(repsiggen_j_16.out_repsig())
        repeat_Cj_15.set_in_ref(fiberlookup_Ck_26.out_ref())
        fiberlookup_Cl_14.set_in_ref(repeat_Cj_15.out_ref())
        intersectl_12.set_in1(fiberlookup_Cl_14.out_ref(), fiberlookup_Cl_14.out_crd())
        intersectl_12.set_in2(fiberlookup_Bl_13.out_ref(), fiberlookup_Bl_13.out_crd())
        repsiggen_l_11.set_istream(intersectl_12.out_crd())
        arrayvals_B_7.set_load(intersectl_12.out_ref2())
        arrayvals_C_8.set_load(intersectl_12.out_ref1())
        repeat_Dl_10.set_in_ref(unionj_18.out_ref2())
        repeat_Dl_10.set_in_repsig(repsiggen_l_11.out_repsig())
        arrayvals_D_9.set_load(repeat_Dl_10.out_ref())
        mul_6.set_in1(arrayvals_B_7.out_val())
        mul_6.set_in2(arrayvals_C_8.out_val())
        add_5.set_in1(arrayvals_D_9.out_val())
        add_5.set_in2(mul_6.out_val())
        reduce_4.set_in_val(add_5.out_val())
        fiberwrite_Xvals_0.set_input(reduce_4.out_val())
        fiberlookup_Ci_32.update()

        fiberlookup_Ck_26.update()
        fiberwrite_X0_3.update()
        repsiggen_i_30.update()
        repeat_Bi_27.update()
        repeat_Di_28.update()
        fiberwrite_X2_2.update()
        repsiggen_k_24.update()
        repeat_Bk_21.update()
        repeat_Dk_22.update()
        fiberlookup_Bj_19.update()
        fiberlookup_Dj_20.update()
        unionj_18.update()
        fiberlookup_Bl_13.update()
        fiberwrite_X1_1.update()
        repsiggen_j_16.update()
        repeat_Cj_15.update()
        fiberlookup_Cl_14.update()
        intersectl_12.update()
        repsiggen_l_11.update()
        arrayvals_B_7.update()
        arrayvals_C_8.update()
        repeat_Dl_10.update()
        arrayvals_D_9.update()
        mul_6.update()
        add_5.update()
        reduce_4.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_3.out_done() and fiberwrite_X2_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_3.autosize()
    fiberwrite_X2_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X2_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X2_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print(time_cnt)
    pytest.set_trace()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_D_shape"] = D_shape
    sample_dict = fiberlookup_Ci_32.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_32" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bi_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bi_27" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bk_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bk_21" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_19" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Cj_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cj_15" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cl_14" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_12" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Dl_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Dl_10" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_D_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_D_9" + "_" + k] = sample_dict[k]

    sample_dict = reduce_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_7" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_8" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_13" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Di_28.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Di_28" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Dk_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Dk_22" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Dj_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Dj_20" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_26" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_2" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_linear(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "sss021")
    samBench(bench, extra_info)
