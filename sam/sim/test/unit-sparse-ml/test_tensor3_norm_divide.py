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
def test_tensor3_norm_divide(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor3_norm_divide")
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

    C_dirname = os.path.join(formatted_dir, frosttname, "tensor3_norm_divide")
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


    fiberlookup_Bi_16 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ci_17 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    intersecti_15 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj_14 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
    intersectj_12 = Intersect2(debug=debug_sim, statistics=report_stats)
    crddrop_7 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_11 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_k_9 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ck_8 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    mul_4 = Divide2(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    div0_in = []
    div1_in = []
    div_out = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_16.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_C) > 0:
            fiberlookup_Ci_17.set_in_ref(in_ref_C.pop(0))
        intersecti_15.set_in1(fiberlookup_Bi_16.out_ref(), fiberlookup_Bi_16.out_crd())
        intersecti_15.set_in2(fiberlookup_Ci_17.out_ref(), fiberlookup_Ci_17.out_crd())
        fiberlookup_Bj_13.set_in_ref(intersecti_15.out_ref1())
        fiberlookup_Cj_14.set_in_ref(intersecti_15.out_ref2())
        intersectj_12.set_in1(fiberlookup_Bj_13.out_ref(), fiberlookup_Bj_13.out_crd())
        intersectj_12.set_in2(fiberlookup_Cj_14.out_ref(), fiberlookup_Cj_14.out_crd())
        crddrop_7.set_outer_crd(intersecti_15.out_crd())
        crddrop_7.set_inner_crd(intersectj_12.out_crd())
        fiberlookup_Bk_11.set_in_ref(intersectj_12.out_ref1())
        arrayvals_B_5.set_load(fiberlookup_Bk_11.out_ref())
        fiberwrite_X2_1.set_input(fiberlookup_Bk_11.out_crd())
        repsiggen_k_9.set_istream(fiberlookup_Bk_11.out_crd())
        repeat_Ck_8.set_in_ref(intersectj_12.out_ref2())
        repeat_Ck_8.set_in_repsig(repsiggen_k_9.out_repsig())
        arrayvals_C_6.set_load(repeat_Ck_8.out_ref())
        mul_4.set_in1(arrayvals_C_6.out_val())
        mul_4.set_in2(arrayvals_B_5.out_val())

        div0_in.append(arrayvals_C_6.out_val())
        div1_in.append(arrayvals_B_5.out_val())
        fiberwrite_Xvals_0.set_input(mul_4.out_val())
        fiberwrite_X0_3.set_input(crddrop_7.out_crd_outer())
        fiberwrite_X1_2.set_input(crddrop_7.out_crd_inner())
        fiberlookup_Bi_16.update()

        fiberlookup_Ci_17.update()

        intersecti_15.update()
        fiberlookup_Bj_13.update()
        fiberlookup_Cj_14.update()
        intersectj_12.update()
        crddrop_7.update()
        fiberlookup_Bk_11.update()
        arrayvals_B_5.update()
        fiberwrite_X2_1.update()
        repsiggen_k_9.update()
        repeat_Ck_8.update()
        arrayvals_C_6.update()
        mul_4.update()
        div_out.append(mul_4.out_val())

        print("div0_in", remove_emptystr(div0_in))
        print("div1_in", remove_emptystr(div1_in))
        print("div_out", remove_emptystr(div_out))
        fiberwrite_Xvals_0.update()
        fiberwrite_X0_3.update()
        fiberwrite_X1_2.update()

        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print(remove_emptystr(out_vals))

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberlookup_Bi_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_16" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_15" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_7" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_13" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_12" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ck_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ck_8" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_14" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_17" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_norm_divide(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
