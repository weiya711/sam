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
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2, SpAcc2
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
def test_tensor4_mult2_ijlmk(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor4_mult2_ijlmk")
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

    V_dirname = os.path.join(formatted_dir, frosttname, "tensor4_mult2_ijlmk")
    V_shape_filename = os.path.join(V_dirname, "tensor_V_mode_shape")
    V_shape = read_inputs(V_shape_filename)

    V0_seg_filename = os.path.join(V_dirname, "tensor_V_mode_0_seg")
    V_seg0 = read_inputs(V0_seg_filename)
    V0_crd_filename = os.path.join(V_dirname, "tensor_V_mode_0_crd")
    V_crd0 = read_inputs(V0_crd_filename)

    V1_seg_filename = os.path.join(V_dirname, "tensor_V_mode_1_seg")
    V_seg1 = read_inputs(V1_seg_filename)
    V1_crd_filename = os.path.join(V_dirname, "tensor_V_mode_1_crd")
    V_crd1 = read_inputs(V1_crd_filename)

    V2_seg_filename = os.path.join(V_dirname, "tensor_V_mode_2_seg")
    V_seg2 = read_inputs(V2_seg_filename)
    V2_crd_filename = os.path.join(V_dirname, "tensor_V_mode_2_crd")
    V_crd2 = read_inputs(V2_crd_filename)

    V3_seg_filename = os.path.join(V_dirname, "tensor_V_mode_3_seg")
    V_seg3 = read_inputs(V3_seg_filename)
    V3_crd_filename = os.path.join(V_dirname, "tensor_V_mode_3_crd")
    V_crd3 = read_inputs(V3_crd_filename)

    V_vals_filename = os.path.join(V_dirname, "tensor_V_mode_vals")
    V_vals = read_inputs(V_vals_filename, float)


    fiberlookup_Bi_27 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vi_28 = CompressedCrdRdScan(crd_arr=V_crd0, seg_arr=V_seg0, debug=debug_sim, statistics=report_stats)
    intersecti_26 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_24 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vj_25 = CompressedCrdRdScan(crd_arr=V_crd2, seg_arr=V_seg2, debug=debug_sim, statistics=report_stats)
    intersectj_23 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl_21 = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vl_22 = CompressedCrdRdScan(crd_arr=V_crd1, seg_arr=V_seg1, debug=debug_sim, statistics=report_stats)
    intersectl_20 = Intersect2(debug=debug_sim, statistics=report_stats)
    crddrop_11 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberlookup_Vm_19 = CompressedCrdRdScan(crd_arr=V_crd3, seg_arr=V_seg3, debug=debug_sim, statistics=report_stats)
    crddrop_10 = CrdDrop(debug=debug_sim, statistics=report_stats)
    repsiggen_m_17 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_3 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    repeat_Bm_16 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_15 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    arrayvals_B_8 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    repsiggen_k_13 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Vk_12 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_V_9 = Array(init_arr=V_vals, debug=debug_sim, statistics=report_stats)
    mul_7 = Multiply2(debug=debug_sim, statistics=report_stats)
    reduce_6 = Reduce(debug=debug_sim, statistics=report_stats)
    spaccumulator2_5 = SpAcc2(val_stkn=True, debug=debug_sim, statistics=report_stats)
    # spaccumulator2_5_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats)
    # spaccumulator2_5_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats)
    # spaccumulator2_5_drop_val = StknDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * V_shape[3] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] * V_shape[3] + 1, size=B_shape[0] * B_shape[1] * V_shape[3] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X3_2 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * V_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_V = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_27.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_V) > 0:
            fiberlookup_Vi_28.set_in_ref(in_ref_V.pop(0))
        intersecti_26.set_in1(fiberlookup_Bi_27.out_ref(), fiberlookup_Bi_27.out_crd())
        intersecti_26.set_in2(fiberlookup_Vi_28.out_ref(), fiberlookup_Vi_28.out_crd())
        fiberlookup_Bj_24.set_in_ref(intersecti_26.out_ref1())
        fiberlookup_Vj_25.set_in_ref(intersecti_26.out_ref2())
        intersectj_23.set_in1(fiberlookup_Bj_24.out_ref(), fiberlookup_Bj_24.out_crd())
        intersectj_23.set_in2(fiberlookup_Vj_25.out_ref(), fiberlookup_Vj_25.out_crd())
        fiberlookup_Bl_21.set_in_ref(intersectj_23.out_ref1())
        fiberlookup_Vl_22.set_in_ref(intersectj_23.out_ref2())
        intersectl_20.set_in1(fiberlookup_Bl_21.out_ref(), fiberlookup_Bl_21.out_crd())
        intersectl_20.set_in2(fiberlookup_Vl_22.out_ref(), fiberlookup_Vl_22.out_crd())
        crddrop_11.set_outer_crd(intersectj_23.out_crd())
        crddrop_11.set_inner_crd(intersectl_20.out_crd())
        fiberlookup_Vm_19.set_in_ref(intersectl_20.out_ref2())
        repsiggen_m_17.set_istream(fiberlookup_Vm_19.out_crd())
        repeat_Bm_16.set_in_ref(intersectl_20.out_ref1())
        repeat_Bm_16.set_in_repsig(repsiggen_m_17.out_repsig())
        crddrop_10.set_outer_crd(intersecti_26.out_crd())
        crddrop_10.set_inner_crd(crddrop_11.out_crd_outer())
        fiberwrite_X0_4.set_input(crddrop_10.out_crd_outer())
        fiberwrite_X2_3.set_input(crddrop_10.out_crd_inner())
        fiberlookup_Bk_15.set_in_ref(repeat_Bm_16.out_ref())
        arrayvals_B_8.set_load(fiberlookup_Bk_15.out_ref())
        repsiggen_k_13.set_istream(fiberlookup_Bk_15.out_crd())
        repeat_Vk_12.set_in_repsig(repsiggen_k_13.out_repsig())
        repeat_Vk_12.set_in_ref(fiberlookup_Vm_19.out_ref())
        arrayvals_V_9.set_load(repeat_Vk_12.out_ref())
        mul_7.set_in1(arrayvals_V_9.out_val())
        mul_7.set_in2(arrayvals_B_8.out_val())
        spaccumulator2_5.set_in_crd0(fiberlookup_Bk_15.out_crd())
        spaccumulator2_5.set_in_crd1(fiberlookup_Vm_19.out_crd())
        spaccumulator2_5.set_in_crd2(crddrop_11.out_crd_inner())
        spaccumulator2_5.set_val(mul_7.out_val())
        fiberwrite_Xvals_0.set_input(spaccumulator2_5.out_val())
        fiberwrite_X1_1.set_input(spaccumulator2_5.out_crd0())
        fiberwrite_X3_2.set_input(spaccumulator2_5.out_crd1())
        fiberlookup_Bi_27.update()

        fiberlookup_Vi_28.update()

        intersecti_26.update()
        fiberlookup_Bj_24.update()
        fiberlookup_Vj_25.update()
        intersectj_23.update()
        fiberlookup_Bl_21.update()
        fiberlookup_Vl_22.update()
        intersectl_20.update()
        crddrop_11.update()
        fiberlookup_Vm_19.update()
        repsiggen_m_17.update()
        repeat_Bm_16.update()
        crddrop_10.update()
        fiberwrite_X0_4.update()
        fiberwrite_X2_3.update()
        fiberlookup_Bk_15.update()
        arrayvals_B_8.update()
        repsiggen_k_13.update()
        repeat_Vk_12.update()
        arrayvals_V_9.update()
        mul_7.update()
        reduce_6.update()
        spaccumulator2_5.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X1_1.update()
        fiberwrite_X3_2.update()

        done = fiberwrite_X0_4.out_done() and fiberwrite_X2_3.out_done() and fiberwrite_X3_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_4.autosize()
    fiberwrite_X2_3.autosize()
    fiberwrite_X3_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_4.get_arr(), fiberwrite_X2_3.get_arr(), fiberwrite_X3_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_4.get_seg_arr(), fiberwrite_X2_3.get_seg_arr(), fiberwrite_X3_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print(out_vals)
    print("# Cycles:", time_cnt)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_V_shape"] = V_shape
    sample_dict = fiberlookup_Bi_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_27" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_26" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_10" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_24" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_23" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_11" + "_" + k] = sample_dict[k]

    sample_dict = spaccumulator2_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator2_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_21" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_20" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bm_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bm_16" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_15" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Vk_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Vk_12" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_V_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_V_9" + "_" + k] = sample_dict[k]

    sample_dict = reduce_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_6" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_8" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vm_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vm_19" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vl_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vl_22" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vj_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vj_25" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vi_28.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vi_28" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_mult2_ijlmk(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "ssss0231")
    samBench(bench, extra_info)
