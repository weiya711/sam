import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
from sam.sim.src.unary_alu import ScalarMult, Max, Exp
from sam.sim.src.crd_masker import LowerTriangular, RandomDropout, UpperTriangular
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce, MaxReduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2, SpAcc1New
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
def test_tensor4_mult2(samBench, frosttname, cast, check_gold, debug_sim, backpressure, depth, report_stats, fill=0):
    test_name = "tensor4_fused_mul_T4"
    B_dirname = os.path.join(formatted_dir, frosttname, test_name)
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

    V_dirname = os.path.join(formatted_dir, frosttname, test_name)
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


    fiberlookup_Bi_28 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vi_29 = CompressedCrdRdScan(crd_arr=V_crd0, seg_arr=V_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti_27 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bj_25 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vj_26 = CompressedCrdRdScan(crd_arr=V_crd2, seg_arr=V_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj_24 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bk_23 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bl_18 = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_k_21 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Vk_20 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vl_19 = CompressedCrdRdScan(crd_arr=V_crd1, seg_arr=V_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectl_17 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crddrop_12 = CrdDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vm_16 = CompressedCrdRdScan(crd_arr=V_crd3, seg_arr=V_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crddrop_11 = CrdDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_V_9 = Array(init_arr=V_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_m_14 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crddrop_10 = CrdDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bm_13 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_3 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_8 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_7 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_6 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5 = SpAcc1New(debug=debug_sim, val_stkn=True, statistics=report_stats, back_en=backpressure, depth=int(depth))
    # spaccumulator1_5_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    # spaccumulator1_5_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    # spaccumulator1_5_drop_val = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2] * V_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X3_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] * B_shape[2] + 1, size=B_shape[0] * B_shape[1] * B_shape[2] * V_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    in_ref_B = [0, 'D']
    in_ref_V = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_28.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_V) > 0:
            fiberlookup_Vi_29.set_in_ref(in_ref_V.pop(0))
        intersecti_27.set_in1(fiberlookup_Bi_28.out_ref(), fiberlookup_Bi_28.out_crd())
        intersecti_27.set_in2(fiberlookup_Vi_29.out_ref(), fiberlookup_Vi_29.out_crd())
        fiberlookup_Bj_25.set_in_ref(intersecti_27.out_ref1())
        fiberlookup_Vj_26.set_in_ref(intersecti_27.out_ref2())
        intersectj_24.set_in1(fiberlookup_Bj_25.out_ref(), fiberlookup_Bj_25.out_crd())
        intersectj_24.set_in2(fiberlookup_Vj_26.out_ref(), fiberlookup_Vj_26.out_crd())
        fiberlookup_Bk_23.set_in_ref(intersectj_24.out_ref1())
        fiberlookup_Bl_18.set_in_ref(fiberlookup_Bk_23.out_ref())
        repsiggen_k_21.set_istream(fiberlookup_Bk_23.out_crd())
        repeat_Vk_20.set_in_ref(intersectj_24.out_ref2())
        repeat_Vk_20.set_in_repsig(repsiggen_k_21.out_repsig())
        fiberlookup_Vl_19.set_in_ref(repeat_Vk_20.out_ref())
        intersectl_17.set_in1(fiberlookup_Vl_19.out_ref(), fiberlookup_Vl_19.out_crd())
        intersectl_17.set_in2(fiberlookup_Bl_18.out_ref(), fiberlookup_Bl_18.out_crd())
        crddrop_12.set_outer_crd(fiberlookup_Bk_23.out_crd())
        crddrop_12.set_inner_crd(intersectl_17.out_crd())
        fiberlookup_Vm_16.set_in_ref(intersectl_17.out_ref1())
        arrayvals_V_9.set_load(fiberlookup_Vm_16.out_ref())
        repsiggen_m_14.set_istream(fiberlookup_Vm_16.out_crd())
        repeat_Bm_13.set_in_ref(intersectl_17.out_ref2())
        repeat_Bm_13.set_in_repsig(repsiggen_m_14.out_repsig())
        crddrop_11.set_outer_crd(intersectj_24.out_crd())
        crddrop_11.set_inner_crd(crddrop_12.out_crd_outer())
        crddrop_10.set_outer_crd(intersecti_27.out_crd())
        crddrop_10.set_inner_crd(crddrop_11.out_crd_outer())
        fiberwrite_X1_2.set_input(fiberlookup_Bk_23.out_crd())
        arrayvals_B_8.set_load(repeat_Bm_13.out_ref())
        mul_7.set_in1(arrayvals_B_8.out_val())
        mul_7.set_in2(arrayvals_V_9.out_val())
        # FIXME: Compiler bug possibly
        reduce_6.set_in_val(mul_7.out_val())
        # spaccumulator1_5_drop_crd_outer.set_in_stream(crddrop_12.out_crd_inner())
        # spaccumulator1_5_drop_val.set_in_stream(reduce_6.out_val())
        # spaccumulator1_5_drop_crd_inner.set_in_stream(fiberlookup_Vm_16.out_crd())
        spaccumulator1_5.set_in_crd0(fiberlookup_Vm_16.out_crd())
        spaccumulator1_5.set_in_crd1(intersectl_17.out_crd())
        # spaccumulator1_5.set_val(reduce_6.out_val())
        spaccumulator1_5.set_val(mul_7.out_val())
        fiberwrite_X0_4.set_input(intersecti_27.out_crd())
        fiberwrite_X2_3.set_input(intersectj_24.out_crd())
        fiberwrite_Xvals_0.set_input(spaccumulator1_5.out_val())
        fiberwrite_X3_1.set_input(spaccumulator1_5.out_crd0())
        fiberlookup_Bi_28.update()

        fiberlookup_Vi_29.update()

        intersecti_27.update()
        fiberlookup_Bj_25.update()
        fiberlookup_Vj_26.update()
        intersectj_24.update()
        fiberlookup_Bk_23.update()
        fiberlookup_Bl_18.update()
        repsiggen_k_21.update()
        repeat_Vk_20.update()
        fiberlookup_Vl_19.update()
        intersectl_17.update()
        crddrop_12.update()
        fiberlookup_Vm_16.update()
        arrayvals_V_9.update()
        repsiggen_m_14.update()
        repeat_Bm_13.update()
        crddrop_11.update()
        crddrop_10.update()
        fiberwrite_X1_2.update()
        arrayvals_B_8.update()
        mul_7.update()
        reduce_6.update()
        # spaccumulator1_5_drop_crd_inner.update()
        # spaccumulator1_5_drop_val.update()
        spaccumulator1_5.update()
        fiberwrite_X0_4.update()
        fiberwrite_X2_3.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X3_1.update()

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

    print("vals", out_vals)
    print("# cycles: ", time_cnt)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_V_shape"] = V_shape
    sample_dict = fiberlookup_Bi_28.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_28" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_27" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_10" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_25" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_24" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_23" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_12" + "_" + k] = sample_dict[k]

    sample_dict = spaccumulator1_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Vk_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Vk_20" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vl_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vl_19" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_17" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bm_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bm_13" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_8" + "_" + k] = sample_dict[k]

    sample_dict = reduce_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vm_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vm_16" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_V_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_V_9" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_18" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vj_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vj_26" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vi_29.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vi_29" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_mult2(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "ssss0213")
    samBench(bench, extra_info)
