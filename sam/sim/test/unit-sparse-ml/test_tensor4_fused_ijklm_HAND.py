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
def test_tensor4_fused_ijklm_HAND(samBench, frosttname, cast, check_gold, debug_sim, backpressure, depth, report_stats, fill=0):
    Q_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fused_mul_T1")
    Q_shape_filename = os.path.join(Q_dirname, "tensor_Q_mode_shape")
    Q_shape = read_inputs(Q_shape_filename)

    Q0_seg_filename = os.path.join(Q_dirname, "tensor_Q_mode_0_seg")
    Q_seg0 = read_inputs(Q0_seg_filename)
    Q0_crd_filename = os.path.join(Q_dirname, "tensor_Q_mode_0_crd")
    Q_crd0 = read_inputs(Q0_crd_filename)

    Q1_seg_filename = os.path.join(Q_dirname, "tensor_Q_mode_1_seg")
    Q_seg1 = read_inputs(Q1_seg_filename)
    Q1_crd_filename = os.path.join(Q_dirname, "tensor_Q_mode_1_crd")
    Q_crd1 = read_inputs(Q1_crd_filename)

    Q2_seg_filename = os.path.join(Q_dirname, "tensor_Q_mode_2_seg")
    Q_seg2 = read_inputs(Q2_seg_filename)
    Q2_crd_filename = os.path.join(Q_dirname, "tensor_Q_mode_2_crd")
    Q_crd2 = read_inputs(Q2_crd_filename)

    Q3_seg_filename = os.path.join(Q_dirname, "tensor_Q_mode_3_seg")
    Q_seg3 = read_inputs(Q3_seg_filename)
    Q3_crd_filename = os.path.join(Q_dirname, "tensor_Q_mode_3_crd")
    Q_crd3 = read_inputs(Q3_crd_filename)

    Q_vals_filename = os.path.join(Q_dirname, "tensor_Q_mode_vals")
    Q_vals = read_inputs(Q_vals_filename, float)

    K_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fused_mul_T1")
    K_shape_filename = os.path.join(K_dirname, "tensor_K_mode_shape")
    K_shape = read_inputs(K_shape_filename)

    K0_seg_filename = os.path.join(K_dirname, "tensor_K_mode_0_seg")
    K_seg0 = read_inputs(K0_seg_filename)
    K0_crd_filename = os.path.join(K_dirname, "tensor_K_mode_0_crd")
    K_crd0 = read_inputs(K0_crd_filename)

    K1_seg_filename = os.path.join(K_dirname, "tensor_K_mode_1_seg")
    K_seg1 = read_inputs(K1_seg_filename)
    K1_crd_filename = os.path.join(K_dirname, "tensor_K_mode_1_crd")
    K_crd1 = read_inputs(K1_crd_filename)

    K2_seg_filename = os.path.join(K_dirname, "tensor_K_mode_2_seg")
    K_seg2 = read_inputs(K2_seg_filename)
    K2_crd_filename = os.path.join(K_dirname, "tensor_K_mode_2_crd")
    K_crd2 = read_inputs(K2_crd_filename)

    K3_seg_filename = os.path.join(K_dirname, "tensor_K_mode_3_seg")
    K_seg3 = read_inputs(K3_seg_filename)
    K3_crd_filename = os.path.join(K_dirname, "tensor_K_mode_3_crd")
    K_crd3 = read_inputs(K3_crd_filename)

    K_vals_filename = os.path.join(K_dirname, "tensor_K_mode_vals")
    K_vals = read_inputs(K_vals_filename, float)

    V_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fused_mul_T1")
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

    print(Q_seg0)
    print(Q_seg1)
    print(Q_seg2)
    print(Q_seg3)

    print(K_seg0)
    print(K_seg1)
    print(K_seg2)
    print(K_seg3)

    print(V_seg0)
    print(V_seg1)
    print(V_seg2)
    print(V_seg3)
    # pytest.set_trace()

    fiberlookup_Vi_35 = CompressedCrdRdScan(crd_arr=V_crd0, seg_arr=V_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qi_425 = CompressedCrdRdScan(crd_arr=Q_crd0, seg_arr=Q_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ki_426 = CompressedCrdRdScan(crd_arr=K_crd0, seg_arr=K_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti_424 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti2_424 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti3_424 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vj_32 = CompressedCrdRdScan(crd_arr=V_crd2, seg_arr=V_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qj_422 = CompressedCrdRdScan(crd_arr=Q_crd2, seg_arr=Q_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Kj_423 = CompressedCrdRdScan(crd_arr=K_crd2, seg_arr=K_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj_421 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj2_421 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj3_421 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crddrop_49 = CrdDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qk_420 = CompressedCrdRdScan(crd_arr=Q_crd1, seg_arr=Q_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_14 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_44 = CompressWrScan(seg_size=2, size=Q_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_2 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[2] + 1, size=Q_shape[0] * Q_shape[2] * Q_shape[1], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_k_418 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_13 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Vk_26 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Kk_417 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_10 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vl_25 = CompressedCrdRdScan(crd_arr=V_crd1, seg_arr=V_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Kl_416 = CompressedCrdRdScan(crd_arr=K_crd1, seg_arr=K_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectl_23 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_12 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vm_22 = CompressedCrdRdScan(crd_arr=V_crd3, seg_arr=V_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Km_412 = CompressedCrdRdScan(crd_arr=K_crd3, seg_arr=K_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_l_414 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_9 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Ql_413 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_7 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qm_411 = CompressedCrdRdScan(crd_arr=Q_crd3, seg_arr=Q_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectm_410 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectm2_410 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectm3_410 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_11 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_Q_47 = Array(init_arr=Q_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_K_48 = Array(init_arr=K_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_V_17 = Array(init_arr=V_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_m_20 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_8 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_46 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_6 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_45 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    maxreduce_434 = MaxReduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_QKl_437 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    add_433 = Add2(debug=debug_sim, neg2=True, statistics=report_stats, back_en=backpressure, depth=int(depth))
    exp_427 = Exp(in2=0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_428 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_QKl_431 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    div_432 = Divide2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_QKm_19 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_15 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5 = SparseAccumulator1(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator1_5_drop_val = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * Q_shape[0] * Q_shape[2] * Q_shape[1] * K_shape[1], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X3_1 = CompressWrScan(seg_size=V_shape[0] * V_shape[1] * V_shape[2] + 1, size=V_shape[0] * V_shape[1] * V_shape[2] * V_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_3 = CompressWrScan(seg_size=V_shape[0] + 1, size=V_shape[0] * V_shape[1], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    in_ref_V = [0, 'D']
    in_ref_Q = [0, 'D']
    in_ref_K = [0, 'D']
    done = False
    time_cnt = 0

    out_debug = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_V) > 0:
            fiberlookup_Vi_35.set_in_ref(in_ref_V.pop(0))
        if len(in_ref_Q) > 0:
            fiberlookup_Qi_425.set_in_ref(in_ref_Q.pop(0))
        if len(in_ref_K) > 0:
            fiberlookup_Ki_426.set_in_ref(in_ref_K.pop(0))
        intersecti_424.set_in1(fiberlookup_Vi_35.out_ref(), fiberlookup_Vi_35.out_crd())
        intersecti_424.set_in2(fiberlookup_Qi_425.out_ref(), fiberlookup_Qi_425.out_crd())
        # intersecti_424.set_in3(fiberlookup_Ki_426.out_ref(), fiberlookup_Ki_426.out_crd())

        intersecti2_424.set_in1(fiberlookup_Ki_426.out_ref(), fiberlookup_Ki_426.out_crd())
        intersecti2_424.set_in2(intersecti_424.out_ref1(), intersecti_424.out_crd())

        intersecti3_424.set_in1(fiberlookup_Ki_426.out_ref(), fiberlookup_Ki_426.out_crd())
        intersecti3_424.set_in2(intersecti_424.out_ref2(), intersecti_424.out_crd())

        # fiberlookup_Vj_32.set_in_ref(intersecti_424.out_ref1())
        # fiberlookup_Qj_422.set_in_ref(intersecti_424.out_ref2())
        # fiberlookup_Kj_423.set_in_ref(intersecti_424.out_ref3())
        fiberlookup_Vj_32.set_in_ref(intersecti2_424.out_ref2())
        fiberlookup_Qj_422.set_in_ref(intersecti3_424.out_ref2())
        fiberlookup_Kj_423.set_in_ref(intersecti3_424.out_ref1())

        intersectj_421.set_in1(fiberlookup_Vj_32.out_ref(), fiberlookup_Vj_32.out_crd())
        intersectj_421.set_in2(fiberlookup_Qj_422.out_ref(), fiberlookup_Qj_422.out_crd())
        # intersectj_421.set_in3(fiberlookup_Kj_423.out_ref(), fiberlookup_Kj_423.out_crd())

        intersectj2_421.set_in1(fiberlookup_Kj_423.out_ref(), fiberlookup_Kj_423.out_crd())
        intersectj2_421.set_in2(intersectj_421.out_ref1(), intersectj_421.out_crd())

        intersectj3_421.set_in1(fiberlookup_Kj_423.out_ref(), fiberlookup_Kj_423.out_crd())
        intersectj3_421.set_in2(intersectj_421.out_ref2(), intersectj_421.out_crd())

        crddrop_49.set_outer_crd(intersecti3_424.out_crd())
        crddrop_49.set_inner_crd(intersectj3_421.out_crd())

        fiberlookup_Qk_420.set_in_ref(intersectj3_421.out_ref2())
        fiberwrite_X2_2.set_input(fiberlookup_Qk_420.out_crd())
        repsiggen_k_418.set_istream(fiberlookup_Qk_420.out_crd())
        repeat_Vk_26.set_in_ref(intersectj2_421.out_ref2())
        repeat_Vk_26.set_in_repsig(repsiggen_k_418.out_repsig())
        repeat_Kk_417.set_in_ref(intersectj3_421.out_ref1())

        repeat_Kk_417.set_in_repsig(repsiggen_k_418.out_repsig())
        fiberlookup_Kl_416.set_in_ref(repeat_Kk_417.out_ref())
        fiberlookup_Vl_25.set_in_ref(repeat_Vk_26.out_ref())
        intersectl_23.set_in1(fiberlookup_Vl_25.out_ref(), fiberlookup_Vl_25.out_crd())
        intersectl_23.set_in2(fiberlookup_Kl_416.out_ref(), fiberlookup_Kl_416.out_crd())
        fiberlookup_Vm_22.set_in_ref(intersectl_23.out_ref1())
        fiberlookup_Km_412.set_in_ref(intersectl_23.out_ref2())
        repsiggen_l_414.set_istream(intersectl_23.out_crd())
        crdhold_14.set_outer_crd(crddrop_49.out_crd_outer())
        crdhold_14.set_inner_crd(crddrop_49.out_crd_inner())
        fiberwrite_X0_44.set_input(crddrop_49.out_crd_outer())
        crdhold_13.set_outer_crd(crdhold_14.out_crd_outer())
        crdhold_13.set_inner_crd(fiberlookup_Qk_420.out_crd())
        repeat_Ql_413.set_in_ref(fiberlookup_Qk_420.out_ref())
        repeat_Ql_413.set_in_repsig(repsiggen_l_414.out_repsig())
        fiberlookup_Qm_411.set_in_ref(repeat_Ql_413.out_ref())

        intersectm_410.set_in1(fiberlookup_Vm_22.out_ref(), fiberlookup_Vm_22.out_crd())
        intersectm_410.set_in2(fiberlookup_Qm_411.out_ref(), fiberlookup_Qm_411.out_crd())
        # intersectm_410.set_in3(fiberlookup_Km_412.out_ref(), fiberlookup_Km_412.out_crd())

        intersectm2_410.set_in1(fiberlookup_Km_412.out_ref(), fiberlookup_Km_412.out_crd())
        intersectm2_410.set_in2(intersectm_410.out_ref1(), intersectm_410.out_crd())

        intersectm3_410.set_in1(fiberlookup_Km_412.out_ref(), fiberlookup_Km_412.out_crd())
        intersectm3_410.set_in2(intersectm_410.out_ref2(), intersectm_410.out_crd())

        arrayvals_Q_47.set_load(intersectm3_410.out_ref2())
        arrayvals_K_48.set_load(intersectm3_410.out_ref1())
        arrayvals_V_17.set_load(intersectm2_410.out_ref2())

        repsiggen_m_20.set_istream(intersectm_410.out_crd())
        mul_46.set_in1(arrayvals_Q_47.out_val())
        mul_46.set_in2(arrayvals_K_48.out_val())
        reduce_45.set_in_val(mul_46.out_val())
        maxreduce_434.set_in_val(reduce_45.out_val())
        repeat_QKl_437.set_in_repsig(repsiggen_l_414.out_repsig())
        repeat_QKl_437.set_in_ref(maxreduce_434.out_val())
        add_433.set_in1(reduce_45.out_val())
        add_433.set_in2(repeat_QKl_437.out_ref())
        crdhold_10.set_outer_crd(crdhold_14.out_crd_inner())
        crdhold_10.set_inner_crd(crdhold_13.out_crd_inner())
        crdhold_12.set_outer_crd(crdhold_13.out_crd_outer())
        crdhold_12.set_inner_crd(intersectl_23.out_crd())
        crdhold_11.set_outer_crd(crdhold_12.out_crd_outer())
        crdhold_9.set_inner_crd(crdhold_12.out_crd_inner())
        crdhold_9.set_outer_crd(crdhold_10.out_crd_outer())
        crdhold_8.set_inner_crd(crdhold_11.out_crd_inner())
        crdhold_8.set_outer_crd(crdhold_9.out_crd_outer())
        crdhold_7.set_inner_crd(crdhold_9.out_crd_inner())
        crdhold_7.set_outer_crd(crdhold_10.out_crd_inner())
        crdhold_6.set_inner_crd(crdhold_8.out_crd_inner())
        crdhold_6.set_outer_crd(crdhold_7.out_crd_outer())
        exp_427.set_in1(add_433.out_val())
        reduce_428.set_in_val(exp_427.out_val())
        repeat_QKl_431.set_in_repsig(repsiggen_l_414.out_repsig())
        repeat_QKl_431.set_in_ref(reduce_428.out_val())
        div_432.set_in1(exp_427.out_val())
        div_432.set_in2(repeat_QKl_431.out_ref())
        repeat_QKm_19.set_in_repsig(repsiggen_m_20.out_repsig())
        repeat_QKm_19.set_in_ref(div_432.out_val())
        mul_15.set_in1(repeat_QKm_19.out_ref())
        mul_15.set_in2(arrayvals_V_17.out_val())
        spaccumulator1_5_drop_crd_outer.set_in_stream(crdhold_6.out_crd_outer())
        spaccumulator1_5_drop_crd_inner.set_in_stream(crdhold_6.out_crd_inner())
        spaccumulator1_5_drop_val.set_in_stream(mul_15.out_val())

        out_debug.append(maxreduce_434.out_val())
        print("val: ", remove_emptystr(out_debug))

        spaccumulator1_5.set_crd_outer(spaccumulator1_5_drop_crd_outer.out_val())
        # spaccumulator1_5.set_crd_outer(spaccumulator1_5_drop_crd_outer.out_val())
        # spaccumulator1_5.set_crd_outer(spaccumulator1_5_drop_crd_outer.out_val())
        spaccumulator1_5.set_crd_inner(spaccumulator1_5_drop_crd_inner.out_val())
        # spaccumulator1_5.set_crd_inner(spaccumulator1_5_drop_crd_inner.out_val())
        spaccumulator1_5.set_val(spaccumulator1_5_drop_val.out_val())
        fiberwrite_Xvals_0.set_input(spaccumulator1_5.out_val())
        fiberwrite_X3_1.set_input(spaccumulator1_5.out_crd_inner())
        fiberwrite_X2_3.set_input(spaccumulator1_5.out_crd_outer())
        fiberlookup_Vi_35.update()

        fiberlookup_Qi_425.update()

        fiberlookup_Ki_426.update()

        intersecti_424.update()
        intersecti2_424.update()
        intersecti3_424.update()
        fiberlookup_Vj_32.update()
        fiberlookup_Qj_422.update()
        fiberlookup_Kj_423.update()
        intersectj_421.update()
        intersectj2_421.update()
        intersectj3_421.update()
        crddrop_49.update()
        fiberlookup_Qk_420.update()
        fiberwrite_X2_2.update()
        repsiggen_k_418.update()
        repeat_Vk_26.update()
        repeat_Kk_417.update()
        fiberlookup_Kl_416.update()
        fiberlookup_Vl_25.update()
        intersectl_23.update()
        fiberlookup_Vm_22.update()
        fiberlookup_Km_412.update()
        repsiggen_l_414.update()
        crdhold_14.update()
        fiberwrite_X0_44.update()
        crdhold_13.update()
        repeat_Ql_413.update()
        fiberlookup_Qm_411.update()
        intersectm_410.update()
        intersectm2_410.update()
        intersectm3_410.update()
        arrayvals_Q_47.update()
        arrayvals_K_48.update()
        arrayvals_V_17.update()
        repsiggen_m_20.update()
        mul_46.update()
        reduce_45.update()
        maxreduce_434.update()
        repeat_QKl_437.update()
        add_433.update()
        crdhold_10.update()
        crdhold_12.update()
        crdhold_11.update()
        crdhold_9.update()
        crdhold_8.update()
        crdhold_7.update()
        crdhold_6.update()
        exp_427.update()
        reduce_428.update()
        repeat_QKl_431.update()
        div_432.update()
        repeat_QKm_19.update()
        mul_15.update()
        spaccumulator1_5_drop_crd_outer.update()
        # spaccumulator1_5_drop_crd_outer.update()
        # spaccumulator1_5_drop_crd_outer.update()
        spaccumulator1_5_drop_crd_inner.update()
        # spaccumulator1_5_drop_crd_inner.update()
        spaccumulator1_5_drop_val.update()
        spaccumulator1_5.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X3_1.update()
        fiberwrite_X2_3.update()

        done = fiberwrite_X0_44.out_done() and fiberwrite_X2_2.out_done() and fiberwrite_X3_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_44.autosize()
    fiberwrite_X2_2.autosize()
    fiberwrite_X3_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_44.get_arr(), fiberwrite_X2_2.get_arr(), fiberwrite_X3_1.get_arr()]
    out_segs = [fiberwrite_X0_44.get_seg_arr(), fiberwrite_X2_2.get_seg_arr(), fiberwrite_X3_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print(out_crds)
    print(out_segs)
    print(out_vals)

    pytest.set_trace()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_Q_shape"] = Q_shape
    extra_info["tensor_K_shape"] = K_shape
    extra_info["tensor_V_shape"] = V_shape
    sample_dict = spaccumulator1_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Vk_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Vk_26" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vl_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vl_25" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_23" + "_" + k] = sample_dict[k]

    sample_dict = repeat_QKm_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_QKm_19" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vm_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vm_22" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_V_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_V_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vj_32.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vj_32" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vi_35.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vi_35" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qi_425.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qi_425" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_424.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_424" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_49.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_49" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_44.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_44" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qj_422.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qj_422" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_421.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_421" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qk_420.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qk_420" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Kk_417.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Kk_417" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kl_416.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kl_416" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ql_413.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ql_413" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qm_411.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qm_411" + "_" + k] = sample_dict[k]

    sample_dict = intersectm_410.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectm_410" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_Q_47.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_Q_47" + "_" + k] = sample_dict[k]

    sample_dict = reduce_45.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_45" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_K_48.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_K_48" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Km_412.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Km_412" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kj_423.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kj_423" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ki_426.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ki_426" + "_" + k] = sample_dict[k]

    sample_dict = reduce_428.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_428" + "_" + k] = sample_dict[k]

    sample_dict = repeat_QKl_437.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_QKl_437" + "_" + k] = sample_dict[k]

    sample_dict = repeat_QKl_431.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_QKl_431" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_fused_ijklm_HAND(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "ssss0213")
    samBench(bench, extra_info)