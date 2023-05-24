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
def test_tensor4_fusedsoftmax_ijklm(samBench, frosttname, cast, check_gold, debug_sim, backpressure, depth, report_stats, fill=0):
    Q_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fusedsoftmax_ijklm")
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

    K_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fusedsoftmax_ijklm")
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

    V_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fusedsoftmax_ijklm")
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


    fiberlookup_Qi_41 = CompressedCrdRdScan(crd_arr=Q_crd0, seg_arr=Q_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ki_42 = CompressedCrdRdScan(crd_arr=K_crd0, seg_arr=K_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vi_43 = CompressedCrdRdScan(crd_arr=V_crd0, seg_arr=V_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti_40 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti1_40 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersecti2_40 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qj_37 = CompressedCrdRdScan(crd_arr=Q_crd2, seg_arr=Q_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Kj_38 = CompressedCrdRdScan(crd_arr=K_crd2, seg_arr=K_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vj_39 = CompressedCrdRdScan(crd_arr=V_crd2, seg_arr=V_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj_36 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj1_36 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectj2_36 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crddrop_19 = CrdDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Kl_34 = CompressedCrdRdScan(crd_arr=K_crd1, seg_arr=K_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vl_35 = CompressedCrdRdScan(crd_arr=V_crd1, seg_arr=V_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_13 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectl_33 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_12 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_l_31 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_9 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Ql_30 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qk_29 = CompressedCrdRdScan(crd_arr=Q_crd1, seg_arr=Q_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Qm_21 = CompressedCrdRdScan(crd_arr=Q_crd3, seg_arr=Q_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_11 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_k_27 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_8 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Kk_24 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Vk_25 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Km_22 = CompressedCrdRdScan(crd_arr=K_crd3, seg_arr=K_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Vm_23 = CompressedCrdRdScan(crd_arr=V_crd3, seg_arr=V_seg3, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectm_20 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectm1_20 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectm2_20 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_10 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_Q_16 = Array(init_arr=Q_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_K_17 = Array(init_arr=K_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_V_18 = Array(init_arr=V_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_m_45 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_7 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_15 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crdhold_6 = CrdHold(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_44 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Km_47 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_14 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator2_5 = SparseAccumulator1(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator2_5_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator2_5_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    spaccumulator2_5_drop_val = StknDrop(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * Q_shape[0] * Q_shape[2] * Q_shape[1] * Q_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X3_1 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[2] * Q_shape[1] + 1, size=Q_shape[0] * Q_shape[2] * Q_shape[1] * Q_shape[3], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_2 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[2] + 1, size=Q_shape[0] * Q_shape[2] * Q_shape[1], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_3 = CompressWrScan(seg_size=Q_shape[0] + 1, size=Q_shape[0] * Q_shape[2], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=Q_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    in_ref_Q = [0, 'D']
    in_ref_K = [0, 'D']
    in_ref_V = [0, 'D']
    done = False
    time_cnt = 0

    inner = []
    outer = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_Q) > 0:
            fiberlookup_Qi_41.set_in_ref(in_ref_Q.pop(0))
        if len(in_ref_K) > 0:
            fiberlookup_Ki_42.set_in_ref(in_ref_K.pop(0))
        if len(in_ref_V) > 0:
            fiberlookup_Vi_43.set_in_ref(in_ref_V.pop(0))
        intersecti_40.set_in1(fiberlookup_Qi_41.out_ref(), fiberlookup_Qi_41.out_crd())
        intersecti_40.set_in2(fiberlookup_Ki_42.out_ref(), fiberlookup_Ki_42.out_crd())
        
        intersecti1_40.set_in1(fiberlookup_Vi_43.out_ref(), fiberlookup_Vi_43.out_crd())
        intersecti1_40.set_in2(intersecti_40.out_ref1(), intersecti_40.out_crd())

        intersecti2_40.set_in1(fiberlookup_Vi_43.out_ref(), fiberlookup_Vi_43.out_crd())
        intersecti2_40.set_in2(intersecti_40.out_ref2(), intersecti_40.out_crd())

        fiberlookup_Qj_37.set_in_ref(intersecti1_40.out_ref2())
        fiberlookup_Kj_38.set_in_ref(intersecti2_40.out_ref2())
        fiberlookup_Vj_39.set_in_ref(intersecti2_40.out_ref1())
        intersectj_36.set_in1(fiberlookup_Qj_37.out_ref(), fiberlookup_Qj_37.out_crd())
        intersectj_36.set_in2(fiberlookup_Kj_38.out_ref(), fiberlookup_Kj_38.out_crd())

        intersectj1_36.set_in1(fiberlookup_Vj_39.out_ref(), fiberlookup_Vj_39.out_crd())
        intersectj1_36.set_in2(intersectj_36.out_ref1(), intersectj_36.out_crd())

        intersectj2_36.set_in1(fiberlookup_Vj_39.out_ref(), fiberlookup_Vj_39.out_crd())
        intersectj2_36.set_in2(intersectj_36.out_ref2(), intersectj_36.out_crd())

        crddrop_19.set_outer_crd(intersecti2_40.out_crd())
        crddrop_19.set_inner_crd(intersectj2_36.out_crd())
        fiberlookup_Kl_34.set_in_ref(intersectj2_36.out_ref2())
        fiberlookup_Vl_35.set_in_ref(intersectj2_36.out_ref1())
        intersectl_33.set_in1(fiberlookup_Kl_34.out_ref(), fiberlookup_Kl_34.out_crd())
        intersectl_33.set_in2(fiberlookup_Vl_35.out_ref(), fiberlookup_Vl_35.out_crd())
        repsiggen_l_31.set_istream(intersectl_33.out_crd())
        repeat_Ql_30.set_in_ref(intersectj1_36.out_ref2())
        repeat_Ql_30.set_in_repsig(repsiggen_l_31.out_repsig())
        crdhold_13.set_outer_crd(crddrop_19.out_crd_outer())
        crdhold_13.set_inner_crd(crddrop_19.out_crd_inner())
        crdhold_12.set_outer_crd(crdhold_13.out_crd_outer())
        crdhold_12.set_inner_crd(intersectl_33.out_crd())
        crdhold_9.set_outer_crd(crdhold_13.out_crd_inner())
        crdhold_9.set_inner_crd(crdhold_12.out_crd_inner())
        fiberlookup_Qk_29.set_in_ref(repeat_Ql_30.out_ref())
        fiberlookup_Qm_21.set_in_ref(fiberlookup_Qk_29.out_ref())
        crdhold_11.set_outer_crd(crdhold_12.out_crd_outer())
        crdhold_11.set_inner_crd(fiberlookup_Qk_29.out_crd())
        repsiggen_k_27.set_istream(fiberlookup_Qk_29.out_crd())
        repeat_Kk_24.set_in_repsig(repsiggen_k_27.out_repsig())
        repeat_Kk_24.set_in_ref(intersectl_33.out_ref1())
        repeat_Vk_25.set_in_repsig(repsiggen_k_27.out_repsig())
        repeat_Vk_25.set_in_ref(intersectl_33.out_ref2())
        fiberlookup_Km_22.set_in_ref(repeat_Kk_24.out_ref())
        fiberlookup_Vm_23.set_in_ref(repeat_Vk_25.out_ref())

        intersectm_20.set_in1(fiberlookup_Km_22.out_ref(), fiberlookup_Km_22.out_crd())
        intersectm_20.set_in2(fiberlookup_Vm_23.out_ref(), fiberlookup_Vm_23.out_crd())

        intersectm1_20.set_in1(fiberlookup_Qm_21.out_ref(), fiberlookup_Qm_21.out_crd())
        intersectm1_20.set_in2(intersectm_20.out_ref1(), intersectm_20.out_crd())

        intersectm2_20.set_in1(fiberlookup_Qm_21.out_ref(), fiberlookup_Qm_21.out_crd())
        intersectm2_20.set_in2(intersectm_20.out_ref2(), intersectm_20.out_crd())

        # intersectm_20.set_in3(fiberlookup_Qm_21.out_ref(), fiberlookup_Qm_21.out_crd())
        crdhold_10.set_outer_crd(crdhold_11.out_crd_outer())
        crdhold_10.set_inner_crd(intersectm1_20.out_crd())
        crdhold_8.set_inner_crd(crdhold_11.out_crd_inner())
        crdhold_8.set_outer_crd(crdhold_9.out_crd_outer())
        crdhold_7.set_inner_crd(crdhold_10.out_crd_inner())
        crdhold_7.set_outer_crd(crdhold_8.out_crd_outer())
        crdhold_6.set_inner_crd(crdhold_7.out_crd_inner())
        crdhold_6.set_outer_crd(crdhold_8.out_crd_inner())
        arrayvals_Q_16.set_load(intersectm2_20.out_ref1())
        arrayvals_K_17.set_load(intersectm1_20.out_ref2())
        arrayvals_V_18.set_load(intersectm2_20.out_ref2())
        repsiggen_m_45.set_istream(intersectm2_20.out_crd())
        mul_15.set_in1(arrayvals_Q_16.out_val())
        mul_15.set_in2(arrayvals_K_17.out_val())
        reduce_44.set_in_val(mul_15.out_val())
        repeat_Km_47.set_in_ref(reduce_44.out_val())
        repeat_Km_47.set_in_repsig(repsiggen_m_45.out_repsig())
        mul_14.set_in1(repeat_Km_47.out_ref())
        mul_14.set_in2(arrayvals_V_18.out_val())
        spaccumulator2_5_drop_crd_outer.set_in_stream(crdhold_6.out_crd_outer())
        spaccumulator2_5_drop_crd_inner.set_in_stream(crdhold_6.out_crd_inner())


        spaccumulator2_5_drop_val.set_in_stream(mul_14.out_val())
        spaccumulator2_5.set_crd_outer(spaccumulator2_5_drop_crd_outer.out_val())
        # spaccumulator2_5.set_crd_outer(spaccumulator2_5_drop_crd_outer.out_val())
        # spaccumulator2_5.set_crd_outer(spaccumulator2_5_drop_crd_outer.out_val())
        spaccumulator2_5.set_crd_inner(spaccumulator2_5_drop_crd_inner.out_val())
        # spaccumulator2_5.set_crd_inner(spaccumulator2_5_drop_crd_inner.out_val())
        spaccumulator2_5.set_val(spaccumulator2_5_drop_val.out_val())
        inner.append(spaccumulator2_5_drop_crd_outer.out_val())
        outer.append(spaccumulator2_5.out_crd_outer())
        print("Inner:", remove_emptystr(inner))
        print("Outer:", remove_emptystr(outer))
        fiberwrite_Xvals_0.set_input(spaccumulator2_5.out_val())
        fiberwrite_X3_1.set_input(spaccumulator2_5.out_crd_inner())
        fiberwrite_X1_2.set_input(spaccumulator2_5.out_crd_outer())
        fiberwrite_X2_3.set_input(fiberlookup_Qk_29.out_crd())
        # fiberwrite_X0_4.set_input(spaccumulator2_5.out_crd_outer())
        fiberwrite_X0_4.set_input(intersecti1_40.out_crd())
        fiberlookup_Qi_41.update()

        fiberlookup_Ki_42.update()

        fiberlookup_Vi_43.update()

        intersecti_40.update()
        intersecti1_40.update()
        intersecti2_40.update()
        fiberlookup_Qj_37.update()
        fiberlookup_Kj_38.update()
        fiberlookup_Vj_39.update()
        intersectj_36.update()
        intersectj1_36.update()
        intersectj2_36.update()
        crddrop_19.update()
        fiberlookup_Kl_34.update()
        fiberlookup_Vl_35.update()
        intersectl_33.update()
        repsiggen_l_31.update()
        repeat_Ql_30.update()
        crdhold_13.update()
        crdhold_12.update()
        crdhold_9.update()
        fiberlookup_Qk_29.update()
        fiberlookup_Qm_21.update()
        crdhold_11.update()
        repsiggen_k_27.update()
        repeat_Kk_24.update()
        repeat_Vk_25.update()
        fiberlookup_Km_22.update()
        fiberlookup_Vm_23.update()
        intersectm_20.update()
        intersectm1_20.update()
        intersectm2_20.update()
        crdhold_10.update()
        crdhold_8.update()
        crdhold_7.update()
        crdhold_6.update()
        arrayvals_Q_16.update()
        arrayvals_K_17.update()
        arrayvals_V_18.update()
        repsiggen_m_45.update()
        mul_15.update()
        reduce_44.update()
        repeat_Km_47.update()
        mul_14.update()
        spaccumulator2_5_drop_crd_outer.update()
        # spaccumulator2_5_drop_crd_outer.update()
        # spaccumulator2_5_drop_crd_outer.update()
        spaccumulator2_5_drop_crd_inner.update()
        # spaccumulator2_5_drop_crd_inner.update()
        spaccumulator2_5_drop_val.update()
        spaccumulator2_5.update()
        fiberwrite_Xvals_0.update()
        fiberwrite_X3_1.update()
        fiberwrite_X1_2.update()
        fiberwrite_X2_3.update()
        fiberwrite_X0_4.update()

        done = fiberwrite_X0_4.out_done() and fiberwrite_X2_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X3_1.out_done() and fiberwrite_Xvals_0.out_done()
        # done = fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_4.autosize()
    fiberwrite_X2_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X3_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_4.get_arr(), fiberwrite_X2_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X3_1.get_arr()]
    out_segs = [fiberwrite_X0_4.get_seg_arr(), fiberwrite_X2_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X3_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print("Segs:", out_segs)
    print("Crds:", out_crds)
    print("Vals:", out_vals)

    pytest.set_trace()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_Q_shape"] = Q_shape
    extra_info["tensor_K_shape"] = K_shape
    extra_info["tensor_V_shape"] = V_shape
    sample_dict = fiberlookup_Qi_41.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qi_41" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_40.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_40" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_19" + "_" + k] = sample_dict[k]

    sample_dict = spaccumulator2_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator2_5" + "_" + k] = sample_dict[k]

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

    sample_dict = fiberlookup_Qj_37.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qj_37" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_36.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_36" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ql_30.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ql_30" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qk_29.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qk_29" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Kk_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Kk_24" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Km_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Km_22" + "_" + k] = sample_dict[k]

    sample_dict = intersectm_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectm_20" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_Q_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_Q_16" + "_" + k] = sample_dict[k]

    sample_dict = reduce_44.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_44" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Km_47.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Km_47" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_K_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_K_17" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_V_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_V_18" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Vk_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Vk_25" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vm_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vm_23" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qm_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qm_21" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kl_34.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kl_34" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_33.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_33" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vl_35.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vl_35" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kj_38.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kj_38" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vj_39.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vj_39" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ki_42.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ki_42" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vi_43.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vi_43" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_fusedsoftmax_ijklm(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "ssss0213")
    samBench(bench, extra_info)
