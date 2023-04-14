import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.crd_masker import RandomDropout
from sam.sim.src.unary_alu import Max, Exp
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
def test_tensor4_fused(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    Q_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fusedmul")
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

    K_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fusedmul")
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

    V_dirname = os.path.join(formatted_dir, frosttname, "tensor4_fusedmul")
    V_shape_filename = os.path.join(V_dirname, "tensor_V_mode_shape")
    V_shape = read_inputs(V_shape_filename)

    V0_seg_filename = os.path.join(V_dirname, "tensor_V_mode_0_seg")
    V_seg0 = read_inputs(V0_seg_filename)
    V0_crd_filename = os.path.join(V_dirname, "tensor_V_mode_0_crd")
    V_crd0 = read_inputs(V0_crd_filename)

    V1_seg_filename = os.path.join(V_dirname, "tensor_V_mode_2_seg")
    V_seg1 = read_inputs(V1_seg_filename)
    V1_crd_filename = os.path.join(V_dirname, "tensor_V_mode_2_crd")
    V_crd1 = read_inputs(V1_crd_filename)

    V2_seg_filename = os.path.join(V_dirname, "tensor_V_mode_3_seg")
    V_seg2 = read_inputs(V2_seg_filename)
    V2_crd_filename = os.path.join(V_dirname, "tensor_V_mode_3_crd")
    V_crd2 = read_inputs(V2_crd_filename)

    V3_seg_filename = os.path.join(V_dirname, "tensor_V_mode_1_seg")
    V_seg3 = read_inputs(V3_seg_filename)
    V3_crd_filename = os.path.join(V_dirname, "tensor_V_mode_1_crd")
    V_crd3 = read_inputs(V3_crd_filename)

    V_vals_filename = os.path.join(V_dirname, "tensor_V_mode_vals")
    V_vals = read_inputs(V_vals_filename, float)


    fiberlookup_Qi_32 = CompressedCrdRdScan(crd_arr=Q_crd0, seg_arr=Q_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ki_33 = CompressedCrdRdScan(crd_arr=K_crd0, seg_arr=K_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vi_34 = CompressedCrdRdScan(crd_arr=V_crd0, seg_arr=V_seg0, debug=debug_sim, statistics=report_stats)
    intersecti_31 = Intersect2(debug=debug_sim, statistics=report_stats)
    intersecti2_31 = Intersect2(debug=debug_sim, statistics=report_stats)
    intersecti3_31 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=Q_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Qk_30 = CompressedCrdRdScan(crd_arr=Q_crd1, seg_arr=Q_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Qj_22 = CompressedCrdRdScan(crd_arr=Q_crd2, seg_arr=Q_seg2, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_3 = CompressWrScan(seg_size=Q_shape[0] + 1, size=Q_shape[0] * Q_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_k_28 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Kk_25 = Repeat(debug=debug_sim, statistics=report_stats)
    repeat_Vk_26 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Kj_23 = CompressedCrdRdScan(crd_arr=K_crd1, seg_arr=K_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vj_24 = CompressedCrdRdScan(crd_arr=V_crd1, seg_arr=V_seg1, debug=debug_sim, statistics=report_stats)
    intersectj_21 = Intersect2(debug=debug_sim, statistics=report_stats)
    intersectj2_21 = Intersect2(debug=debug_sim, statistics=report_stats)
    intersectj3_21 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Qm_18 = CompressedCrdRdScan(crd_arr=Q_crd3, seg_arr=Q_seg3, debug=debug_sim, statistics=report_stats)
    fiberlookup_Km_19 = CompressedCrdRdScan(crd_arr=K_crd2, seg_arr=K_seg2, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vm_20 = CompressedCrdRdScan(crd_arr=V_crd2, seg_arr=V_seg2, debug=debug_sim, statistics=report_stats)
    intersectm_17 = Intersect2(debug=debug_sim, statistics=report_stats)
    intersectm2_17 = Intersect2(debug=debug_sim, statistics=report_stats)
    intersectm3_17 = Intersect2(debug=debug_sim, statistics=report_stats)
    crddrop_11 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberlookup_Kl_15 = CompressedCrdRdScan(crd_arr=K_crd3, seg_arr=K_seg3, debug=debug_sim, statistics=report_stats)
    fiberlookup_Vl_16 = CompressedCrdRdScan(crd_arr=V_crd3, seg_arr=V_seg3, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_2 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[1] + 1, size=Q_shape[0] * Q_shape[1] * Q_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X3_1 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[1] * Q_shape[2] + 1, size=Q_shape[0] * Q_shape[1] * Q_shape[2] * Q_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    intersectl_14 = Intersect2(debug=debug_sim, statistics=report_stats)
    repsiggen_l_13 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    arrayvals_K_9 = Array(init_arr=K_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_V_10 = Array(init_arr=V_vals, debug=debug_sim, statistics=report_stats)
    repeat_Ql_12 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_Q_8 = Array(init_arr=Q_vals, debug=debug_sim, statistics=report_stats)
    mul_7 = Multiply2(debug=debug_sim, statistics=report_stats)
    mul_6 = Multiply2(debug=debug_sim, statistics=report_stats)
    reduce_5 = Reduce(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * Q_shape[0] * Q_shape[1] * Q_shape[2] * Q_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_Q = [0, 'D']
    in_ref_K = [0, 'D']
    in_ref_V = [0, 'D']
    done = False
    time_cnt = 0

    vl = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_Q) > 0:
            fiberlookup_Qi_32.set_in_ref(in_ref_Q.pop(0))
        if len(in_ref_K) > 0:
            fiberlookup_Ki_33.set_in_ref(in_ref_K.pop(0))
        if len(in_ref_V) > 0:
            fiberlookup_Vi_34.set_in_ref(in_ref_V.pop(0))
        intersecti_31.set_in1(fiberlookup_Qi_32.out_ref(), fiberlookup_Qi_32.out_crd())
        intersecti_31.set_in2(fiberlookup_Ki_33.out_ref(), fiberlookup_Ki_33.out_crd())
        # intersecti_31.set_in1(fiberlookup_Vi_34.out_ref(), fiberlookup_Vi_34.out_crd())
        intersecti2_31.set_in1(fiberlookup_Vi_34.out_ref(), fiberlookup_Vi_34.out_crd())
        intersecti2_31.set_in2(intersecti_31.out_ref1(), intersecti_31.out_crd())

        intersecti3_31.set_in1(fiberlookup_Vi_34.out_ref(), fiberlookup_Vi_34.out_crd())
        intersecti3_31.set_in2(intersecti_31.out_ref2(), intersecti_31.out_crd())

        fiberwrite_X0_4.set_input(intersecti2_31.out_crd())

        fiberlookup_Qk_30.set_in_ref(intersecti2_31.out_ref2())
        fiberlookup_Qj_22.set_in_ref(fiberlookup_Qk_30.out_ref())
        fiberwrite_X1_3.set_input(fiberlookup_Qk_30.out_crd())
        repsiggen_k_28.set_istream(fiberlookup_Qk_30.out_crd())
        # change below to ref2
        repeat_Kk_25.set_in_ref(intersecti3_31.out_ref2())
        repeat_Kk_25.set_in_repsig(repsiggen_k_28.out_repsig())
        # change below to ref3
        repeat_Vk_26.set_in_ref(intersecti3_31.out_ref1())
        repeat_Vk_26.set_in_repsig(repsiggen_k_28.out_repsig())
        fiberlookup_Kj_23.set_in_ref(repeat_Kk_25.out_ref())
        fiberlookup_Vj_24.set_in_ref(repeat_Vk_26.out_ref())

        intersectj_21.set_in1(fiberlookup_Qj_22.out_ref(), fiberlookup_Qj_22.out_crd())
        intersectj_21.set_in2(fiberlookup_Kj_23.out_ref(), fiberlookup_Kj_23.out_crd())
        # intersectj_21.set_in3(fiberlookup_Qj_22.out_ref(), fiberlookup_Qj_22.out_crd())
        intersectj2_21.set_in1(fiberlookup_Vj_24.out_ref(), fiberlookup_Vj_24.out_crd())
        intersectj2_21.set_in2(intersectj_21.out_ref1(), intersectj_21.out_crd())

        intersectj3_21.set_in1(fiberlookup_Vj_24.out_ref(), fiberlookup_Vj_24.out_crd())
        intersectj3_21.set_in2(intersectj_21.out_ref2(), intersectj_21.out_crd())

        fiberlookup_Qm_18.set_in_ref(intersectj2_21.out_ref2())
        fiberlookup_Km_19.set_in_ref(intersectj3_21.out_ref2())

        vl.append(intersectj3_21.out_ref1())
        print(remove_emptystr(vl))
        #intersectj.out_ref2
        fiberlookup_Vm_20.set_in_ref(intersectj3_21.out_ref1())

        intersectm_17.set_in1(fiberlookup_Qm_18.out_ref(), fiberlookup_Qm_18.out_crd())
        intersectm_17.set_in2(fiberlookup_Km_19.out_ref(), fiberlookup_Km_19.out_crd())
        # intersectm_17.set_in3(fiberlookup_Vm_20.out_ref(), fiberlookup_Vm_20.out_crd())
        intersectm2_17.set_in2(fiberlookup_Vm_20.out_ref(), fiberlookup_Vm_20.out_crd())
        intersectm2_17.set_in1(intersectm_17.out_ref1(), intersectm_17.out_crd())

        intersectm3_17.set_in2(fiberlookup_Vm_20.out_ref(), fiberlookup_Vm_20.out_crd())
        intersectm3_17.set_in1(intersectm_17.out_ref2(), intersectm_17.out_crd())

        # crddrop_11.set_outer_crd(intersectj_21.out_crd())
        crddrop_11.set_outer_crd(intersectj2_21.out_crd())
        crddrop_11.set_inner_crd(intersectm3_17.out_crd())
        fiberlookup_Kl_15.set_in_ref(intersectm3_17.out_ref2())

        fiberlookup_Vl_16.set_in_ref(intersectm3_17.out_ref1())

        intersectl_14.set_in1(fiberlookup_Kl_15.out_ref(), fiberlookup_Kl_15.out_crd())
        intersectl_14.set_in2(fiberlookup_Vl_16.out_ref(), fiberlookup_Vl_16.out_crd())
        repsiggen_l_13.set_istream(intersectl_14.out_crd())
        arrayvals_K_9.set_load(intersectl_14.out_ref1())
        arrayvals_V_10.set_load(intersectl_14.out_ref2())
        # Changed to out_ref2 from out_ref1
        repeat_Ql_12.set_in_ref(intersectm2_17.out_ref2())
        repeat_Ql_12.set_in_repsig(repsiggen_l_13.out_repsig())
        fiberwrite_X2_2.set_input(crddrop_11.out_crd_outer())
        fiberwrite_X3_1.set_input(crddrop_11.out_crd_inner())
        arrayvals_Q_8.set_load(repeat_Ql_12.out_ref())
        mul_7.set_in1(arrayvals_Q_8.out_val())
        mul_7.set_in2(arrayvals_K_9.out_val())
        mul_6.set_in1(mul_7.out_val())
        mul_6.set_in2(arrayvals_V_10.out_val())
        reduce_5.set_in_val(mul_6.out_val())
        fiberwrite_Xvals_0.set_input(reduce_5.out_val())
        fiberlookup_Qi_32.update()

        fiberlookup_Ki_33.update()

        fiberlookup_Vi_34.update()

        intersecti_31.update()
        intersecti2_31.update()
        intersecti3_31.update()
        fiberwrite_X0_4.update()
        fiberlookup_Qk_30.update()
        fiberlookup_Qj_22.update()
        fiberwrite_X1_3.update()
        repsiggen_k_28.update()
        repeat_Kk_25.update()
        repeat_Vk_26.update()
        fiberlookup_Kj_23.update()
        fiberlookup_Vj_24.update()
        intersectj_21.update()
        intersectj2_21.update()
        intersectj3_21.update()
        fiberlookup_Qm_18.update()
        fiberlookup_Km_19.update()
        fiberlookup_Vm_20.update()
        intersectm_17.update()
        intersectm2_17.update()
        intersectm3_17.update()
        crddrop_11.update()
        fiberlookup_Kl_15.update()
        fiberlookup_Vl_16.update()
        intersectl_14.update()
        repsiggen_l_13.update()
        arrayvals_K_9.update()
        arrayvals_V_10.update()
        repeat_Ql_12.update()
        fiberwrite_X2_2.update()
        fiberwrite_X3_1.update()
        arrayvals_Q_8.update()
        mul_7.update()
        mul_6.update()
        reduce_5.update()
        fiberwrite_Xvals_0.update()

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

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_Q_shape"] = Q_shape
    extra_info["tensor_K_shape"] = K_shape
    extra_info["tensor_V_shape"] = V_shape
    sample_dict = fiberlookup_Qi_32.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qi_32" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_31.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_31" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qk_30.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qk_30" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Kk_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Kk_25" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kj_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kj_23" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_21" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qm_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qm_18" + "_" + k] = sample_dict[k]

    sample_dict = intersectm_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectm_17" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ql_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ql_12" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_Q_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_Q_8" + "_" + k] = sample_dict[k]

    sample_dict = reduce_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kl_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kl_15" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_14" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_K_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_K_9" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_V_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_V_10" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vl_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vl_16" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Km_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Km_19" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vm_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vm_20" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Vk_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Vk_26" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vj_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vj_24" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qj_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qj_22" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ki_33.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ki_33" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Vi_34.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Vi_34" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_fused(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "ssss0123")
    samBench(bench, extra_info)
