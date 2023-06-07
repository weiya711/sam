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
def test_tensor4_mult1(samBench, frosttname, cast, check_gold, debug_sim, backpressure, depth, report_stats, fill=0):
    test_name = "tensor4_fused_mul_T4"
    Q_dirname = os.path.join(formatted_dir, frosttname, test_name)
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

    K_dirname = os.path.join(formatted_dir, frosttname, test_name)
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


    fiberlookup_Qi_29 = CompressedCrdRdScan(crd_arr=Q_crd0, seg_arr=Q_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ki_30 = CompressedCrdRdScan(crd_arr=K_crd0, seg_arr=K_seg0, debug=debug_sim, statistics=report_stats)
    intersecti_28 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Qj_26 = CompressedCrdRdScan(crd_arr=Q_crd2, seg_arr=Q_seg2, debug=debug_sim, statistics=report_stats)
    fiberlookup_Kj_27 = CompressedCrdRdScan(crd_arr=K_crd2, seg_arr=K_seg2, debug=debug_sim, statistics=report_stats)
    intersectj_25 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Qk_24 = CompressedCrdRdScan(crd_arr=Q_crd1, seg_arr=Q_seg1, debug=debug_sim, statistics=report_stats)
    repsiggen_k_22 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Kk_21 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Kl_20 = CompressedCrdRdScan(crd_arr=K_crd1, seg_arr=K_seg1, debug=debug_sim, statistics=report_stats)
    fiberlookup_Km_16 = CompressedCrdRdScan(crd_arr=K_crd3, seg_arr=K_seg3, debug=debug_sim, statistics=report_stats)
    repsiggen_l_18 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ql_17 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Qm_15 = CompressedCrdRdScan(crd_arr=Q_crd3, seg_arr=Q_seg3, debug=debug_sim, statistics=report_stats)
    intersectm_14 = Intersect2(debug=debug_sim, statistics=report_stats)
    crddrop_13 = CrdDrop(debug=debug_sim, statistics=report_stats)
    arrayvals_Q_8 = Array(init_arr=Q_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_K_9 = Array(init_arr=K_vals, debug=debug_sim, statistics=report_stats)
    crddrop_12 = CrdDrop(debug=debug_sim, statistics=report_stats)
    mul_7 = Multiply2(debug=debug_sim, statistics=report_stats)
    crddrop_11 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_X3_1 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[2] * Q_shape[1] + 1, size=Q_shape[0] * Q_shape[2] * Q_shape[1] * K_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    reduce_6 = Reduce(debug=debug_sim, statistics=report_stats)
    crddrop_10 = CrdDrop(debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_2 = CompressWrScan(seg_size=Q_shape[0] * Q_shape[2] + 1, size=Q_shape[0] * Q_shape[2] * Q_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * Q_shape[0] * Q_shape[2] * Q_shape[1] * K_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_4 = CompressWrScan(seg_size=2, size=Q_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_3 = CompressWrScan(seg_size=Q_shape[0] + 1, size=Q_shape[0] * Q_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_Q = [0, 'D']
    in_ref_K = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_Q) > 0:
            fiberlookup_Qi_29.set_in_ref(in_ref_Q.pop(0))
        if len(in_ref_K) > 0:
            fiberlookup_Ki_30.set_in_ref(in_ref_K.pop(0))
        intersecti_28.set_in1(fiberlookup_Qi_29.out_ref(), fiberlookup_Qi_29.out_crd())
        intersecti_28.set_in2(fiberlookup_Ki_30.out_ref(), fiberlookup_Ki_30.out_crd())
        fiberlookup_Qj_26.set_in_ref(intersecti_28.out_ref1())
        fiberlookup_Kj_27.set_in_ref(intersecti_28.out_ref2())
        intersectj_25.set_in1(fiberlookup_Qj_26.out_ref(), fiberlookup_Qj_26.out_crd())
        intersectj_25.set_in2(fiberlookup_Kj_27.out_ref(), fiberlookup_Kj_27.out_crd())
        fiberlookup_Qk_24.set_in_ref(intersectj_25.out_ref1())
        repsiggen_k_22.set_istream(fiberlookup_Qk_24.out_crd())
        repeat_Kk_21.set_in_ref(intersectj_25.out_ref2())
        repeat_Kk_21.set_in_repsig(repsiggen_k_22.out_repsig())
        fiberlookup_Kl_20.set_in_ref(repeat_Kk_21.out_ref())
        fiberlookup_Km_16.set_in_ref(fiberlookup_Kl_20.out_ref())
        repsiggen_l_18.set_istream(fiberlookup_Kl_20.out_crd())
        repeat_Ql_17.set_in_ref(fiberlookup_Qk_24.out_ref())
        repeat_Ql_17.set_in_repsig(repsiggen_l_18.out_repsig())
        fiberlookup_Qm_15.set_in_ref(repeat_Ql_17.out_ref())
        intersectm_14.set_in1(fiberlookup_Qm_15.out_ref(), fiberlookup_Qm_15.out_crd())
        intersectm_14.set_in2(fiberlookup_Km_16.out_ref(), fiberlookup_Km_16.out_crd())
        crddrop_13.set_outer_crd(fiberlookup_Kl_20.out_crd())
        crddrop_13.set_inner_crd(intersectm_14.out_crd())
        arrayvals_Q_8.set_load(intersectm_14.out_ref1())
        arrayvals_K_9.set_load(intersectm_14.out_ref2())
        mul_7.set_in1(arrayvals_Q_8.out_val())
        mul_7.set_in2(arrayvals_K_9.out_val())
        reduce_6.set_in_val(mul_7.out_val())
        fiberwrite_Xvals_0.set_input(reduce_6.out_val())
        crddrop_12.set_outer_crd(fiberlookup_Qk_24.out_crd())
        crddrop_12.set_inner_crd(crddrop_13.out_crd_outer())
        crddrop_11.set_outer_crd(intersectj_25.out_crd())
        crddrop_11.set_inner_crd(crddrop_12.out_crd_outer())
        crddrop_10.set_outer_crd(intersecti_28.out_crd())
        crddrop_10.set_inner_crd(crddrop_11.out_crd_outer())
        # fiberwrite_X0_4.set_input(crddrop_10.out_crd_outer())
        # fiberwrite_X1_3.set_input(crddrop_10.out_crd_inner())
        # fiberwrite_X2_2.set_input(crddrop_11.out_crd_inner())
        # fiberwrite_X3_1.set_input(crddrop_12.out_crd_inner())
        fiberwrite_X0_4.set_input(intersecti_28.out_crd())
        fiberwrite_X1_3.set_input(intersectj_25.out_crd())
        fiberwrite_X2_2.set_input(fiberlookup_Qk_24.out_crd())
        fiberwrite_X3_1.set_input(fiberlookup_Kl_20.out_crd())
        fiberlookup_Qi_29.update()

        fiberlookup_Ki_30.update()

        intersecti_28.update()
        fiberlookup_Qj_26.update()
        fiberlookup_Kj_27.update()
        intersectj_25.update()
        fiberlookup_Qk_24.update()
        repsiggen_k_22.update()
        repeat_Kk_21.update()
        fiberlookup_Kl_20.update()
        fiberlookup_Km_16.update()
        repsiggen_l_18.update()
        repeat_Ql_17.update()
        fiberlookup_Qm_15.update()
        intersectm_14.update()
        crddrop_13.update()
        arrayvals_Q_8.update()
        arrayvals_K_9.update()
        mul_7.update()
        reduce_6.update()
        fiberwrite_Xvals_0.update()
        crddrop_12.update()
        crddrop_11.update()
        fiberwrite_X3_1.update()
        crddrop_10.update()
        fiberwrite_X0_4.update()
        fiberwrite_X1_3.update()
        fiberwrite_X2_2.update()

        done = fiberwrite_X0_4.out_done() and fiberwrite_X1_3.out_done() and fiberwrite_X2_2.out_done() and fiberwrite_X3_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1
        if time_cnt % 100000 == 0:
            print("Cycle: ", time_cnt)

    fiberwrite_X0_4.autosize()
    fiberwrite_X1_3.autosize()
    fiberwrite_X2_2.autosize()
    fiberwrite_X3_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_4.get_arr(), fiberwrite_X1_3.get_arr(), fiberwrite_X2_2.get_arr(), fiberwrite_X3_1.get_arr()]
    out_segs = [fiberwrite_X0_4.get_seg_arr(), fiberwrite_X1_3.get_seg_arr(), fiberwrite_X2_2.get_seg_arr(), fiberwrite_X3_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print(out_segs)
    print(out_crds)
    print(out_vals)
    print("# cycles: ", time_cnt)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_Q_shape"] = Q_shape
    extra_info["tensor_K_shape"] = K_shape
    sample_dict = fiberlookup_Qi_29.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qi_29" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_28.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_28" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_10" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qj_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qj_26" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_25" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_11" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qk_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qk_24" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_12" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Kk_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Kk_21" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kl_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kl_20" + "_" + k] = sample_dict[k]

    sample_dict = crddrop_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_13" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Ql_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ql_17" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Qm_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Qm_15" + "_" + k] = sample_dict[k]

    sample_dict = intersectm_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectm_14" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_Q_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_Q_8" + "_" + k] = sample_dict[k]

    sample_dict = reduce_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_K_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_K_9" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Km_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Km_16" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Kj_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Kj_27" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ki_30.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ki_30" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_multiply1(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, test_name)
    samBench(bench, extra_info)
