import pytest
import time
import scipy.sparse
from sam.sim.src.compression import ValDropper
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.crd_masker import RandomDropout, LowerTriangular, UpperTriangular, Diagonal
from sam.sim.src.unary_alu import Max
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
def test_tensor3_fused_feedforward(samBench, frosttname, cast, check_gold, debug_sim, backpressure, depth, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor3_linear_")
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

    C_dirname = os.path.join(formatted_dir, frosttname, "tensor3_linear_transposed")
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

    d_dirname = os.path.join(formatted_dir, frosttname, "tensor3_linear_")
    d_shape_filename = os.path.join(d_dirname, "tensor_d_mode_shape")
    d_shape = read_inputs(d_shape_filename)

    d0_seg_filename = os.path.join(d_dirname, "tensor_d_mode_0_seg")
    d_seg0 = read_inputs(d0_seg_filename)
    d0_crd_filename = os.path.join(d_dirname, "tensor_d_mode_0_crd")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "tensor_d_mode_vals")
    d_vals = read_inputs(d_vals_filename, float)


    fiberlookup_Bj_31 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_dj_32 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    unionj_30 = Union2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X1_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_j_28 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Cj_27 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ck_26 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Ci_20 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X2_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[2], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_k_24 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Cl_14 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_X0_1 = CompressWrScan(seg_size=B_shape[0] * C_shape[2] + 1, size=B_shape[0] * C_shape[2] * C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_i_18 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bk_21 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_dk_22 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_Bi_15 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_di_16 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberlookup_Bl_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    intersectl_12 = Intersect2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repsiggen_l_11 = RepeatSigGen(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    repeat_dl_10 = Repeat(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    mul_6 = Multiply2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    arrayvals_d_9 = Array(init_arr=d_vals, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    add_5 = Add2(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    reduce_4 = Reduce(debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * C_shape[2] * C_shape[0], fill=fill, debug=debug_sim, statistics=report_stats, back_en=backpressure, depth=int(depth))
    crd = CrdHold(debug=debug_sim)
    max_1 = Max(in2=0.0, debug=debug_sim, statistics=report_stats)
    comp_drop_1 = ValDropper(debug=debug_sim, statistics=report_stats)
    drop_2 = CrdDrop(debug=debug_sim, statistics=report_stats)
    drop_3 = CrdDrop(debug=debug_sim, statistics=report_stats)
    drop_prob = 0.4
    dropout = RandomDropout(dimension=2, drop_probability=drop_prob, debug=debug_sim)
    val_stkn_dropper = EmptyFiberStknDrop()
    crd_stkn_dropper = EmptyFiberStknDrop()
    in_ref_B = [0, 'D']
    in_ref_d = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    ref_arr = []
    crd_arr = []

    mul_arr = []
    add_arr = []

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bj_31.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_d) > 0:
            fiberlookup_dj_32.set_in_ref(in_ref_d.pop(0))

        unionj_30.set_in1(fiberlookup_Bj_31.out_ref(), fiberlookup_Bj_31.out_crd())
        unionj_30.set_in2(fiberlookup_dj_32.out_ref(), fiberlookup_dj_32.out_crd())
        repsiggen_j_28.set_istream(unionj_30.out_crd())
        if len(in_ref_C) > 0:
            repeat_Cj_27.set_in_ref(in_ref_C.pop(0))
        repeat_Cj_27.set_in_repsig(repsiggen_j_28.out_repsig())
        fiberlookup_Ck_26.set_in_ref(repeat_Cj_27.out_ref())
        # ref_arr.append(fiberlookup_Ck_26.out_ref())
        # crd_arr.append(fiberlookup_Ck_26.out_crd())
        # print("Ref: ", remove_emptystr(ref_arr))
        # print("Crd: ", remove_emptystr(crd_arr))
        fiberlookup_Ci_20.set_in_ref(fiberlookup_Ck_26.out_ref())
        repsiggen_k_24.set_istream(fiberlookup_Ck_26.out_crd())
        repeat_Bk_21.set_in_ref(unionj_30.out_ref1())
        repeat_Bk_21.set_in_repsig(repsiggen_k_24.out_repsig())
        repeat_dk_22.set_in_ref(unionj_30.out_ref2())
        repeat_dk_22.set_in_repsig(repsiggen_k_24.out_repsig())
        fiberlookup_Cl_14.set_in_ref(fiberlookup_Ci_20.out_ref())
        repsiggen_i_18.set_istream(fiberlookup_Ci_20.out_crd())
        repeat_Bi_15.set_in_ref(repeat_Bk_21.out_ref())
        repeat_Bi_15.set_in_repsig(repsiggen_i_18.out_repsig())
        repeat_di_16.set_in_ref(repeat_dk_22.out_ref())
        repeat_di_16.set_in_repsig(repsiggen_i_18.out_repsig())
        fiberlookup_Bl_13.set_in_ref(repeat_Bi_15.out_ref())
        intersectl_12.set_in1(fiberlookup_Bl_13.out_ref(), fiberlookup_Bl_13.out_crd())
        intersectl_12.set_in2(fiberlookup_Cl_14.out_ref(), fiberlookup_Cl_14.out_crd())
        # repsiggen_l_11.set_istream(intersectl_12.out_crd())
        repsiggen_l_11.set_istream(fiberlookup_Ck_26.out_crd())
        arrayvals_B_7.set_load(intersectl_12.out_ref1())
        arrayvals_C_8.set_load(intersectl_12.out_ref2())
        arrayvals_d_9.set_load(repeat_di_16.out_ref())
        mul_6.set_in1(arrayvals_B_7.out_val())
        mul_6.set_in2(arrayvals_C_8.out_val())

        reduce_4.set_in_val(mul_6.out_val())
        add_5.set_in1(reduce_4.out_val())
        add_5.set_in2(arrayvals_d_9.out_val())

        max_1.set_in1(add_5.out_val())

        comp_drop_1.set_val(max_1.out_val())
        comp_drop_1.set_crd(fiberlookup_Ci_20.out_crd())

        # TODO: Should these droppers be inside the ValDropper block? Relu can generate empty fibers
        val_stkn_dropper.set_in_val(comp_drop_1.out_val())
        crd_stkn_dropper.set_in_val(comp_drop_1.out_crd())
        drop_2.set_outer_crd(fiberlookup_Ck_26.out_crd())
        drop_2.set_inner_crd(comp_drop_1.out_crd())
        crd.set_outer_crd(drop_2.out_crd_outer())
        crd.set_inner_crd(crd_stkn_dropper.out_val())

        dropout.set_inner_crd(crd.out_crd_inner())
        dropout.set_outer_crd(crd.out_crd_outer())
        dropout.set_inner_ref(val_stkn_dropper.out_val())

        drop_3.set_outer_crd(unionj_30.out_crd())
        drop_3.set_inner_crd(dropout.out_crd(1))

        fiberwrite_Xvals_0.set_input(dropout.out_ref())
        fiberwrite_X1_3.set_input(drop_3.out_crd_outer())
        fiberwrite_X2_2.set_input(dropout.out_crd(1))
        fiberwrite_X0_1.set_input(dropout.out_crd(0))

        ref_arr.append(dropout.out_crd(0))
        print(remove_emptystr(ref_arr))

        fiberlookup_Bj_31.update()

        fiberlookup_dj_32.update()

        unionj_30.update()
        
        repsiggen_j_28.update()
        repeat_Cj_27.update()
        fiberlookup_Ck_26.update()
        fiberlookup_Ci_20.update()
        
        repsiggen_k_24.update()
        repeat_Bk_21.update()
        repeat_dk_22.update()
        fiberlookup_Cl_14.update()

        repsiggen_i_18.update()
        repeat_Bi_15.update()
        repeat_di_16.update()
        fiberlookup_Bl_13.update()
        intersectl_12.update()
        repsiggen_l_11.update()
        arrayvals_B_7.update()
        arrayvals_C_8.update()
        repeat_dl_10.update()
        arrayvals_d_9.update()
        mul_6.update()
        # add_5.update()
        # reduce_4.update()
        reduce_4.update()
        add_5.update()

        max_1.update()
        comp_drop_1.update()
        val_stkn_dropper.update()
        crd_stkn_dropper.update()
        drop_2.update()
        crd.update()
        dropout.update()
        drop_3.update()

        fiberwrite_X1_3.update()
        fiberwrite_X2_2.update()
        fiberwrite_X0_1.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X1_3.out_done() and fiberwrite_X2_2.out_done() and fiberwrite_X0_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X1_3.autosize()
    fiberwrite_X2_2.autosize()
    fiberwrite_X0_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X1_3.get_arr(), fiberwrite_X2_2.get_arr(), fiberwrite_X0_1.get_arr()]
    out_segs = [fiberwrite_X1_3.get_seg_arr(), fiberwrite_X2_2.get_seg_arr(), fiberwrite_X0_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    print("segs:", out_segs)
    print("crds:", out_crds)
    print("vals:", out_vals)

    pytest.set_trace()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_d_shape"] = d_shape
    sample_dict = fiberlookup_Bj_31.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_31" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Cj_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cj_27" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_26" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_2" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bk_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bk_21" + "_" + k] = sample_dict[k]

    sample_dict = repeat_Bi_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bi_15" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_13" + "_" + k] = sample_dict[k]

    sample_dict = intersectl_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_12" + "_" + k] = sample_dict[k]

    sample_dict = repeat_dl_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_dl_10" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_d_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_d_9" + "_" + k] = sample_dict[k]

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

    sample_dict = repeat_dk_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_dk_22" + "_" + k] = sample_dict[k]

    sample_dict = repeat_di_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_di_16" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_20" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_1" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cl_14" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_dj_32.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_dj_32" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_fused_feedforward(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, dropout.random_dropped, "sss120")
    samBench(bench, extra_info)
