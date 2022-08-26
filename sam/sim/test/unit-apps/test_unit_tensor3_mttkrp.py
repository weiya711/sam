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

other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


def test_unit_tensor3_mttkrp(samBench, frosttname, check_gold, debug_sim, fill=0):

    B_dirname = os.path.join(cwd, "tmp_mat")
    B_shape = (4, 4, 4)

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

    C_dirname = B_dirname
    # C_fname = [f for f in os.listdir(C_dirname) if frosttname + "-mat_mode1_mttkrp" in f]
    # assert len(C_fname) == 1, "Should only have one 'other' folder that matches"
    # C_fname = C_fname[0]
    # C_dirname = os.path.join(C_dirname, C_fname)

    C_shape_filename = os.path.join(C_dirname, "shape")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "tensor_C_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_C_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

    D_dirname = C_dirname
    # D_dirname = os.path.join(formatted_dir, frosttname, "other")
    # D_fname = [f for f in os.listdir(D_dirname) if frosttname + "-mat_mode2_mttkrp" in f]
    # assert len(D_fname) == 1, "Should only have one 'other' folder that matches"
    # D_fname = D_fname[0]
    # D_dirname = os.path.join(D_dirname, D_fname)

    D_shape_filename = os.path.join(D_dirname, "shape")
    D_shape = read_inputs(D_shape_filename)

    D0_seg_filename = os.path.join(D_dirname, "tensor_D_mode_0_seg")
    D_seg0 = read_inputs(D0_seg_filename)
    D0_crd_filename = os.path.join(D_dirname, "tensor_D_mode_0_crd")
    D_crd0 = read_inputs(D0_crd_filename)

    D1_seg_filename = os.path.join(D_dirname, "tensor_D_mode_1_seg")
    D_seg1 = read_inputs(D1_seg_filename)
    D1_crd_filename = os.path.join(D_dirname, "tensor_D_mode_1_crd")
    D_crd1 = read_inputs(D1_crd_filename)

    D_vals_filename = os.path.join(D_dirname, "tensor_D_mode_vals")
    D_vals = read_inputs(D_vals_filename, float)

    fiberlookup_Bi_31 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=len(B_crd0), fill=fill, debug=debug_sim)
    repsiggen_i_29 = RepeatSigGen(debug=debug_sim)
    repeat_Ci_26 = Repeat(debug=debug_sim)
    repeat_Di_27 = Repeat(debug=debug_sim)
    fiberlookup_Cj_24 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    fiberlookup_Dj_25 = CompressedCrdRdScan(crd_arr=D_crd0, seg_arr=D_seg0, debug=debug_sim)
    intersectj_23 = Intersect2(debug=debug_sim)
    fiberlookup_Ck_19 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    fiberwrite_X1_1 = CompressWrScan(seg_size=len(B_crd0) + 1, size=len(B_crd0) * len(C_crd0), fill=fill, debug=debug_sim)
    repsiggen_j_21 = RepeatSigGen(debug=debug_sim)
    repeat_Bj_20 = Repeat(debug=debug_sim)
    fiberlookup_Bk_18 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    intersectk_17 = Intersect2(debug=debug_sim)
    repsiggen_k_16 = RepeatSigGen(debug=debug_sim)
    fiberlookup_Bl_13 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim)
    repeat_Dk_15 = Repeat(debug=debug_sim)
    fiberlookup_Dl_14 = CompressedCrdRdScan(crd_arr=D_crd1, seg_arr=D_seg1, debug=debug_sim)
    intersectl_12 = Intersect2(debug=debug_sim)
    repsiggen_l_11 = RepeatSigGen(debug=debug_sim)
    arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_D_9 = Array(init_arr=D_vals, debug=debug_sim)
    repeat_Cl_10 = Repeat(debug=debug_sim)
    arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim)
    mul_6 = Multiply2(debug=debug_sim)
    mul_5 = Multiply2(debug=debug_sim)
    reduce_4 = Reduce(debug=debug_sim)
    reduce_3 = Reduce(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * len(B_crd0) * len(C_crd0), fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_D = [0, 'D']
    done = False
    time_cnt = 0

    di = []
    dj_ref = []
    dj_crd = []
    intj = []
    dk = []
    dl_ref = []
    dl_crd = []
    intl = []
    valb = []
    valc = []
    vald = []
    mul1 = []
    mul2 = []
    red1 = []
    red2 = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_31.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_31.update()

        fiberwrite_X0_2.set_input(fiberlookup_Bi_31.out_crd())
        fiberwrite_X0_2.update()

        repsiggen_i_29.set_istream(fiberlookup_Bi_31.out_crd())
        repsiggen_i_29.update()

        if len(in_ref_C) > 0:
            repeat_Ci_26.set_in_ref(in_ref_C.pop(0))
        repeat_Ci_26.set_in_repsig(repsiggen_i_29.out_repsig())
        repeat_Ci_26.update()

        if len(in_ref_D) > 0:
            repeat_Di_27.set_in_ref(in_ref_D.pop(0))
        repeat_Di_27.set_in_repsig(repsiggen_i_29.out_repsig())
        repeat_Di_27.update()
        di.append(repeat_Di_27.out_ref())

        fiberlookup_Cj_24.set_in_ref(repeat_Ci_26.out_ref())
        fiberlookup_Cj_24.update()

        fiberlookup_Dj_25.set_in_ref(repeat_Di_27.out_ref())
        fiberlookup_Dj_25.update()
        dj_crd.append(fiberlookup_Dj_25.out_crd())
        dj_ref.append(fiberlookup_Dj_25.out_ref())

        intersectj_23.set_in1(fiberlookup_Cj_24.out_ref(), fiberlookup_Cj_24.out_crd())
        intersectj_23.set_in2(fiberlookup_Dj_25.out_ref(), fiberlookup_Dj_25.out_crd())
        intersectj_23.update()
        intj.append(intersectj_23.out_crd())
    
        fiberlookup_Ck_19.set_in_ref(intersectj_23.out_ref1())
        fiberlookup_Ck_19.update()

        fiberwrite_X1_1.set_input(intersectj_23.out_crd())
        fiberwrite_X1_1.update()

        repsiggen_j_21.set_istream(intersectj_23.out_crd())
        repsiggen_j_21.update()

        repeat_Bj_20.set_in_ref(fiberlookup_Bi_31.out_ref())
        repeat_Bj_20.set_in_repsig(repsiggen_j_21.out_repsig())
        repeat_Bj_20.update()

        fiberlookup_Bk_18.set_in_ref(repeat_Bj_20.out_ref())
        fiberlookup_Bk_18.update()

        intersectk_17.set_in1(fiberlookup_Bk_18.out_ref(), fiberlookup_Bk_18.out_crd())
        intersectk_17.set_in2(fiberlookup_Ck_19.out_ref(), fiberlookup_Ck_19.out_crd())
        intersectk_17.update()

        repsiggen_k_16.set_istream(intersectk_17.out_crd())
        repsiggen_k_16.update()

        fiberlookup_Bl_13.set_in_ref(intersectk_17.out_ref1())
        fiberlookup_Bl_13.update()

        repeat_Dk_15.set_in_ref(intersectj_23.out_ref2())
        repeat_Dk_15.set_in_repsig(repsiggen_k_16.out_repsig())
        repeat_Dk_15.update()
        dk.append(repeat_Dk_15.out_ref())

        fiberlookup_Dl_14.set_in_ref(repeat_Dk_15.out_ref())
        fiberlookup_Dl_14.update()
        dl_ref.append(fiberlookup_Dl_14.out_ref())
        dl_crd.append(fiberlookup_Dl_14.out_crd())

        intersectl_12.set_in1(fiberlookup_Dl_14.out_ref(), fiberlookup_Dl_14.out_crd())
        intersectl_12.set_in2(fiberlookup_Bl_13.out_ref(), fiberlookup_Bl_13.out_crd())
        intersectl_12.update()
        intl.append(intersectl_12.out_crd())

        repsiggen_l_11.set_istream(intersectl_12.out_crd())
        repsiggen_l_11.update()

        arrayvals_B_7.set_load(intersectl_12.out_ref2())
        arrayvals_B_7.update()
        valb.append(arrayvals_B_7.out_val())

        arrayvals_D_9.set_load(intersectl_12.out_ref1())
        arrayvals_D_9.update()
        vald.append(arrayvals_D_9.out_val())

        repeat_Cl_10.set_in_ref(intersectk_17.out_ref2())
        repeat_Cl_10.set_in_repsig(repsiggen_l_11.out_repsig())
        repeat_Cl_10.update()

        arrayvals_C_8.set_load(repeat_Cl_10.out_ref())
        arrayvals_C_8.update()
        valc.append(arrayvals_C_8.out_val())

        mul_6.set_in1(arrayvals_C_8.out_val())
        mul_6.set_in2(arrayvals_B_7.out_val())
        mul_6.update()
        mul1.append(mul_6.out_val())

        mul_5.set_in1(mul_6.out_val())
        mul_5.set_in2(arrayvals_D_9.out_val())
        mul_5.update()
        mul2.append(mul_5.out_val())

        reduce_4.set_in_val(mul_5.out_val())
        reduce_4.update()
        red1.append(reduce_4.out_val())

        reduce_3.set_in_val(reduce_4.out_val())
        reduce_3.update()
        red2.append(reduce_3.out_val())

        fiberwrite_Xvals_0.set_input(reduce_3.out_val())
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_2.autosize()
    fiberwrite_X1_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
    out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    if debug_sim:
        print("di", remove_emptystr(di))
        print("dj_ref", remove_emptystr(dj_ref))
        print("dj_crd", remove_emptystr(dj_crd))
        print("intj", remove_emptystr(intj))
        print("dk", remove_emptystr(dk))
        print("dl_ref", remove_emptystr(dl_ref))
        print("dl_crd", remove_emptystr(dl_crd))
        print("intl", remove_emptystr(intl))

        print("valb", remove_emptystr(valb))
        print("valc", remove_emptystr(valc))
        print("vald", remove_emptystr(vald))
        print("mul1", remove_emptystr(mul1))
        print("mul2", remove_emptystr(mul2))
        print("red1", remove_emptystr(red1))
        print("red2", remove_emptystr(red2))


    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_D_shape"] = D_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)
    extra_info["tensor_D/nnz"] = len(D_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Ci_26.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_26" + "/" + k] = sample_dict[k]

    sample_dict = intersectj_23.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_23" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Bj_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bj_20" + "/" + k] = sample_dict[k]

    sample_dict = intersectk_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_17" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Dk_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Dk_15" + "/" + k] = sample_dict[k]

    sample_dict = intersectl_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_12" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Cl_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cl_10" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_8" + "/" + k] = sample_dict[k]

    sample_dict = reduce_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_4" + "/" + k] = sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_7" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_D_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_D_9" + "/" + k] = sample_dict[k]

    sample_dict = repeat_Di_27.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Di_27" + "/" + k] = sample_dict[k]

    sample_dict = mul_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_6" + "/" + k] = sample_dict[k]

    sample_dict = mul_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul_5" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi_31.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_31" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cj_24.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_24" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Dj_25.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Dj_25" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ck_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ck_19" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_18.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_18" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bl_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bl_13" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Dl_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Dl_14" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_mttkrp(frosttname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
