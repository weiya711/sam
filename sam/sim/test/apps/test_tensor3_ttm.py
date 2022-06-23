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
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default = os.path.join(cwd,'mode-formats'))

# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_ttm(samBench, ssname, check_gold, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "dummy", "sss012")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B2_seg_filename = os.path.join(B_dirname, "B2_seg.txt")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "B2_crd.txt")
    B_crd2 = read_inputs(B2_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, ssname, "dummy", "ss01")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_22 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Bj_18 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_20 = RepeatSigGen(debug=debug_sim)
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim)
    repeat_Ci_19 = Repeat(debug=debug_sim)
    repeat_Cj_15 = Repeat(debug=debug_sim)
    fiberlookup_Ck_14 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    fiberlookup_Cl_10 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    fiberwrite_X2_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * C_shape[0], fill=fill, debug=debug_sim)
    repsiggen_k_12 = RepeatSigGen(debug=debug_sim)
    repeat_Bk_11 = Repeat(debug=debug_sim)
    fiberlookup_Bl_9 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim)
    intersectl_8 = Intersect2(debug=debug_sim)
    arrayvals_B_6 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim)
    mul_5 = Multiply2(debug=debug_sim)
    reduce_4 = Reduce(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * C_shape[0], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_22.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_22.update()

        fiberlookup_Bj_18.set_in_ref(fiberlookup_Bi_22.out_ref())
        fiberlookup_Bj_18.update()

        fiberwrite_X0_3.set_input(fiberlookup_Bi_22.out_crd())
        fiberwrite_X0_3.update()

        repsiggen_i_20.set_istream(fiberlookup_Bi_22.out_crd())
        repsiggen_i_20.update()

        if len(in_ref_C) > 0:
            repeat_Ci_19.set_in_ref(in_ref_C.pop(0))
        repeat_Ci_19.set_in_repsig(repsiggen_i_20.out_repsig())
        repeat_Ci_19.update()

        fiberwrite_X1_2.set_input(fiberlookup_Bj_18.out_crd())
        fiberwrite_X1_2.update()

        repsiggen_j_16.set_istream(fiberlookup_Bj_18.out_crd())
        repsiggen_j_16.update()

        repeat_Cj_15.set_in_ref(repeat_Ci_19.out_ref())
        repeat_Cj_15.set_in_repsig(repsiggen_j_16.out_repsig())
        repeat_Cj_15.update()

        fiberlookup_Ck_14.set_in_ref(repeat_Cj_15.out_ref())
        fiberlookup_Ck_14.update()

        fiberlookup_Cl_10.set_in_ref(fiberlookup_Ck_14.out_ref())
        fiberlookup_Cl_10.update()

        fiberwrite_X2_1.set_input(fiberlookup_Ck_14.out_crd())
        fiberwrite_X2_1.update()

        repsiggen_k_12.set_istream(fiberlookup_Ck_14.out_crd())
        repsiggen_k_12.update()

        repeat_Bk_11.set_in_repsig(repsiggen_k_12.out_repsig())
        repeat_Bk_11.set_in_ref(fiberlookup_Bj_18.out_ref())
        repeat_Bk_11.update()

        fiberlookup_Bl_9.set_in_ref(repeat_Bk_11.out_ref())
        fiberlookup_Bl_9.update()

        intersectl_8.set_in1(fiberlookup_Bl_9.out_ref(), fiberlookup_Bl_9.out_crd())
        intersectl_8.set_in2(fiberlookup_Cl_10.out_ref(), fiberlookup_Cl_10.out_crd())
        intersectl_8.update()

        arrayvals_B_6.set_load(intersectl_8.out_ref1())
        arrayvals_B_6.update()

        arrayvals_C_7.set_load(intersectl_8.out_ref2())
        arrayvals_C_7.update()

        mul_5.set_in1(arrayvals_B_6.out_val())
        mul_5.set_in2(arrayvals_C_7.out_val())
        mul_5.update()

        reduce_4.set_in_val(mul_5.out_val())
        reduce_4.update()

        fiberwrite_Xvals_0.set_input(reduce_4.out_val())
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()
    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_Ci_19.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Ci_19" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_Cj_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Cj_15" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_Bk_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_Bk_11" + "_" + k] =  sample_dict[k]

    sample_dict = intersectl_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectl_8" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_B_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_6" + "_" + k] =  sample_dict[k]

    sample_dict = reduce_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_4" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] =  sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_ttm(ssname, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)