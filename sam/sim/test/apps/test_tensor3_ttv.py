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
def test_tensor3_ttv(samBench, ssname, check_gold, debug_sim, fill=0):
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

    c_dirname = os.path.join(formatted_dir, ssname, "dummy", "s0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "c0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "c0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_Bi_17 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Bj_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_15 = RepeatSigGen(debug=debug_sim)
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim)
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    repsiggen_j_11 = RepeatSigGen(debug=debug_sim)
    repeat_ci_14 = Repeat(debug=debug_sim)
    repeat_cj_10 = Repeat(debug=debug_sim)
    fiberlookup_ck_9 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    intersectk_7 = Intersect2(debug=debug_sim)
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_c_6 = Array(init_arr=c_vals, debug=debug_sim)
    mul_4 = Multiply2(debug=debug_sim)
    reduce_3 = Reduce(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_17.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_17.update()

        fiberlookup_Bj_13.set_in_ref(fiberlookup_Bi_17.out_ref())
        fiberlookup_Bj_13.update()

        fiberwrite_X0_2.set_input(fiberlookup_Bi_17.out_crd())
        fiberwrite_X0_2.update()

        repsiggen_i_15.set_istream(fiberlookup_Bi_17.out_crd())
        repsiggen_i_15.update()

        if len(in_ref_c) > 0:
            repeat_ci_14.set_in_ref(in_ref_c.pop(0))
        repeat_ci_14.set_in_repsig(repsiggen_i_15.out_repsig())
        repeat_ci_14.update()

        fiberlookup_Bk_8.set_in_ref(fiberlookup_Bj_13.out_ref())
        fiberlookup_Bk_8.update()

        fiberwrite_X1_1.set_input(fiberlookup_Bj_13.out_crd())
        fiberwrite_X1_1.update()

        repsiggen_j_11.set_istream(fiberlookup_Bj_13.out_crd())
        repsiggen_j_11.update()

        repeat_cj_10.set_in_ref(repeat_ci_14.out_ref())
        repeat_cj_10.set_in_repsig(repsiggen_j_11.out_repsig())
        repeat_cj_10.update()

        fiberlookup_ck_9.set_in_ref(repeat_cj_10.out_ref())
        fiberlookup_ck_9.update()

        intersectk_7.set_in1(fiberlookup_ck_9.out_ref(), fiberlookup_ck_9.out_crd())
        intersectk_7.set_in2(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd())
        intersectk_7.update()

        arrayvals_B_5.set_load(intersectk_7.out_ref2())
        arrayvals_B_5.update()

        arrayvals_c_6.set_load(intersectk_7.out_ref1())
        arrayvals_c_6.update()

        mul_4.set_in1(arrayvals_B_5.out_val())
        mul_4.set_in2(arrayvals_c_6.out_val())
        mul_4.update()

        reduce_3.set_in_val(mul_4.out_val())
        reduce_3.update()

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
    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_c_shape"] = c_shape
    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_ci_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ci_14" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_cj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_cj_10" + "_" + k] =  sample_dict[k]

    sample_dict = intersectk_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_7" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "_" + k] =  sample_dict[k]

    sample_dict = reduce_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_3" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_c_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_6" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] =  sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_ttv(ssname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)