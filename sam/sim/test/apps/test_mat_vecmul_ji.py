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
other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_mat_vecmul_ji(samBench, ssname, check_gold, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss10")
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

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    c_dirname = os.path.join(formatted_dir, ssname, "other", "s0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "c0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "c0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_Bj_11 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberlookup_cj_12 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    intersectj_10 = Intersect2(debug=debug_sim)
    fiberlookup_Bi_9 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim)
    repsiggen_i_7 = RepeatSigGen(debug=debug_sim)
    repeat_ci_6 = Repeat(debug=debug_sim)
    arrayvals_c_5 = Array(init_arr=c_vals, debug=debug_sim)
    mul_3 = Multiply2(debug=debug_sim)
    spaccumulator1_2 = SparseAccumulator1(debug=debug_sim)
    spaccumulator1_2_drop_crd_inner = StknDrop(debug=debug_sim)
    spaccumulator1_2_drop_crd_outer = StknDrop(debug=debug_sim)
    spaccumulator1_2_drop_val = StknDrop(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * B_shape[0], fill=fill, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bj_11.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_c) > 0:
            fiberlookup_cj_12.set_in_ref(in_ref_c.pop(0))
        intersectj_10.set_in1(fiberlookup_Bj_11.out_ref(), fiberlookup_Bj_11.out_crd())
        intersectj_10.set_in2(fiberlookup_cj_12.out_ref(), fiberlookup_cj_12.out_crd())
        fiberlookup_Bi_9.set_in_ref(intersectj_10.out_ref1())
        arrayvals_B_4.set_load(fiberlookup_Bi_9.out_ref())
        repsiggen_i_7.set_istream(fiberlookup_Bi_9.out_crd())
        repeat_ci_6.set_in_ref(intersectj_10.out_ref2())
        repeat_ci_6.set_in_repsig(repsiggen_i_7.out_repsig())
        arrayvals_c_5.set_load(repeat_ci_6.out_ref())
        mul_3.set_in1(arrayvals_c_5.out_val())
        mul_3.set_in2(arrayvals_B_4.out_val())
        spaccumulator1_2_drop_crd.set_in_stream(fiberlookup_Bi_9.out_crd())
        spaccumulator1_2_drop_val.set_in_stream(mul_3.out_val())
        spaccumulator1_2.set_crd(spaccumulator1_2_drop_crd.out_val())
        spaccumulator1_2.set_val(spaccumulator1_2_drop_val.out_val())
        fiberwrite_xvals_0.set_input(spaccumulator1_2.out_val())
        fiberwrite_x0_1.set_input(spaccumulator1_2.out_crd_inner())
        fiberlookup_Bj_11.update()

        fiberlookup_cj_12.update()

        intersectj_10.update()
        fiberlookup_Bi_9.update()
        arrayvals_B_4.update()
        repsiggen_i_7.update()
        repeat_ci_6.update()
        arrayvals_c_5.update()
        mul_3.update()
        spaccumulator1_2.update()
        spaccumulator1_2.update()
        spaccumulator1_2.update()
        fiberwrite_xvals_0.update()
        fiberwrite_x0_1.update()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()
    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_c_shape"] = c_shape
    sample_dict = fiberlookup_Bj_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_11" + "_" + k] =  sample_dict[k]

    sample_dict = intersectj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_10" + "_" + k] =  sample_dict[k]

    sample_dict = fiberlookup_Bi_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_9" + "_" + k] =  sample_dict[k]

    sample_dict = spaccumulator1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["spaccumulator1_2" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_ci_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ci_6" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_c_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_5" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "_" + k] =  sample_dict[k]

    sample_dict = fiberlookup_cj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_cj_12" + "_" + k] =  sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_vecmul_ji(ssname, debug_sim, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)