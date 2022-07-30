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
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

other_dir = os.getenv('OTHER_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_mat_vecmul_ij(samBench, ssname, check_gold, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
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

    c_dirname = os.path.join(formatted_dir, ssname, "other")
    c_fname = [f for f in os.listdir(c_dirname) if ssname + "-vec_mode1" in f]
    assert len(c_fname) == 1, "Should only have one 'other' folder that matches"
    c_fname = c_fname[0]
    c_dirname = os.path.join(c_dirname, c_fname)

    c_shape_filename = os.path.join(c_dirname, "C_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "C0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "C0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "C_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_Bi_12 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Bj_7 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_10 = RepeatSigGen(debug=debug_sim)
    repeat_ci_9 = Repeat(debug=debug_sim)
    fiberlookup_cj_8 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    intersectj_6 = Intersect2(debug=debug_sim)
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_c_5 = Array(init_arr=c_vals, debug=debug_sim)
    mul_3 = Multiply2(debug=debug_sim)
    reduce_2 = Reduce(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * B_shape[0], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_12.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_12.update()

        fiberlookup_Bj_7.set_in_ref(fiberlookup_Bi_12.out_ref())
        fiberlookup_Bj_7.update()

        fiberwrite_x0_1.set_input(fiberlookup_Bi_12.out_crd())
        fiberwrite_x0_1.update()

        repsiggen_i_10.set_istream(fiberlookup_Bi_12.out_crd())
        repsiggen_i_10.update()

        if len(in_ref_c) > 0:
            repeat_ci_9.set_in_ref(in_ref_c.pop(0))
        repeat_ci_9.set_in_repsig(repsiggen_i_10.out_repsig())
        repeat_ci_9.update()

        fiberlookup_cj_8.set_in_ref(repeat_ci_9.out_ref())
        fiberlookup_cj_8.update()

        intersectj_6.set_in1(fiberlookup_cj_8.out_ref(), fiberlookup_cj_8.out_crd())
        intersectj_6.set_in2(fiberlookup_Bj_7.out_ref(), fiberlookup_Bj_7.out_crd())
        intersectj_6.update()

        arrayvals_B_4.set_load(intersectj_6.out_ref2())
        arrayvals_B_4.update()

        arrayvals_c_5.set_load(intersectj_6.out_ref1())
        arrayvals_c_5.update()

        mul_3.set_in1(arrayvals_B_4.out_val())
        mul_3.set_in2(arrayvals_c_5.out_val())
        mul_3.update()

        reduce_2.set_in_val(mul_3.out_val())
        reduce_2.update()

        fiberwrite_xvals_0.set_input(reduce_2.out_val())
        fiberwrite_xvals_0.update()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time_cnt += 1

        if time_cnt % 100 == 0:
            print(fiberwrite_xvals_0.out_done(), reduce_2.out_done(),
                  mul_3.out_done(), arrayvals_c_5.out_done(), arrayvals_B_4.out_done(),
                  intersectj_6.out_done(), fiberlookup_cj_8.out_done(),
                  repeat_ci_9.out_done(), fiberlookup_Bj_7.out_done())
            print("TIME:", time_cnt)

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
    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_ci_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ci_9" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_6" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "_" + k] = sample_dict[k]

    sample_dict = reduce_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_c_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_5" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_vecmul_ij(ssname, debug_sim, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)
