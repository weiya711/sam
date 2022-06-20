import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2
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
@pytest.mark.vec
def test_vec_elemmul(samBench, ssname, check_gold, debug_sim, fill=0):
    b_dirname = os.path.join(formatted_dir, ssname, "orig", "s0")
    b_shape_filename = os.path.join(b_dirname, "b_shape.txt")
    b_shape = read_inputs(b_shape_filename)

    b0_seg_filename = os.path.join(b_dirname, "b0_seg.txt")
    b_seg0 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "b0_crd.txt")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "b_vals.txt")
    b_vals = read_inputs(b_vals_filename, float)

    c_dirname = os.path.join(formatted_dir, ssname, "shift", "s0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "c0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "c0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_bi_6 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim)
    fiberlookup_ci_7 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    intersecti_5 = Intersect2(debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=b_shape[0], fill=fill, debug=debug_sim)
    arrayvals_b_3 = Array(init_arr=b_vals, debug=debug_sim)
    arrayvals_c_4 = Array(init_arr=c_vals, debug=debug_sim)
    mul_2 = Multiply2(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * b_shape[0], fill=fill, debug=debug_sim)
    in_ref_b = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_b) > 0:
            fiberlookup_bi_6.set_in_ref(in_ref_b.pop(0))
        fiberlookup_bi_6.update()

        if len(in_ref_c) > 0:
            fiberlookup_ci_7.set_in_ref(in_ref_c.pop(0))
        fiberlookup_ci_7.update()

        intersecti_5.set_in1(fiberlookup_bi_6.out_ref(), fiberlookup_bi_6.out_crd())
        intersecti_5.set_in2(fiberlookup_ci_7.out_ref(), fiberlookup_ci_7.out_crd())
        intersecti_5.update()

        fiberwrite_x0_1.set_input(intersecti_5.out_crd())
        fiberwrite_x0_1.update()

        arrayvals_b_3.set_load(intersecti_5.out_ref1())
        arrayvals_b_3.update()

        arrayvals_c_4.set_load(intersecti_5.out_ref2())
        arrayvals_c_4.update()

        mul_2.set_in1(arrayvals_b_3.out_load())
        mul_2.set_in2(arrayvals_c_4.out_load())
        mul_2.update()

        fiberwrite_xvals_0.set_input(mul_2.out_val())
        fiberwrite_xvals_0.update()

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
    extra_info["tensor_b_shape"] = b_shape
    extra_info["tensor_c_shape"] = c_shape
    sample_dict = intersecti_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_5" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_b_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_b_3" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_c_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_4" + "_" + k] =  sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_vec_elemmul(ssname, debug_sim, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)