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
@pytest.mark.suitesparse
def test_mat_elemadd(samBench, ssname, check_gold, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname,  "orig", "ss01")
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

    C_dirname = os.path.join(formatted_dir, ssname,  "shift", "ss01")
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

    fiberlookup_Bi_10 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Ci_11 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    unioni_9 = Union2(debug=debug_sim)
    fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    fiberlookup_Bj_7 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberlookup_Cj_8 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    unionj_6 = Union2(debug=debug_sim)
    fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_C_5 = Array(init_arr=C_vals, debug=debug_sim)
    add_3 = Add2(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_10.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_10.update()

        if len(in_ref_C) > 0:
            fiberlookup_Ci_11.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci_11.update()

        unioni_9.set_in1(fiberlookup_Bi_10.out_ref(), fiberlookup_Bi_10.out_crd())
        unioni_9.set_in2(fiberlookup_Ci_11.out_ref(), fiberlookup_Ci_11.out_crd())
        unioni_9.update()

        fiberwrite_X0_2.set_input(unioni_9.out_crd())
        fiberwrite_X0_2.update()

        fiberlookup_Bj_7.set_in_ref(unioni_9.out_ref1())
        fiberlookup_Bj_7.update()

        fiberlookup_Cj_8.set_in_ref(unioni_9.out_ref2())
        fiberlookup_Cj_8.update()

        unionj_6.set_in1(fiberlookup_Bj_7.out_ref(), fiberlookup_Bj_7.out_crd())
        unionj_6.set_in2(fiberlookup_Cj_8.out_ref(), fiberlookup_Cj_8.out_crd())
        unionj_6.update()

        fiberwrite_X1_1.set_input(unionj_6.out_crd())
        fiberwrite_X1_1.update()

        arrayvals_B_4.set_load(unionj_6.out_ref1())
        arrayvals_B_4.update()

        arrayvals_C_5.set_load(unionj_6.out_ref2())
        arrayvals_C_5.update()

        add_3.set_in1(arrayvals_B_4.out_val())
        add_3.set_in2(arrayvals_C_5.out_val())
        add_3.update()

        fiberwrite_Xvals_0.set_input(add_3.out_val())
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
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = fiberwrite_X0_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_2" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X1_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_1" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_C_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_5" + "_" + k] =  sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_elemadd(ssname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)