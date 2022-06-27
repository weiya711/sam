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
def test_mat_residual(samBench, ssname, check_gold, debug_sim, fill=0):
    C_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
    C_shape_filename = os.path.join(C_dirname, "B_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "B0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "B0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "B1_seg.txt")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "B1_crd.txt")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "B_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    b_dirname = os.path.join(formatted_dir, ssname, "other")
    b_fname = [f for f in os.listdir(b_dirname) if ssname + "-vec_mode0" in f]
    assert len(b_fname) == 1, "Should only have one 'other' folder that matches"
    b_fname = b_fname[0]
    b_dirname = os.path.join(b_dirname, b_fname)
    b_shape = [C_shape[0]]

    b0_seg_filename = os.path.join(b_dirname, "C0_seg.txt")
    b_seg0 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "C0_crd.txt")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "C_vals.txt")
    b_vals = read_inputs(b_vals_filename, float)

    d_dirname = os.path.join(formatted_dir, ssname, "other")
    d_fname = [f for f in os.listdir(d_dirname) if ssname + "-vec_mode1" in f]
    assert len(d_fname) == 1, "Should only have one 'other' folder that matches"
    d_fname = d_fname[0]
    d_dirname = os.path.join(d_dirname, d_fname)

    d_shape = [C_shape[1]]

    d0_seg_filename = os.path.join(d_dirname, "C0_seg.txt")
    d_seg0 = read_inputs(d0_seg_filename)
    d0_crd_filename = os.path.join(d_dirname, "C0_crd.txt")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "C_vals.txt")
    d_vals = read_inputs(d_vals_filename, float)

    fiberlookup_bi_17 = CompressedCrdRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim)
    fiberlookup_Ci_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    unioni_16 = Union2(debug=debug_sim)
    fiberlookup_Cj_11 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=b_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_14 = RepeatSigGen(debug=debug_sim)
    repeat_di_13 = Repeat(debug=debug_sim)
    fiberlookup_dj_12 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim)
    intersectj_10 = Intersect2(debug=debug_sim)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim)
    arrayvals_d_7 = Array(init_arr=d_vals, debug=debug_sim)
    mul_5 = Multiply2(debug=debug_sim)
    arrayvals_b_4 = Array(init_arr=b_vals, debug=debug_sim)
    add_3 = Add2(debug=debug_sim, neg2=True)
    reduce_2 = Reduce(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * b_shape[0], fill=fill, debug=debug_sim)
    in_ref_b = [0, 'D']
    in_ref_C = [0, 'D']
    in_ref_d = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_b) > 0:
            fiberlookup_bi_17.set_in_ref(in_ref_b.pop(0))
        fiberlookup_bi_17.update()

        if len(in_ref_C) > 0:
            fiberlookup_Ci_18.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci_18.update()

        unioni_16.set_in1(fiberlookup_bi_17.out_ref(), fiberlookup_bi_17.out_crd())
        unioni_16.set_in2(fiberlookup_Ci_18.out_ref(), fiberlookup_Ci_18.out_crd())
        unioni_16.update()

        fiberlookup_Cj_11.set_in_ref(unioni_16.out_ref2())
        fiberlookup_Cj_11.update()

        fiberwrite_x0_1.set_input(unioni_16.out_crd())
        fiberwrite_x0_1.update()

        repsiggen_i_14.set_istream(unioni_16.out_crd())
        repsiggen_i_14.update()

        if len(in_ref_d) > 0:
            repeat_di_13.set_in_ref(in_ref_d.pop(0))
        repeat_di_13.set_in_repsig(repsiggen_i_14.out_repsig())
        repeat_di_13.update()

        fiberlookup_dj_12.set_in_ref(repeat_di_13.out_ref())
        fiberlookup_dj_12.update()

        intersectj_10.set_in1(fiberlookup_dj_12.out_ref(), fiberlookup_dj_12.out_crd())
        intersectj_10.set_in2(fiberlookup_Cj_11.out_ref(), fiberlookup_Cj_11.out_crd())
        intersectj_10.update()

        arrayvals_C_6.set_load(intersectj_10.out_ref2())
        arrayvals_C_6.update()

        arrayvals_d_7.set_load(intersectj_10.out_ref1())
        arrayvals_d_7.update()

        arrayvals_b_4.set_load(unioni_16.out_ref1())
        arrayvals_b_4.update()

        mul_5.set_in1(arrayvals_C_6.out_val())
        mul_5.set_in2(arrayvals_d_7.out_val())
        mul_5.update()

        reduce_2.set_in_val(mul_5.out_val())
        reduce_2.update()

        add_3.set_in1(arrayvals_b_4.out_val())
        add_3.set_in2(reduce_2.out_val())
        add_3.update()

        fiberwrite_xvals_0.set_input(add_3.out_val())
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
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_d_shape"] = d_shape


    extra_info["tensor_b/nnz"] = len(b_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)
    extra_info["tensor_d/nnz"] = len(d_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = unioni_16.return_statistics()
    for k in sample_dict.keys():
        extra_info["unioni_16" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "/" + k] = sample_dict[k]

    sample_dict = repeat_di_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_di_13" + "/" + k] = sample_dict[k]

    sample_dict = intersectj_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_10" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_b_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_b_4" + "/" + k] = sample_dict[k]

    sample_dict = reduce_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_d_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_d_7" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_residual(ssname, debug_sim, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)
