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
def test_mat_mattransmul(samBench, ssname, check_gold, debug_sim, fill=0):
    b_dirname = os.path.join(formatted_dir, ssname, "dummy", "none")
    b_shape_filename = os.path.join(b_dirname, "b_shape.txt")
    b_shape = read_inputs(b_shape_filename)

    b_vals_filename = os.path.join(b_dirname, "b_vals.txt")
    b_vals = read_inputs(b_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, ssname, "dummy", "ss10")
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

    d_dirname = os.path.join(formatted_dir, ssname, "dummy", "s0")
    d_shape_filename = os.path.join(d_dirname, "d_shape.txt")
    d_shape = read_inputs(d_shape_filename)

    d0_seg_filename = os.path.join(d_dirname, "d0_seg.txt")
    d_seg0 = read_inputs(d0_seg_filename)
    d0_crd_filename = os.path.join(d_dirname, "d0_crd.txt")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "d_vals.txt")
    d_vals = read_inputs(d_vals_filename, float)

    e_dirname = os.path.join(formatted_dir, ssname, "dummy", "none")
    e_shape_filename = os.path.join(e_dirname, "e_shape.txt")
    e_shape = read_inputs(e_shape_filename)

    e_vals_filename = os.path.join(e_dirname, "e_vals.txt")
    e_vals = read_inputs(e_vals_filename, float)

    f_dirname = os.path.join(formatted_dir, ssname, "dummy", "s0")
    f_shape_filename = os.path.join(f_dirname, "f_shape.txt")
    f_shape = read_inputs(f_shape_filename)

    f0_seg_filename = os.path.join(f_dirname, "f0_seg.txt")
    f_seg0 = read_inputs(f0_seg_filename)
    f0_crd_filename = os.path.join(f_dirname, "f0_crd.txt")
    f_crd0 = read_inputs(f0_crd_filename)

    f_vals_filename = os.path.join(f_dirname, "f_vals.txt")
    f_vals = read_inputs(f_vals_filename, float)

    fiberlookup_Ci_27 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    fiberlookup_fi_28 = CompressedCrdRdScan(crd_arr=f_crd0, seg_arr=f_seg0, debug=debug_sim)
    unioni_26 = Union2(debug=debug_sim)
    fiberlookup_Cj_18 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=C_shape[1], fill=fill, debug=debug_sim)
    repsiggen_i_24 = RepeatSigGen(debug=debug_sim)
    repeat_bi_20 = Repeat(debug=debug_sim)
    repeat_di_21 = Repeat(debug=debug_sim)
    repeat_ei_22 = Repeat(debug=debug_sim)
    fiberlookup_dj_19 = CompressedCrdRdScan(crd_arr=d_crd0, seg_arr=d_seg0, debug=debug_sim)
    intersectj_17 = Intersect2(debug=debug_sim)
    repsiggen_j_16 = RepeatSigGen(debug=debug_sim)
    arrayvals_C_7 = Array(init_arr=C_vals, debug=debug_sim)
    arrayvals_d_8 = Array(init_arr=d_vals, debug=debug_sim)
    repeat_bj_12 = Repeat(debug=debug_sim)
    repeat_ej_13 = Repeat(debug=debug_sim)
    repeat_fj_14 = Repeat(debug=debug_sim)
    arrayvals_b_6 = Array(init_arr=b_vals, debug=debug_sim)
    arrayvals_e_10 = Array(init_arr=e_vals, debug=debug_sim)
    arrayvals_f_11 = Array(init_arr=f_vals, debug=debug_sim)
    mul_5 = Multiply2(debug=debug_sim)
    mul_9 = Multiply2(debug=debug_sim)
    mul_4 = Multiply2(debug=debug_sim)
    add_3 = Add2(debug=debug_sim)
    reduce_2 = Reduce(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * C_shape[1], fill=fill, debug=debug_sim)
    in_ref_C = [0, 'D']
    in_ref_f = [0, 'D']
    in_ref_b = [0, 'D']
    in_ref_d = [0, 'D']
    in_ref_e = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_C) > 0:
            fiberlookup_Ci_27.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci_27.update()

        if len(in_ref_f) > 0:
            fiberlookup_fi_28.set_in_ref(in_ref_f.pop(0))
        fiberlookup_fi_28.update()

        unioni_26.set_in1(fiberlookup_Ci_27.out_ref(), fiberlookup_Ci_27.out_crd())
        unioni_26.set_in2(fiberlookup_fi_28.out_ref(), fiberlookup_fi_28.out_crd())
        unioni_26.update()

        fiberlookup_Cj_18.set_in_ref(unioni_26.out_ref1())
        fiberlookup_Cj_18.update()

        fiberwrite_x0_1.set_input(unioni_26.out_crd())
        fiberwrite_x0_1.update()

        repsiggen_i_24.set_istream(unioni_26.out_crd())
        repsiggen_i_24.update()

        if len(in_ref_b) > 0:
            repeat_bi_20.set_in_ref(in_ref_b.pop(0))
        repeat_bi_20.set_in_repsig(repsiggen_i_24.out_repsig())
        repeat_bi_20.update()

        if len(in_ref_d) > 0:
            repeat_di_21.set_in_ref(in_ref_d.pop(0))
        repeat_di_21.set_in_repsig(repsiggen_i_24.out_repsig())
        repeat_di_21.update()

        if len(in_ref_e) > 0:
            repeat_ei_22.set_in_ref(in_ref_e.pop(0))
        repeat_ei_22.set_in_repsig(repsiggen_i_24.out_repsig())
        repeat_ei_22.update()

        fiberlookup_dj_19.set_in_ref(repeat_di_21.out_ref())
        fiberlookup_dj_19.update()

        intersectj_17.set_in1(fiberlookup_dj_19.out_ref(), fiberlookup_dj_19.out_crd())
        intersectj_17.set_in2(fiberlookup_Cj_18.out_ref(), fiberlookup_Cj_18.out_crd())
        intersectj_17.update()

        repsiggen_j_16.set_istream(intersectj_17.out_crd())
        repsiggen_j_16.update()

        arrayvals_C_7.set_load(intersectj_17.out_ref2())
        arrayvals_C_7.update()

        arrayvals_d_8.set_load(intersectj_17.out_ref1())
        arrayvals_d_8.update()

        repeat_bj_12.set_in_ref(repeat_bi_20.out_ref())
        repeat_bj_12.set_in_repsig(repsiggen_j_16.out_repsig())
        repeat_bj_12.update()

        repeat_ej_13.set_in_repsig(repsiggen_j_16.out_repsig())
        repeat_ej_13.set_in_ref(repeat_ei_22.out_ref())
        repeat_ej_13.update()

        repeat_fj_14.set_in_ref(unioni_26.out_ref2())
        repeat_fj_14.set_in_repsig(repsiggen_j_16.out_repsig())
        repeat_fj_14.update()

        arrayvals_e_10.set_load(repeat_ej_13.out_ref())
        arrayvals_e_10.update()

        arrayvals_f_11.set_load(repeat_fj_14.out_ref())
        arrayvals_f_11.update()

        mul_9.set_in1(arrayvals_e_10.out_val())
        mul_9.set_in2(arrayvals_f_11.out_val())
        mul_9.update()

        arrayvals_b_6.set_load(repeat_bj_12.out_ref())
        arrayvals_b_6.update()

        mul_5.set_in1(arrayvals_b_6.out_val())
        mul_5.set_in2(arrayvals_C_7.out_val())
        mul_5.update()

        mul_4.set_in1(mul_5.out_val())
        mul_4.set_in2(arrayvals_d_8.out_val())
        mul_4.update()

        add_3.set_in1(mul_4.out_val())
        add_3.set_in2(mul_9.out_val())
        add_3.update()

        reduce_2.set_in_val(add_3.out_val())
        reduce_2.update()

        fiberwrite_xvals_0.set_input(reduce_2.out_val())
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
    extra_info["tensor_e_shape"] = e_shape
    extra_info["tensor_f_shape"] = f_shape
    sample_dict = fiberwrite_x0_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_x0_1" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_bi_20.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_bi_20" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_bj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_bj_12" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_b_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_b_6" + "_" + k] =  sample_dict[k]

    sample_dict = reduce_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["reduce_2" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_di_21.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_di_21" + "_" + k] =  sample_dict[k]

    sample_dict = intersectj_17.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_17" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_ej_13.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ej_13" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_e_10.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_e_10" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_fj_14.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_fj_14" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_f_11.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_f_11" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_C_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_7" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_d_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_d_8" + "_" + k] =  sample_dict[k]

    sample_dict = repeat_ei_22.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_ei_22" + "_" + k] =  sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_mattransmul(ssname, debug_sim, out_crds, out_segs, out_vals, "s0")
    samBench(bench, extra_info)