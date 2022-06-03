import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
import os
import csv
cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_elemmul(samBench, ssname, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "sss012")
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

    C_dirname = os.path.join(formatted_dir, ssname, "shift", "sss012")
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

    C2_seg_filename = os.path.join(C_dirname, "C2_seg.txt")
    C_seg2 = read_inputs(C2_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "C2_crd.txt")
    C_crd2 = read_inputs(C2_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_16 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Ci_17 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    intersecti_15 = Intersect2(debug=debug_sim)
    fiberlookup_Bj_13 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberlookup_Cj_14 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    intersectj_12 = Intersect2(debug=debug_sim)
    fiberlookup_Bk_10 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim)
    fiberlookup_Ck_11 = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim)
    intersectk_9 = Intersect2(debug=debug_sim)
    crddrop_8 = CrdDrop(debug=debug_sim)
    arrayvals_B_5 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_C_6 = Array(init_arr=C_vals, debug=debug_sim)
    crddrop_7 = CrdDrop(debug=debug_sim)
    fiberwrite_X2_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim)
    mul_4 = Multiply2(debug=debug_sim)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_16.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_16.update()

        if len(in_ref_C) > 0:
            fiberlookup_Ci_17.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci_17.update()

        intersecti_15.set_in1(fiberlookup_Bi_16.out_ref(), fiberlookup_Bi_16.out_crd())
        intersecti_15.set_in2(fiberlookup_Ci_17.out_ref(), fiberlookup_Ci_17.out_crd())
        intersecti_15.update()

        fiberlookup_Bj_13.set_in_ref(intersecti_15.out_ref1())
        fiberlookup_Bj_13.update()

        fiberlookup_Cj_14.set_in_ref(intersecti_15.out_ref2())
        fiberlookup_Cj_14.update()

        intersectj_12.set_in1(fiberlookup_Bj_13.out_ref(), fiberlookup_Bj_13.out_crd())
        intersectj_12.set_in2(fiberlookup_Cj_14.out_ref(), fiberlookup_Cj_14.out_crd())
        intersectj_12.update()

        fiberlookup_Bk_10.set_in_ref(intersectj_12.out_ref1())
        fiberlookup_Bk_10.update()

        fiberlookup_Ck_11.set_in_ref(intersectj_12.out_ref2())
        fiberlookup_Ck_11.update()

        intersectk_9.set_in1(fiberlookup_Bk_10.out_ref(), fiberlookup_Bk_10.out_crd())
        intersectk_9.set_in2(fiberlookup_Ck_11.out_ref(), fiberlookup_Ck_11.out_crd())
        intersectk_9.update()

        crddrop_8.set_outer_crd(intersectj_12.out_crd())
        crddrop_8.set_inner_crd(intersectk_9.out_crd())
        arrayvals_B_5.set_load(intersectk_9.out_ref1())
        arrayvals_B_5.update()

        arrayvals_C_6.set_load(intersectk_9.out_ref2())
        arrayvals_C_6.update()

        mul_4.set_in1(arrayvals_B_5.out_load())
        mul_4.set_in2(arrayvals_C_6.out_load())
        mul_4.update()

        crddrop_7.set_outer_crd(intersecti_15.out_crd())
        crddrop_7.set_inner_crd(crddrop_8.out_crd())
        fiberwrite_X2_1.set_input(crddrop_8.out_crd_inner())
        fiberwrite_X2_1.update()

        fiberwrite_Xvals_0.set_input(mul_4.out_val())
        fiberwrite_Xvals_0.update()

        fiberwrite_X0_3.set_input(crddrop_7.out_crd_outer())
        fiberwrite_X0_3.update()

        fiberwrite_X1_2.set_input(crddrop_7.out_crd_inner())
        fiberwrite_X1_2.update()

        done = fiberwrite_X2_1.out_done() and fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_X2_1.autosize()
    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X2_1.get_arr(), fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr()]
    out_segs = [fiberwrite_X2_1.get_seg_arr(), fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()
    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    sample_dict = intersecti_15.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_15" + "_" + k] =  sample_dict[k]

    sample_dict = crddrop_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_7" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] =  sample_dict[k]

    sample_dict = intersectj_12.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_12" + "_" + k] =  sample_dict[k]

    sample_dict = crddrop_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["crddrop_8" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] =  sample_dict[k]

    sample_dict = intersectk_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectk_9" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_B_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_5" + "_" + k] =  sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] =  sample_dict[k]

    sample_dict = arrayvals_C_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C_6" + "_" + k] =  sample_dict[k]

    samBench(bench, extra_info)