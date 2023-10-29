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
formatted_dir = os.getenv('CUSTOM_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
def test_trans_masked_broadcast(samBench, ssname, cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "masked_broadcast")
    B_shape_filename = os.path.join(B_dirname, "tensor_mask_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname,  "tensor_mask_mode_0_seg" )
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_mask_mode_0_crd" )
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_mask_mode_1_seg" )
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_mask_mode_1_crd" )
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_mask_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    c_dirname = os.path.join(formatted_dir, ssname, "masked_broadcast")
    c_shape_filename = os.path.join(c_dirname, "tensor_vec_mode_shape")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "tensor_vec_mode_0_seg")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "tensor_vec_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "tensor_vec_mode_vals")
    c_vals = read_inputs(c_vals_filename, float)


    fiberlookup_Bi_0 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_ci_1 = CompressedCrdRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim, statistics=report_stats)
    intersecti_2 = Intersect2(debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_3 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_7 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_j_4 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_8 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    repeat_cj_5 = Repeat(debug=debug_sim, statistics=report_stats)
    arrayvals_c_6 = Array(init_arr=c_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_9 = ValsWrScan(size=1 * B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_0.set_in_ref(in_ref_B.pop(0))
        if len(in_ref_c) > 0:
            fiberlookup_ci_1.set_in_ref(in_ref_c.pop(0))
        intersecti_2.set_in1(fiberlookup_Bi_0.out_ref(), fiberlookup_Bi_0.out_crd())
        intersecti_2.set_in2(fiberlookup_ci_1.out_ref(), fiberlookup_ci_1.out_crd())
        fiberlookup_Bj_3.set_in_ref(intersecti_2.out_ref1())
        fiberwrite_X0_7.set_input(intersecti_2.out_crd())
        repsiggen_j_4.set_istream(fiberlookup_Bj_3.out_crd())
        fiberwrite_X1_8.set_input(fiberlookup_Bj_3.out_crd())
        repeat_cj_5.set_in_ref(intersecti_2.out_ref2())
        repeat_cj_5.set_in_repsig(repsiggen_j_4.out_repsig())
        arrayvals_c_6.set_load(repeat_cj_5.out_ref())
        fiberwrite_Xvals_9.set_input(arrayvals_c_6.out_val())
        fiberlookup_Bi_0.update()
        fiberlookup_ci_1.update()
        intersecti_2.update()
        fiberlookup_Bj_3.update()
        fiberwrite_X0_7.update()
        repsiggen_j_4.update()
        fiberwrite_X1_8.update()
        repeat_cj_5.update()
        arrayvals_c_6.update()
        fiberwrite_Xvals_9.update()

        done = fiberwrite_X0_7.out_done() and fiberwrite_X1_8.out_done() and fiberwrite_Xvals_9.out_done()
        time_cnt += 1

    fiberwrite_X0_7.autosize()
    fiberwrite_X1_8.autosize()
    fiberwrite_Xvals_9.autosize()

    out_crds = [fiberwrite_X0_7.get_arr(), fiberwrite_X1_8.get_arr()]
    out_segs = [fiberwrite_X0_7.get_seg_arr(), fiberwrite_X1_8.get_seg_arr()]
    out_vals = fiberwrite_Xvals_9.get_arr()

    print(out_crds)
    print(out_segs)
    print(out_vals)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_c_shape"] = c_shape
    sample_dict = fiberlookup_Bi_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_0" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_ci_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_ci_1" + "_" + k] = sample_dict[k]

    sample_dict = intersecti_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_3" + "_" + k] = sample_dict[k]

    sample_dict = repeat_cj_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_cj_5" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_c_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_c_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_7" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_8" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_9" + "_" + k] = sample_dict[k]

    # if check_gold:
    #     print("Checking gold...")
    #     check_gold_trans_masked_broadcast(, debug_sim, cast, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
