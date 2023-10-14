import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
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
@pytest.mark.suitesparse
def test_mat_identity(samBench, ssname, check_gold, debug_sim, report_stats, fill=0):
    mask_dirname = os.path.join(formatted_dir, ssname, "masked_broadcast")
    mask_shape_filename = os.path.join(mask_dirname, "tensor_mask_mode_shape")
    mask_shape = read_inputs(mask_shape_filename)

    mask0_seg_filename = os.path.join(mask_dirname, "tensor_mask_mode_0_seg")
    mask_seg0 = read_inputs(mask0_seg_filename)
    mask0_crd_filename = os.path.join(mask_dirname, "tensor_mask_mode_0_crd")
    mask_crd0 = read_inputs(mask0_crd_filename)

    mask1_seg_filename = os.path.join(mask_dirname, "tensor_mask_mode_1_seg")
    mask_seg1 = read_inputs(mask1_seg_filename)
    mask1_crd_filename = os.path.join(mask_dirname, "tensor_mask_mode_1_crd")
    mask_crd1 = read_inputs(mask1_crd_filename)

    mask_vals_filename = os.path.join(mask_dirname, "tensor_mask_mode_vals")
    mask_vals = read_inputs(mask_vals_filename, float)

    vec_dirname = os.path.join(formatted_dir, ssname, "masked_broadcast")
    vec_shape_filename = os.path.join(vec_dirname, "tensor_vec_mode_shape")
    vec_shape = read_inputs(vec_shape_filename)

    vec0_seg_filename = os.path.join(vec_dirname, "tensor_vec_mode_0_seg")
    vec_seg0 = read_inputs(vec0_seg_filename)
    vec0_crd_filename = os.path.join(vec_dirname, "tensor_vec_mode_0_crd")
    vec_crd0 = read_inputs(vec0_crd_filename)

    vec_val_filename = os.path.join(vec_dirname, "tensor_vec_mode_vals")
    vec_vals = read_inputs(vec_val_filename)

    fiberlookup_maski_0 = CompressedCrdRdScan(crd_arr=mask_crd0, seg_arr=mask_seg0, debug=debug_sim, statistics=report_stats)
    repsiggen_i_1 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_maskj_2 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_vecj_3 = CompressedCrdRdScan(crd_arr=vec_crd0, seg_arr=vec_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_maskj_4 = CompressedCrdRdScan(crd_arr=mask_crd1, seg_arr=mask_seg1, debug=debug_sim, statistics=report_stats)
    intersectj_5 = Intersect2(debug=debug_sim, statistics=report_stats)
    arrayvals_vec_6 = Array(init_arr=vec_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xi_7 = CompressWrScan(seg_size=2, size=mask_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xj_8 = CompressWrScan(seg_size=mask_shape[0] + 1, size=mask_shape[0] * mask_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_9 = ValsWrScan(size=1 * mask_shape[0] * mask_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)

    in_ref_mask = [0, 'D']
    in_ref_vec = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < 1000:
        if len(in_ref_mask) > 0:
            fiberlookup_maski_0.set_in_ref(in_ref_mask.pop(0))
        if len(in_ref_vec) > 0:
            repeat_maskj_2.set_in_ref(in_ref_vec.pop(0))
        repsiggen_i_1.set_istream(fiberlookup_maski_0.out_crd())
        repeat_maskj_2.set_in_repsig(repsiggen_i_1.out_repsig())
        fiberlookup_vecj_3.set_in_ref(repeat_maskj_2.out_ref())
        fiberlookup_maskj_4.set_in_ref(fiberlookup_maski_0.out_ref())
        intersectj_5.set_in1(fiberlookup_vecj_3.out_ref(), fiberlookup_vecj_3.out_crd())
        intersectj_5.set_in2(fiberlookup_maskj_4.out_ref(), fiberlookup_maskj_4.out_crd())
        arrayvals_vec_6.set_load(intersectj_5.out_ref1())
        fiberwrite_Xi_7.set_input(fiberlookup_maski_0.out_crd())
        fiberwrite_Xj_8.set_input(intersectj_5.out_crd())
        fiberwrite_Xvals_9.set_input(arrayvals_vec_6.out_val())

        fiberlookup_maski_0.update()
        repsiggen_i_1.update()
        repeat_maskj_2.update()
        fiberlookup_vecj_3.update()
        fiberlookup_maskj_4.update()
        intersectj_5.update()
        arrayvals_vec_6.update()
        fiberwrite_Xi_7.update()
        fiberwrite_Xj_8.update()
        fiberwrite_Xvals_9.update()

        done = fiberwrite_Xi_7.out_done() and fiberwrite_Xj_8.out_done() and fiberwrite_Xvals_9.out_done()
        time_cnt += 1

    fiberwrite_Xi_7.autosize()
    fiberwrite_Xj_8.autosize()
    fiberwrite_Xvals_9.autosize()

    out_crds = [fiberwrite_Xi_7.get_arr(), fiberwrite_Xj_8.get_arr()]
    out_segs = [fiberwrite_Xi_7.get_seg_arr(), fiberwrite_Xj_8.get_seg_arr()]
    out_vals = fiberwrite_Xvals_9.get_arr()

    print("out segs", out_segs)
    print("out crds", out_crds)
    print("out_vals", out_vals)

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_mask_shape"] = mask_shape
    extra_info["tensor_vec_shape"] = vec_shape
    sample_dict = fiberlookup_maski_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_maski_0" + "_" + k] = sample_dict[k]

    sample_dict = repsiggen_i_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["repsiggen_i_1" + "_" + k] = sample_dict[k]

    sample_dict = repeat_maskj_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["repeat_maskj_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_vecj_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_vecj_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_maskj_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_maskj_4" + "_" + k] = sample_dict[k]

    sample_dict = intersectj_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj_5" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_vec_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_vec_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xi_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xi_7" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xj_8.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xj_8" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_9.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_9" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_mat_identity(ssname, debug_sim, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
