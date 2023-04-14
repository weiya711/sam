import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
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
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor4_softmax(samBench, frosttname, cast, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "tensor4_softmax")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename)

    B2_seg_filename = os.path.join(B_dirname, "tensor_B_mode_2_seg")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "tensor_B_mode_2_crd")
    B_crd2 = read_inputs(B2_crd_filename)

    B3_seg_filename = os.path.join(B_dirname, "tensor_B_mode_3_seg")
    B_seg3 = read_inputs(B3_seg_filename)
    B3_crd_filename = os.path.join(B_dirname, "tensor_B_mode_3_crd")
    B_crd3 = read_inputs(B3_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)


    fiberlookup_Bi_7 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberwrite_X0_3 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_6 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
    fiberwrite_X1_2 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_5 = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    fiberlookup_Bl_6 = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats)
    fiberwrite_X2_1 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    fiberwrite_X3_0 = CompressWrScan(seg_size=B_shape[0] * B_shape[1] * B_shape[2] * B_shape[3], size=B_shape[0] * B_shape[1] * B_shape[2] * B_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_B_10 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2] * B_shape[3], fill=fill, debug=debug_sim, statistics=report_stats)
    repsiggen_l_13 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bl_12 = Repeat(debug=debug_sim, statistics=report_stats)
    exp_1 = Exp(in2=0, debug=debug_sim, statistics=report_stats)
    reduce_5 = Reduce(debug=debug_sim, statistics=report_stats)
    drop_9 = CrdDrop(debug=debug_sim)
    div_6 = Divide2(debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    done = False
    time_cnt = 0

    print("B seg1", B_seg1)
    print("B seg2", B_seg2)
    print("B seg3", B_seg3)

    # pytest.set_trace()

    out_debug = []

    div_in = []
    div1_in = []
    div_out = []
    div1_out = []

    repeater = []
    reducer = []
    fiber_crd = []
    repsig = []

    out_l_crd = []

    count = 0
    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_7.set_in_ref(in_ref_B.pop(0))
        fiberwrite_X0_3.set_input(fiberlookup_Bi_7.out_crd())
        fiberlookup_Bj_6.set_in_ref(fiberlookup_Bi_7.out_ref())
        fiberwrite_X1_2.set_input(fiberlookup_Bj_6.out_crd())
        fiberlookup_Bk_5.set_in_ref(fiberlookup_Bj_6.out_ref())
        fiberlookup_Bl_6.set_in_ref(fiberlookup_Bk_5.out_ref())
        fiberwrite_X2_1.set_input(fiberlookup_Bk_5.out_crd())
        fiberwrite_X3_0.set_input(fiberlookup_Bl_6.out_crd())
        arrayvals_B_4.set_load(fiberlookup_Bl_6.out_ref())

        exp_1.set_in1(arrayvals_B_4.out_load())
        reduce_5.set_in_val(exp_1.out_val())
        repsiggen_l_13.set_istream(fiberlookup_Bl_6.out_ref())
        repeat_Bl_12.set_in_ref(reduce_5.out_val())
        repeat_Bl_12.set_in_repsig(repsiggen_l_13.out_repsig())
        div_6.set_in1(exp_1.out_val())
        div_6.set_in2(repeat_Bl_12.out_ref())

        # fiber_crd.append(fiberlookup_Bl_6.out_crd())
        reducer.append(reduce_5.out_val())
        repeater.append(repeat_Bl_12.out_ref())
        repsig.append(repsiggen_l_13.out_repsig())

        # print("crd:", remove_emptystr(fiber_crd))
        # print('=' * 100)
        # print("Reduce:", remove_emptystr(reducer))
        # print("Repeater:", remove_emptystr(repeater))
        # print()
        # print("Repsig:", remove_emptystr(repsig))

        div_in.append(exp_1.out_val())
        div1_in.append(repeat_Bl_12.out_ref())
        out_debug.append(div_6.out_val())
        # print("div0 in", remove_emptystr(div_in))
        # print()
        # print("div1 in", remove_emptystr(div1_in))
        # print()
        # print("div out", remove_emptystr(out_debug))
        fiberwrite_Xvals_0.set_input(div_6.out_val())

        fiberlookup_Bi_7.update()
        fiberlookup_Bj_6.update()
        fiberlookup_Bk_5.update()
        fiberlookup_Bl_6.update()
        arrayvals_B_4.update()
        exp_1.update()
        reduce_5.update()
        # arrayvals_B_10.update()
        repsiggen_l_13.update()
        repeat_Bl_12.update()
        div_6.update()
        fiberwrite_X0_3.update()
        fiberwrite_X1_2.update()
        fiberwrite_X2_1.update()
        fiberwrite_X3_0.update()
        fiberwrite_Xvals_0.update()

        done_ = fiberwrite_X0_3.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X2_1.out_done() and fiberwrite_Xvals_0.out_done()
        if done_:
            count += 1
        done = False
        if count == 4:
            done = True
        # done = exp_1.out_done()
        time_cnt += 1

    fiberwrite_X0_3.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X2_1.autosize()
    fiberwrite_X3_0.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_3.get_arr(), fiberwrite_X1_2.get_arr(), fiberwrite_X2_1.get_arr(), fiberwrite_X3_0.get_arr()]
    out_segs = [fiberwrite_X0_3.get_seg_arr(), fiberwrite_X1_2.get_seg_arr(), fiberwrite_X2_1.get_seg_arr(), fiberwrite_X3_0.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    # extra_info["dataset"] = 
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    sample_dict = fiberlookup_Bi_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_7" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor4_softmax(frosttname, debug_sim, cast, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
