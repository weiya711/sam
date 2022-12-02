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
import torch
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
def test_tensor4_mul(samBench, frosttname, check_gold, report_stats, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "orig", "ssss0123")
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

    B3_seg_filename = os.path.join(B_dirname, "B3_seg.txt")
    B_seg3 = read_inputs(B3_seg_filename)
    B3_crd_filename = os.path.join(B_dirname, "B3_crd.txt")
    B_crd3 = read_inputs(B3_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, frosttname, "other", "ssss0123")
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

    C3_seg_filename = os.path.join(C_dirname, "C3_seg.txt")
    C_seg3 = read_inputs(C3_seg_filename)
    C3_crd_filename = os.path.join(C_dirname, "C3_crd.txt")
    C_crd3 = read_inputs(C3_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, frosttname + "_dense", "orig", "ssss0123")
    C_vals_filename = os.path.join(C_dirname, "B_vals.txt")
    B_ref_dense = read_inputs(C_vals_filename)
    B_ref = torch.Tensor(B_ref_dense)
    B_ref = B_ref.view(B_shape)

    C_dirname = os.path.join(formatted_dir, "K_dense", "orig", "ssss0123")
    C_vals_filename = os.path.join(C_dirname, "B_vals.txt")
    C_ref_dense = read_inputs(C_vals_filename)
    C_ref = torch.Tensor(C_ref_dense)
    C_ref = C_ref.view(C_shape)

    gold_ref = torch.einsum('ikjm, iljm->ijkl', B_ref, C_ref)
    print(gold_ref)

    # print(B_vals)
    # print()
    # print(C_vals)
    # print()
    # print(B_shape)
    # print(C_shape)

    fiberlookup_Bi = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
    fiberlookup_Ci = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
    intersecti = Intersect2(debug=debug_sim, statistics=report_stats)

    fiberwrite_X0 = CompressWrScan(seg_size=2, size=2 * len(B_crd0), fill=fill, debug=debug_sim, statistics=report_stats)

    fiberlookup_Bk = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)

    repsiggen_k = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ck = Repeat(debug=debug_sim, statistics=report_stats)

    fiberlookup_Cl = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)

    repsiggen_l = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bl = Repeat(debug=debug_sim, statistics=report_stats)

    fiberlookup_Bj = CompressedCrdRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cj = CompressedCrdRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim, statistics=report_stats)
    intersectj = Intersect2(debug=debug_sim, statistics=report_stats)

    fiberwrite_X1 = CompressWrScan(seg_size=2 * len(B_crd0) + 1, size=2 * len(B_crd1), fill=fill,
                                     debug=debug_sim, statistics=report_stats)

    fiberlookup_Bm = CompressedCrdRdScan(crd_arr=B_crd3, seg_arr=B_seg3, debug=debug_sim, statistics=report_stats)
    fiberlookup_Cm = CompressedCrdRdScan(crd_arr=C_crd3, seg_arr=C_seg3, debug=debug_sim, statistics=report_stats)
    intersectm = Intersect2(debug=debug_sim, statistics=report_stats)

    fiberwrite_X2 = CompressWrScan(seg_size=2 * len(B_crd1) + 1, size=2 * len(B_crd2), fill=fill,
                                     debug=debug_sim, statistics=report_stats)

    # TODO: Figure out proper size
    fiberwrite_X3 = CompressWrScan(seg_size=2 * len(B_crd2) + 1, size=len(B_vals) * 2, fill=fill,
                                     debug=debug_sim, statistics=report_stats)
    arrayvals_B = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    arrayvals_C = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
    mul = Multiply2(debug=debug_sim, statistics=report_stats)
    print("Multiply done")
    reduce = Reduce(debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt = 0

    if debug_sim:
        print("blocks done")

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))

        if len(in_ref_C) > 0:
            fiberlookup_Ci.set_in_ref(in_ref_C.pop(0))

        intersecti.set_in1(fiberlookup_Bi.out_ref(), fiberlookup_Bi.out_crd())
        intersecti.set_in2(fiberlookup_Ci.out_ref(), fiberlookup_Ci.out_crd())

        fiberwrite_X0.set_input(intersecti.out_crd())

        fiberlookup_Bk.set_in_ref(intersecti.out_ref1())

        fiberwrite_X2.set_input(fiberlookup_Bk.out_crd())

        repsiggen_k.set_istream(fiberlookup_Bk.out_crd())

        repeat_Ck.set_in_ref(intersecti.out_ref2())
        repeat_Ck.set_in_repsig(repsiggen_k.out_repsig())

        fiberlookup_Cl.set_in_ref(repeat_Ck.out_ref())

        fiberwrite_X3.set_input(fiberlookup_Cl.out_crd())

        repsiggen_l.set_istream(fiberlookup_Cl.out_crd())

        repeat_Bl.set_in_ref(fiberlookup_Bk.out_ref())
        repeat_Bl.set_in_repsig(repsiggen_l.out_repsig())

        fiberlookup_Bj.set_in_ref(repeat_Bl.out_ref())
        fiberlookup_Cj.set_in_ref(fiberlookup_Cl.out_ref())

        intersectj.set_in1(fiberlookup_Bj.out_ref(), fiberlookup_Bj.out_crd())
        intersectj.set_in2(fiberlookup_Cj.out_ref(), fiberlookup_Cj.out_crd())

        fiberwrite_X1.set_input(intersectj.out_crd())

        fiberlookup_Bm.set_in_ref(intersectj.out_ref1())
        fiberlookup_Cm.set_in_ref(intersectj.out_ref2())

        intersectm.set_in1(fiberlookup_Bm.out_ref(), fiberlookup_Bm.out_crd())
        intersectm.set_in2(fiberlookup_Cm.out_ref(), fiberlookup_Cm.out_crd())

        arrayvals_B.set_load(intersectm.out_ref1())

        arrayvals_C.set_load(intersectm.out_ref2())

        mul.set_in1(arrayvals_B.out_val())
        mul.set_in2(arrayvals_C.out_val())

        reduce.set_in_val(mul.out_val())

        fiberwrite_Xvals.set_input(reduce.out_val())

        fiberlookup_Bi.update()
        fiberlookup_Ci.update()
        intersecti.update()
        fiberwrite_X0.update()
        fiberlookup_Bk.update()
        repsiggen_k.update()
        repeat_Ck.update()
        fiberwrite_X2.update()
        fiberlookup_Cl.update()
        fiberwrite_X3.update()
        repsiggen_l.update()
        repeat_Bl.update()
        fiberlookup_Bj.update()
        fiberlookup_Cj.update()
        fiberwrite_X1.update()
        intersectj.update()
        fiberlookup_Bm.update()
        fiberlookup_Cm.update()
        intersectm.update()
        arrayvals_B.update()
        arrayvals_C.update()
        mul.update()
        reduce.update()
        fiberwrite_Xvals.update()

        done = fiberwrite_X0.out_done() and fiberwrite_X1.out_done() and fiberwrite_X2.out_done() and fiberwrite_X3.out_done and \
            fiberwrite_Xvals.out_done()

    fiberwrite_X0.autosize()
    fiberwrite_X1.autosize()
    fiberwrite_X2.autosize()
    fiberwrite_X3.autosize()
    fiberwrite_Xvals.autosize()

    out_crds = [fiberwrite_X0.get_arr(), fiberwrite_X1.get_arr(), fiberwrite_X2.get_arr(), fiberwrite_X3.get_arr()]
    out_segs = [fiberwrite_X0.get_seg_arr(), fiberwrite_X1.get_seg_arr(), fiberwrite_X2.get_seg_arr(), fiberwrite_X3.get_seg_arr()]
    out_vals = fiberwrite_Xvals.get_arr()

    print(out_segs)
    print(out_crds)
    print(out_vals)
    # print(len(out_vals))


    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_C_shape"] = C_shape
    extra_info["tensor_B/nnz"] = len(B_vals)
    extra_info["tensor_C/nnz"] = len(C_vals)

    extra_info["result/vals_size"] = len(out_vals)
    extra_info["result/nnz"] = len([x for x in out_vals if x != 0])

    sample_dict = intersecti.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersecti" + "/" + k] = sample_dict[k]

    sample_dict = intersectj.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectj" + "/" + k] = sample_dict[k]

    sample_dict = intersectm.return_statistics()
    for k in sample_dict.keys():
        extra_info["intersectm" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_X3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X3" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_B.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B" + "/" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals" + "/" + k] = sample_dict[k]

    sample_dict = arrayvals_C.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_C" + "/" + k] = sample_dict[k]

    sample_dict = mul.return_statistics()
    for k in sample_dict.keys():
        extra_info["mul" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bi.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_14" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Ci.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Ci_15" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_11" + "/" + k] = sample_dict[k]

    sample_dict = fiberlookup_Cl.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Cj_12" + "/" + k] = sample_dict[k]

    # sample_dict = fiberlookup_Bk_8.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberlookup_Bk_8" + "/" + k] = sample_dict[k]

    # sample_dict = fiberlookup_Ck_9.return_statistics()
    # for k in sample_dict.keys():
    #     extra_info["fiberlookup_Ck_9" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_elemadd(frosttname, debug_sim, out_crds, out_segs, out_vals, "sss012")
    samBench(bench, extra_info)
