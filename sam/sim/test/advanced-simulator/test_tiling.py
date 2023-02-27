import copy
import pytest

import time
from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.reorder import Reorder_and_split, repeated_token_dopper
from sam.sim.src.split import *
from sam.sim.src.wr_scanner import *
from sam.sim.src.base import remove_emptystr
from sam.onyx.generate_matrices import *
import os
import csv

cwd = os.getcwd()

formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH',  default=os.path.join(cwd, 'mode-formats'))

@pytest.mark.skipif(
        os.getenv('CI', 'false') == 'true',
        reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_tiling(samBench, ssname, check_gold, report_stats, debug_sim, split_factor=16, fill=0):
    # split_factor = split_factor #6 # * 128
    B_dirname = os.path.join(formatted_dir, ssname, "mat_identity") # "orig", "ss01")
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

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)


    rdB_0 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0)
    split_block = Split(split_factor=split_factor)
    crdscan = Reorder_and_split(seg_arr=B_seg1, crd_arr=B_crd1, limit=10, sf=split_factor, debug=debug_sim, statistics=True)
    crd_k = repeated_token_dopper(name="crdk")
    ref_k = repeated_token_dopper(name="refk")
    crd_i = repeated_token_dopper(name="crdi")
    ref_i = repeated_token_dopper(name="refi")
    crd_k_out = repeated_token_dopper(name="crdkout")
    ref_k_out = repeated_token_dopper(name="refkout")
    
    # THIS IS FOR SIZE INFO
    Bs_dirname = B_dirname 
    # os.path.join(formatted_dir, ssname, "orig", "ss01")
    Bs_seg = read_inputs(os.path.join(Bs_dirname, "tensor_B_mode_0_seg"))

    arrayvals = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals = ValsWrScan(size=1 * Bs_seg[-1] * Bs_seg[-1],
                                  fill=fill, debug=debug_sim,
                                  statistics=report_stats)
    fiberwrite_X3 = CompressWrScan(seg_size=Bs_seg[-1] + 1, size=Bs_seg[-1] * Bs_seg[-1], fill=fill,
                                     debug=debug_sim, statistics=report_stats, name="X3")
    fiberwrite_X2 = CompressWrScan(seg_size=(Bs_seg[-1]//split_factor + 1)**2, size=Bs_seg[-1],
                                   fill=fill, debug=debug_sim, statistics=report_stats, name="X2")
 
    fiberwrite_X1 = CompressWrScan(seg_size=Bs_seg[-1]//split_factor + 1,
                                   size=(Bs_seg[-1]//split_factor + 1)**2, fill=fill,
                                   debug=debug_sim, statistics=report_stats, name="X1")
    fiberwrite_X0 = CompressWrScan(seg_size=2, size=Bs_seg[-1]//split_factor + 1,
                                   fill=fill, debug=debug_sim, statistics=report_stats,
                                   name="X0")
    done = False
    time_cnt = 0
    in_ref = [0, "D"]
    out_crd = []
    out_crd_i = []
    out_crd_k_out = []
    out_crd_i_out = []
    temp_arr = []
    while not done and time_cnt < TIMEOUT:
        if len(in_ref) > 0:
            rdB_0.set_in_ref(in_ref.pop(0))
        split_block.set_in_crd(rdB_0.out_crd())        
        fiberwrite_X0.set_input(split_block.out_outer_crd())

        temp_arr.append(split_block.out_inner_crd())
        crdscan.set_input(rdB_0.out_ref(), split_block.out_inner_crd())
        crd_k.add_token(crdscan.out_crd_k())
        ref_k.add_token(crdscan.out_ref_k())
        crd_i.add_token(crdscan.out_crd_i())
        ref_i.add_token(crdscan.out_ref_i())
        crd_k_out.add_token(crdscan.out_crd_k_outer())
        ref_k_out.add_token(crdscan.out_ref_k_outer())
        

        fiberwrite_X1.set_input(crd_k_out.get_token())
        fiberwrite_X2.set_input(crd_i.get_token())
        fiberwrite_X3.set_input(crd_k.get_token())
        arrayvals.set_load(ref_k.get_token())
        fiberwrite_Xvals.set_input(arrayvals.out_val())

        if crd_k.get_token() != "":
            out_crd.append(crd_k.get_token())
        if crd_i.get_token() != "":
            out_crd_i.append(crd_i.get_token())
        if crd_k_out.get_token() != "":
            out_crd_k_out.append(crd_k_out.get_token())
        if split_block.out_outer_crd() != "":
            out_crd_i_out.append(split_block.out_outer_crd())
     
        rdB_0.update()
        split_block.update()
        crdscan.update()
        crd_k.update()
        ref_k.update()
        crd_i.update()
        ref_i.update()
        crd_k_out.update()
        ref_k_out.update()
        fiberwrite_X0.update()
        fiberwrite_X1.update()
        fiberwrite_X2.update()
        fiberwrite_X3.update()
        arrayvals.update()
        fiberwrite_Xvals.update()
        done = fiberwrite_X0.done and fiberwrite_X1.done and fiberwrite_X0.done and fiberwrite_X1.done and fiberwrite_X2.done and fiberwrite_X3.done
        time_cnt += 1

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_B/nnz"] = len(B_vals)

    sample_dict = crdscan.return_statistics()
    for k in sample_dict.keys():
        extra_info["reorder_block" + "/" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_matmul(ssname, debug_sim, cast, out_crds, out_segs, out_vals, "ss01")
    samBench(bench, extra_info)
    print("Done and time: ", done, time)
