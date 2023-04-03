import copy
import pytest

import time
from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.reorder import Reorder_and_split, repeated_token_dopper

from sam.sim.src.reorder_baseline import * #Reorder_and_split, repeated_token_dopper
from sam.sim.src.split import *
from sam.sim.src.wr_scanner import *
from sam.sim.src.base import remove_emptystr
from sam.sim.src.base import *
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
def test_tiling(samBench, ssname, check_gold, report_stats, debug_sim, reorder_not_ideal, reorder_block_len, split_factor, fill=0):
    # split_factor = split_factor #6 # * 128
    split_factor = int(split_factor)
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
    split_block_0 = Split(split_factor=split_factor, takes_ref=False, debug=debug_sim) #debug_sim)
    rdB_1 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1)
    #crdscan = Reorder_and_split(seg_arr=B_seg1, crd_arr=B_crd1, not_idealized=reorder_not_ideal, block_size_len=reorder_block_len, sf=split_factor, debug=debug_sim, alpha=1, statistics=True)
    split_block_1 = Split(split_factor=split_factor, takes_ref=False, debug=debug_sim) #debug_sim)
    
    
    
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
    #print("@@@@@@@@@@@@@@@@@")
    done = False
    time_cnt = 0
    in_ref = [0, "D"]
    out_crd = []
    out_crd_i = []
    out_crd_k_out = []
    out_crd_i_out = []
    temp_arr = []
    max_cnt = -1
    temp_count = 0
    average = 0
    average_len = 0
    a_crd = []



    while not done and time_cnt < TIMEOUT:
        if len(in_ref) > 0:
            rdB_0.set_in_ref(in_ref.pop(0))

        split_block_0.set_in_crd(rdB_0.out_crd())
        rdB_1.set_in_ref(rdB_0.out_ref())
        #split_block.set_in_ref(rdB_0.out_ref())
        fiberwrite_X0.set_input(split_block_0.out_outer_crd())
        #print(rdB_1.out_crd(), " ", split_block_1.out_inner_crd(), split_block_1.out_outer_crd())
        split_block_1.set_in_crd(rdB_1.out_crd())
        fiberwrite_X1.set_input(split_block_0.out_inner_crd())
        fiberwrite_X2.set_input(split_block_1.out_outer_crd())
        fiberwrite_X3.set_input(split_block_1.out_inner_crd())
        
        # rdB_0.out_ref()
        #temp_arr.append(split_block.out_inner_crd())
        #crdscan.set_input(split_block.out_inner_crd(), split_block.out_inner_ref())
        
        #crd_k.add_token(crdscan.out_crd_k())
        #ref_k.add_token(crdscan.out_ref_k())
        #crd_i.add_token(crdscan.out_crd_i())
        #ref_i.add_token(crdscan.out_ref_i())
        #crd_k_out.add_token(crdscan.out_crd_k_outer())
        #ref_k_out.add_token(crdscan.out_ref_k_outer())
        
        #fiberwrite_X1.set_input(crd_k_out.get_token())
        #fiberwrite_X2.set_input(crd_i.get_token())
        #fiberwrite_X3.set_input(crd_k.get_token())
        #arrayvals.set_load(ref_k.get_token())
        #fiberwrite_Xvals.set_input(arrayvals.out_val())

        rdB_0.update()
        split_block_0.update()
        #crdscan.update()
        rdB_1.update()
        split_block_1.update()

        #crd_k.update()
        #ref_k.update()
        #crd_i.update()
        #ref_i.update()
        #crd_k_out.update()
        #ref_k_out.update()
        fiberwrite_X0.update()
        fiberwrite_X1.update()
        fiberwrite_X2.update()
        fiberwrite_X3.update()
        #arrayvals.update()
        #fiberwrite_Xvals.update()
        done = fiberwrite_X0.done and fiberwrite_X1.done and fiberwrite_X0.done and fiberwrite_X1.done and fiberwrite_X2.done and fiberwrite_X3.done
        time_cnt += 1
        #if debug_sim:
        #print("Timestep", time_cnt, max_cnt, " ", average / (average_len + 1), "\t k_out_crd:", crdscan.out_crd_k_outer(), "\t k_out_ref:", crdscan.out_ref_k_outer(), "\t Crd i:", crdscan.out_crd_i(), "\t Ref i:", crdscan.out_ref_i(), "\t Crd:", crdscan.out_crd_k(), "\t Ref:", crdscan.out_ref_k())
        #print(a_crd)
        #print("______________________________________________________________________")

    fiberwrite_X0.autosize()
    fiberwrite_X1.autosize()
    fiberwrite_X2.autosize()
    fiberwrite_X3.autosize()
    #print(fiberwrite_X0.get_arr(), fiberwrite_X1.get_arr(), fiberwrite_X2.get_arr(), fiberwrite_X3.get_arr())

    rd_scan_0 = CompressedCrdRdScan(crd_arr=fiberwrite_X0.get_arr(), seg_arr=fiberwrite_X0.get_seg_arr())  
    rd_scan_1 = CompressedCrdRdScan(crd_arr=fiberwrite_X1.get_arr(), seg_arr=fiberwrite_X1.get_seg_arr())  
    crdscan = Reorder_baseline(crd_arr=fiberwrite_X2.get_arr(), seg_arr=fiberwrite_X2.get_seg_arr(), sf=split_factor, debug=debug_sim)
    # rd_scan_2 = CompressedCrdRdScan(crd_arr=fiberwrite_X2.get_arr(), seg_arr=fiberwrite_X2.get_seg_arr())  
    rd_scan_3 = CompressedCrdRdScan(crd_arr=fiberwrite_X3.get_arr(), seg_arr=fiberwrite_X3.get_seg_arr())  
    #out_crds = [fiberwrite_X0.get_arr(), fiberwrite_X1_1.get_arr()]
    #out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]

    fiberwrite_X3 = CompressWrScan(seg_size=Bs_seg[-1] + 1, size=Bs_seg[-1] * Bs_seg[-1], fill=fill,
                                   debug=debug_sim, statistics=report_stats, name="X3")
    fiberwrite_X2 = CompressWrScan(seg_size=(Bs_seg[-1]//split_factor + 1)**2, size=Bs_seg[-1],
                                   fill=fill, debug=debug_sim, statistics=report_stats, name="X2")
    fiberwrite_X1 = CompressWrScan(seg_size=Bs_seg[-1]//split_factor + 1,
                                   size=(Bs_seg[-1]//split_factor + 1)**2, fill=fill,
                                   debug=debug_sim, statistics=report_stats, name="X1")

    in_ref = [0, "D"]
    done = False
    while not done:
        if len(in_ref) > 0:
            rd_scan_0.set_in_ref(in_ref.pop(0))

        rd_scan_1.set_in_ref(rd_scan_0.out_ref())

        crdscan.input_crd(rd_scan_1.out_crd())
        crdscan.input_ref(rd_scan_1.out_ref())
        rd_scan_3.set_in_ref(crdscan.return_final_ref())
        arrayvals.set_load(rd_scan_3.out_ref())
        fiberwrite_X1.set_input(crdscan.return_final_crd())
        fiberwrite_X2.set_input(crdscan.out_ocrd_i())
        fiberwrite_X3.set_input(rd_scan_3.out_crd())
        fiberwrite_Xvals.set_input(arrayvals.out_val())

        rd_scan_0.update()
        rd_scan_1.update()
        crdscan.update()
        rd_scan_3.update()
        arrayvals.update()
        fiberwrite_X1.update()
        fiberwrite_X2.update()
        fiberwrite_X3.update()
        fiberwrite_Xvals.update()
        done = fiberwrite_Xvals.done
        time_cnt += 1


    def bench():
        time.sleep(0.01)

    fiberwrite_X1.autosize()
    fiberwrite_X2.autosize()
    fiberwrite_X3.autosize()
    #print(fiberwrite_X0.get_arr(), fiberwrite_X1.get_arr(), fiberwrite_X2.get_arr(), fiberwrite_X3.get_arr())
    print(time_cnt)
    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["cycles"] = time_cnt
    extra_info["max_tile_size"] = max_cnt
    # extra_info["avg_tile_size"] = average / average_len
    extra_info["total_tile_cnts"] = average_len
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_B/nnz"] = len(B_vals)

    sample_dict = crdscan.return_statistics()
    for k in sample_dict.keys():
        extra_info["reorder_block" + "/" + k] = sample_dict[k]

    samBench(bench, extra_info)
    print("Done and time: ", done, time_cnt)
