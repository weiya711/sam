import copy
import pytest
from collections import defaultdict
import time
import yaml
from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.channel import memory_block, output_memory_block
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
from sam.sim.src.tiling.generate_tile_crd import *
from sam.sim.src.reorder import ReorderAndSplit, RepeatedTokenDropper
from sam.sim.src.split import *
from sam.sim.src.wr_scanner import *
from sam.sim.src.base import remove_emptystr
from sam.sim.src.base import *
from sam.onyx.generate_matrices import *
import os
import csv

cwd = os.getcwd()
sam_home = os.getenv('SAM_HOME')

formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH',  default=os.path.join(cwd, 'mode-formats'))

@pytest.mark.skipif(
        os.getenv('CI', 'false') == 'true',
        reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_tiling(samBench, ssname, check_gold, report_stats, yaml_name, debug_sim, nbuffer, backpressure, depth, reorder_not_ideal, reorder_block_len, split_factor, fill=0):
    skip_empty = False
    flag = False
    tile_signalled = False
    print("############################# ", yaml_name)
    # Tile sizes at GLB and memory level
    with open(os.path.join(sam_home, "sam/sim/src/tiling/" + yaml_name), "r") as stream:
        loop_config = yaml.safe_load(stream)
    # Loading the same thing again (redundant)
    with open(os.path.join(sam_home, "./sam/sim/src/tiling/" + yaml_name), "r") as stream:
        memory_config = yaml.safe_load(stream)


    #split_factor = split_factor #6 # * 128
    split_factor = int(split_factor)
    #print("SPLIT FACTOR: ", split_factor)
    B_dirname = os.path.join(formatted_dir, ssname, "matmul_ikj") # "orig", "ss01")
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
    nnz = len(B_vals)

    C_dirname = os.path.join(formatted_dir, ssname, "matmul_ikj") # "orig", "ss01")
    C_shape_filename = os.path.join(B_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)
    C0_seg_filename = os.path.join(B_dirname, "tensor_C_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(B_dirname, "tensor_C_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename)
    C1_seg_filename = os.path.join(B_dirname, "tensor_C_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(B_dirname, "tensor_C_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename)
    C_vals_filename = os.path.join(B_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)


    rdB_0 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0)
    split_block1 = SplitRef(split_factor=split_factor, takes_ref=True, debug=debug_sim)
    crdscan = ReorderAndSplit(seg_arr=B_seg1, crd_arr=B_crd1, not_idealized=bool(reorder_not_ideal),
                                block_size_len=int(reorder_block_len), sf=split_factor,
                                debug=debug_sim, alpha=1, statistics=True)
    
    rdC_0 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0)
    split_block2 = SplitRef(split_factor=split_factor, takes_ref=True, debug=debug_sim)
    crdscanC = ReorderAndSplit(seg_arr=C_seg1, crd_arr=C_crd1, not_idealized=bool(reorder_not_ideal),
                                block_size_len=int(reorder_block_len), sf=split_factor,
                                debug=debug_sim, alpha=1, statistics=True)

    crd_k = RepeatedTokenDropper(name="crdk")
    ref_k = RepeatedTokenDropper(name="refk")
    crd_i = RepeatedTokenDropper(name="crdi")
    ref_i = RepeatedTokenDropper(name="refi")
    crd_k_out = RepeatedTokenDropper(name="crdkout")
    ref_k_out = RepeatedTokenDropper(name="refkout")

    crd_j_C = RepeatedTokenDropper(name="crdj")
    ref_j_C = RepeatedTokenDropper(name="refj")
    crd_k_C = RepeatedTokenDropper(name="crdk")
    ref_k_C = RepeatedTokenDropper(name="refk")
    crd_j_out_C = RepeatedTokenDropper(name="crdjout")
    ref_j_out_C = RepeatedTokenDropper(name="refjout")

    
    # THIS IS FOR SIZE INFO
    Bs_dirname = B_dirname 
    Cs_dirname = B_dirname 
    # os.path.join(formatted_dir, ssname, "orig", "ss01")
    Bs_seg = read_inputs(os.path.join(Bs_dirname, "tensor_B_mode_0_seg"))
    arrayvals = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    # THIS IS FOR SIZE INFO
    Cs_dirname = C_dirname
    # os.path.join(formatted_dir, ssname, "orig", "ss01")
    Cs_seg = read_inputs(os.path.join(Cs_dirname, "tensor_C_mode_0_seg"))
    arrayvals_C = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)

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

    fiberwrite_Yvals = ValsWrScan(size=1 * Cs_seg[-1] * Cs_seg[-1],
                                  fill=fill, debug=debug_sim,
                                  statistics=report_stats)
    fiberwrite_Y3 = CompressWrScan(seg_size=Cs_seg[-1] + 1, size=Cs_seg[-1] * Cs_seg[-1], fill=fill,
                                     debug=debug_sim, statistics=report_stats, name="Y3")
    fiberwrite_Y2 = CompressWrScan(seg_size=(Cs_seg[-1]//split_factor + 1)**2, size=Cs_seg[-1],
                                   fill=fill, debug=debug_sim, statistics=report_stats, name="Y2")
    fiberwrite_Y1 = CompressWrScan(seg_size=Cs_seg[-1]//split_factor + 1,
                                   size=(Cs_seg[-1]//split_factor + 1)**2, fill=fill,
                                   debug=debug_sim, statistics=report_stats, name="Y1")
    fiberwrite_Y0 = CompressWrScan(seg_size=2, size=Cs_seg[-1]//split_factor + 1,
                                   fill=fill, debug=debug_sim, statistics=report_stats,
                                   name="Y0")

    done = False
    time_cnt = 0
    in_ref = [0, "D"]
    in_ref2 = [0, "D"]
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

    full_arr0 = []
    full_arr1 = []
    full_arr2 = []
    full_arr3 = []

    sizes_list1 = []
    sizes_list2 = []
    #print("B_crd0", B_seg0, B_crd0)
    #print("B_crd1", B_seg1, B_crd1)
    while not done and time_cnt < TIMEOUT:
        if len(in_ref) > 0:
            rdB_0.set_in_ref(in_ref.pop(0))
        split_block1.set_in_crd(rdB_0.out_crd())
        split_block1.set_in_ref(rdB_0.out_ref())
        fiberwrite_X0.set_input(split_block1.out_outer_crd())
        temp_arr.append(split_block1.out_inner_crd())
        crdscan.set_input(split_block1.out_inner_ref(), split_block1.out_inner_crd())
        crd_k.add_token(crdscan.out_crd_k())
        ref_k.add_token(crdscan.out_ref_k())
        crd_i.add_token(crdscan.out_crd_i())
        ref_i.add_token(crdscan.out_ref_i())
        crd_k_out.add_token(crdscan.out_crd_k_outer())
        ref_k_out.add_token(crdscan.out_ref_k_outer())
        
        if len(in_ref2) > 0:
            rdC_0.set_in_ref(in_ref2.pop(0))
        split_block2.set_in_crd(rdC_0.out_crd())
        split_block2.set_in_ref(rdC_0.out_ref())
        fiberwrite_Y0.set_input(split_block2.out_outer_crd())
        crdscanC.set_input(split_block2.out_inner_ref(), split_block2.out_inner_crd())
        crd_j_C.add_token(crdscanC.out_crd_k())
        ref_j_C.add_token(crdscanC.out_ref_k())
        crd_k_C.add_token(crdscanC.out_crd_i())
        ref_k_C.add_token(crdscanC.out_ref_i())
        crd_j_out_C.add_token(crdscanC.out_crd_k_outer())
        ref_j_out_C.add_token(crdscanC.out_ref_k_outer())


        fiberwrite_X1.set_input(crd_k_out.get_token())
        fiberwrite_X2.set_input(crd_i.get_token())
        fiberwrite_X3.set_input(crd_k.get_token())
        arrayvals.set_load(ref_k.get_token())

        fiberwrite_Xvals.set_input(arrayvals.out_val())
        fiberwrite_Y1.set_input(crd_j_out_C.get_token())
        fiberwrite_Y2.set_input(crd_k_C.get_token())
        fiberwrite_Y3.set_input(crd_j_C.get_token())
        arrayvals_C.set_load(ref_j_C.get_token())
        fiberwrite_Yvals.set_input(arrayvals_C.out_val())

        if split_block1.out_outer_crd() != "":
            full_arr0.append(split_block1.out_outer_crd())
        if crd_k_out.get_token() != "":
            full_arr1.append(crd_k_out.get_token())
        if crd_i.get_token() != "":
            full_arr2.append(crd_i.get_token())
        if crd_k.get_token() != "":
            full_arr3.append(crd_k.get_token())

        if ref_k.get_token() != "":
            out_crd.append(crd_k.get_token())
            if True: #not is_stkn(crd_k.get_token()):
                a_crd.append(crd_k.get_token())
                if isinstance(crd_k.get_token(), int):
                    temp_count += 1
            if crd_k.get_token() == "S1" or crd_k.get_token() == "S2" or crd_k.get_token() == "S3" or crd_k.get_token() == "S4":
                #print("out_crd_i_out", out_crd_i_out)
                #print("out_crd_k_out", out_crd_k_out)
                #print("out_crd_i", out_crd_i)
                #print("out_crd_k", out_crd)
                #print(ref_k.get_token())
                #print("___________________________________________________________")
                max_cnt = max(max_cnt, temp_count)
                average += temp_count
                average_len += 1
                # print("TEMP PRINT", " nnz ",  temp_count, " nnz + stkns ",  len(a_crd), " avg: ",  average / average_len)
                # print(a_crd)
                temp_count = 0
                a_crd = []

        if crd_i.get_token() != "":
            out_crd_i.append(crd_i.get_token())
        if crd_k_out.get_token() != "":
            out_crd_k_out.append(crd_k_out.get_token())
        if split_block1.out_outer_crd() != "":
            out_crd_i_out.append(split_block1.out_outer_crd())
     

        rdB_0.update()
        split_block1.update()
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

        rdC_0.update()
        split_block2.update()
        crdscanC.update()
        crd_j_C.update()
        ref_j_C.update()
        crd_k_C.update()
        ref_k_C.update()
        crd_j_out_C.update()
        ref_j_out_C.update()
        fiberwrite_Y0.update()
        fiberwrite_Y1.update()
        fiberwrite_Y2.update()
        fiberwrite_Y3.update()
        arrayvals_C.update()
        fiberwrite_Yvals.update()


        done = fiberwrite_X0.done and fiberwrite_X1.done and fiberwrite_X0.done and fiberwrite_X1.done and fiberwrite_X2.done and fiberwrite_X3.done
        done2 = fiberwrite_Y0.done and fiberwrite_Y1.done and fiberwrite_Y0.done and fiberwrite_Y1.done and fiberwrite_Y2.done and fiberwrite_Y3.done
        done = done and done2
        time_cnt += 1
        if debug_sim:
            print("Timestep", time_cnt, max_cnt, " ", average / (average_len + 1),
                  "\t k_out_crd:", crdscan.out_crd_k_outer(), "\t k_out_ref:",
                  crdscan.out_ref_k_outer(), "\t Crd i:", crdscan.out_crd_i(),
                  "\t Ref i:", crdscan.out_ref_i(), "\t Crd:", crdscan.out_crd_k(),
                  "\t Ref:", crdscan.out_ref_k())
            print(a_crd)
            print(temp_count, max_cnt)
            print("t_arr ", full_arr0, full_arr1, full_arr2, full_arr3)
            print("______________________________________________________________________", time_cnt)
    

    fiberwrite_X0.autosize()
    fiberwrite_X1.autosize()
    fiberwrite_X2.autosize()
    fiberwrite_X3.autosize()
    fiberwrite_Xvals.autosize()

    fiberwrite_Y0.autosize()
    fiberwrite_Y1.autosize()
    fiberwrite_Y2.autosize()
    fiberwrite_Y3.autosize()
    fiberwrite_Yvals.autosize()
    #print("ARRAYS", max_cnt)
    if False:
        print(fiberwrite_X0.get_arr(), fiberwrite_X1.get_arr(), fiberwrite_X2.get_arr(), fiberwrite_X3.get_arr(), fiberwrite_Xvals.get_arr())
        print(fiberwrite_Y0.get_arr(), fiberwrite_Y1.get_arr(), fiberwrite_Y2.get_arr(), fiberwrite_Y3.get_arr(), fiberwrite_Yvals.get_arr())
        print(fiberwrite_X0.get_seg_arr(), fiberwrite_X1.get_seg_arr(), fiberwrite_X2.get_seg_arr(), fiberwrite_X3.get_seg_arr())
        print(fiberwrite_Y0.get_seg_arr(), fiberwrite_Y1.get_seg_arr(), fiberwrite_Y2.get_seg_arr(), fiberwrite_Y3.get_seg_arr())
    dict_x = {}
    dict_y = {}
    dict_seg_arr_B0 = defaultdict(dict)
    dict_arr_B0 = defaultdict(dict)
    dict_seg_arr_B1 = defaultdict(dict)
    dict_arr_B1 = defaultdict(dict)
    dict_vals_B = defaultdict(dict)
    dict_seg_arr_B1_  = defaultdict(dict)
    
    dict_seg_arr_C0 = defaultdict(dict)
    dict_arr_C0 = defaultdict(dict)
    dict_seg_arr_C1 = defaultdict(dict)
    dict_arr_C1 = defaultdict(dict)
    dict_vals_C = defaultdict(dict)
    dict_seg_arr_C1_  = defaultdict(dict)

    for i in range(len(fiberwrite_X2.get_seg_arr()) - 1):
        size = fiberwrite_X3.get_seg_arr()[fiberwrite_X2.get_seg_arr()[i+1]] - fiberwrite_X3.get_seg_arr()[fiberwrite_X2.get_seg_arr()[i]]
        dict_x[i] = size
        dict_seg_arr_B0[i] = [fiberwrite_X2.get_seg_arr()[i], fiberwrite_X2.get_seg_arr()[i+1]]
        dict_arr_B0[i] = fiberwrite_X2.get_arr()[fiberwrite_X2.get_seg_arr()[i] : fiberwrite_X2.get_seg_arr()[i+1]] 
        dict_seg_arr_B1[i] = fiberwrite_X3.get_seg_arr()[fiberwrite_X2.get_seg_arr()[i] : fiberwrite_X2.get_seg_arr()[i+1] + 1]
        dict_arr_B1[i] = fiberwrite_X3.get_arr()[dict_seg_arr_B1[i][0] : dict_seg_arr_B1[i][-1]] 
        dict_vals_B[i] = fiberwrite_Xvals.get_arr()[dict_seg_arr_B1[i][0] : dict_seg_arr_B1[i][-1]]
        dict_seg_arr_B0[i] = [0, dict_seg_arr_B0[i][1] - dict_seg_arr_B0[i][0]]
        dict_seg_arr_B1_[i] = []
        for k in range(len(dict_seg_arr_B1[i])):
            dict_seg_arr_B1_[i].append(dict_seg_arr_B1[i][k] - dict_seg_arr_B1[i][0])
        dict_seg_arr_B1[i] = dict_seg_arr_B1_[i]

    for j in range(len(fiberwrite_Y2.get_seg_arr()) - 1):
        size = fiberwrite_Y3.get_seg_arr()[fiberwrite_Y2.get_seg_arr()[j+1]] - fiberwrite_Y3.get_seg_arr()[fiberwrite_Y2.get_seg_arr()[j]]
        dict_y[j] = size
        dict_seg_arr_C0[j] = [fiberwrite_Y2.get_seg_arr()[j], fiberwrite_Y2.get_seg_arr()[j+1]]
        dict_arr_C0[j] = fiberwrite_Y2.get_arr()[fiberwrite_Y2.get_seg_arr()[j] : fiberwrite_Y2.get_seg_arr()[j+1]]
        dict_seg_arr_C1[j] = fiberwrite_Y3.get_seg_arr()[fiberwrite_Y2.get_seg_arr()[j] : fiberwrite_Y2.get_seg_arr()[j+1] + 1]
        dict_arr_C1[j] = fiberwrite_Y3.get_arr()[dict_seg_arr_C1[j][0] : dict_seg_arr_C1[j][-1]]
        dict_vals_C[j] = fiberwrite_Xvals.get_arr()[dict_seg_arr_C1[j][0] : dict_seg_arr_C1[j][-1]]
        dict_seg_arr_C0[j] = [0, dict_seg_arr_C0[j][1] - dict_seg_arr_C0[j][0]]
        dict_seg_arr_C1_[j] = []
        for k in range(len(dict_seg_arr_C1[j])):
            dict_seg_arr_C1_[j].append(dict_seg_arr_C1[j][k] - dict_seg_arr_C1[j][0])
        dict_seg_arr_C1[j] = dict_seg_arr_C1_[j]
    if False:
        print(dict_x)
        print(dict_y)

        print(dict_seg_arr_B0)
        print(dict_arr_B0)
        print(dict_seg_arr_B1)
        print(dict_arr_B1)
        ###########
        print("#############")
        print(dict_seg_arr_C0)
        print(dict_arr_C0)
        print(dict_seg_arr_C1)
        print(dict_arr_C1)
    
    
    # Intitialize software loops
    fiberlookup_Bi00 = CompressedCrdRdScan(crd_arr=fiberwrite_X0.get_arr(), seg_arr=fiberwrite_X0.get_seg_arr(), debug=debug_sim)
    fiberlookup_Bk00 = CompressedCrdRdScan(crd_arr=fiberwrite_X1.get_arr(), seg_arr=fiberwrite_X1.get_seg_arr(), debug=debug_sim)
    repsiggen_i00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ci00 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck00 = CompressedCrdRdScan(crd_arr=fiberwrite_Y0.get_arr(), seg_arr=fiberwrite_Y0.get_seg_arr(), debug=debug_sim)
    intersect_00 = Intersect2(debug=debug_sim)
    fiberlookup_Cj00 = CompressedCrdRdScan(crd_arr=fiberwrite_Y1.get_arr(), seg_arr=fiberwrite_Y1.get_seg_arr(), debug=debug_sim)
    repsiggen_j00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bj00 = Repeat(debug=debug_sim, statistics=report_stats)

    # Reference for software loops
    in_ref_B00 = [0, 'D']
    in_ref_C00 = [0, 'D']
    # done = False
    time_cnt2 = 0
    # Intitialize memory blocks
    mem_model_b = memory_block(name="B", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"],
                               size=memory_config["Mem_memory"],
                               bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                               latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_c = memory_block(name="C", skip_blocks=skip_empty, nbuffer=nbuffer,
                               element_size=memory_config["Bytes_per_element"],
                               size=memory_config["Mem_memory"],
                               bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                               latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_bk = memory_block(name="Bk", skip_blocks=skip_empty, nbuffer=nbuffer,
                                element_size=memory_config["Bytes_per_element"],
                                size=memory_config["Mem_memory"],
                                bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                                latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_ck = memory_block(name="Ck", skip_blocks=skip_empty, nbuffer=nbuffer,
                                element_size=memory_config["Bytes_per_element"],
                                size=memory_config["Mem_memory"],
                                bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                                latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_bi = memory_block(name="Bi", skip_blocks=skip_empty, nbuffer=nbuffer,
                                element_size=memory_config["Bytes_per_element"],
                                size=memory_config["Mem_memory"],
                                bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                                latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_cj = memory_block(name="Cj", skip_blocks=skip_empty, nbuffer=nbuffer,
                                element_size=memory_config["Bytes_per_element"],
                                size=memory_config["Mem_memory"],
                                bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                                latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_bvals = memory_block(name="Bvals", skip_blocks=skip_empty, nbuffer=nbuffer,
                                   element_size=memory_config["Bytes_per_element"],
                                   size=memory_config["Mem_memory"],
                                   bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                                   latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    mem_model_cvals = memory_block(name="Cvals", skip_blocks=skip_empty, nbuffer=nbuffer,
                                   element_size=memory_config["Bytes_per_element"],
                                   size=memory_config["Mem_memory"],
                                   bandwidth=memory_config["Mem_tile_bandwidth"] / memory_config["Mem_tiles"],
                                   latency=memory_config["Glb_Mem_latency"], debug=debug_sim, statistics=report_stats)
    nxt_tile_present = [True] * 6
    mem_blocks_decl_flag = False
    done = False
    debug_sim2 = False # True
    while not done and time_cnt2 < TIMEOUT:
        if debug_sim:
            print(time_cnt2)
        # Software loop update every cycle
        if len(in_ref_B00) > 0:
            fiberlookup_Bi00.set_in_ref(in_ref_B00.pop(0))
        fiberlookup_Bk00.set_in_ref(fiberlookup_Bi00.out_ref())
        repsiggen_i00.set_istream(fiberlookup_Bi00.out_crd())
        if len(in_ref_C00) > 0:
            repeat_Ci00.set_in_ref(in_ref_C00.pop(0))
        repeat_Ci00.set_in_repsig(repsiggen_i00.out_repsig())
        fiberlookup_Ck00.set_in_ref(repeat_Ci00.out_ref())
        intersect_00.set_in1(fiberlookup_Ck00.out_ref(), fiberlookup_Ck00.out_crd())
        intersect_00.set_in2(fiberlookup_Bk00.out_ref(), fiberlookup_Bk00.out_crd())
        fiberlookup_Cj00.set_in_ref(intersect_00.out_ref1())
        repsiggen_j00.set_istream(fiberlookup_Cj00.out_crd())
        repeat_Bj00.set_in_ref(intersect_00.out_ref2())
        repeat_Bj00.set_in_repsig(repsiggen_j00.out_repsig())


        if isinstance(repeat_Bj00.out_ref(), int):
            # Get size of tile from a the datastructure
            # Get the coordinates for the tile
            #print("ADDING TILE B ", repeat_Bj00.out_ref(), " ", dict_x[repeat_Bj00.out_ref()])
            mem_model_b.add_tile(repeat_Bj00.out_ref(), dict_x[repeat_Bj00.out_ref()] * 2)
            mem_model_bvals.add_tile(repeat_Bj00.out_ref(), dict_x[repeat_Bj00.out_ref()] * 2)
            mem_model_bi.add_tile(repeat_Bj00.out_ref(), dict_x[repeat_Bj00.out_ref()] * 2)
            mem_model_bk.add_tile(repeat_Bj00.out_ref(), dict_x[repeat_Bj00.out_ref()] * 2)
        elif repeat_Bj00.out_ref() != "":
            # Add Done token with size 8
            # print("ADDING TILE B ", repeat_Bj00.out_ref(), " ", 1)
            mem_model_b.add_tile(repeat_Bj00.out_ref(), 8)
            mem_model_bvals.add_tile(repeat_Bj00.out_ref(), 8)
            mem_model_bi.add_tile(repeat_Bj00.out_ref(), 8)
            mem_model_bk.add_tile(repeat_Bj00.out_ref(), 8)

        if isinstance(fiberlookup_Cj00.out_ref(), int):
            # Add an actual tile for the datastructure
            # Get crds from the reference
            #print("ADDING TILE C ", fiberlookup_Cj00.out_ref(), " ", dict_y[fiberlookup_Cj00.out_ref()] * 32)
            mem_model_c.add_tile(fiberlookup_Cj00.out_ref(), dict_y[fiberlookup_Cj00.out_ref()] * 2)
            mem_model_cvals.add_tile(fiberlookup_Cj00.out_ref(),  dict_y[fiberlookup_Cj00.out_ref()] * 2)
            mem_model_cj.add_tile(fiberlookup_Cj00.out_ref(), dict_y[fiberlookup_Cj00.out_ref()] * 2)
            mem_model_ck.add_tile(fiberlookup_Cj00.out_ref(), dict_y[fiberlookup_Cj00.out_ref()] * 2)
        elif fiberlookup_Cj00.out_ref() != "":
            # Add Done tile
            #print("ADDING TILE C ", fiberlookup_Cj00.out_ref(), " ", 1)
            mem_model_c.add_tile(fiberlookup_Cj00.out_ref(), 8)
            mem_model_cvals.add_tile(fiberlookup_Cj00.out_ref(), 8)
            mem_model_cj.add_tile(fiberlookup_Cj00.out_ref(), 8)
            mem_model_ck.add_tile(fiberlookup_Cj00.out_ref(), 8)
        # Evict tile and move ahead
        if not mem_blocks_decl_flag and mem_model_b.valid_tile() and mem_model_c.valid_tile():
            # Valid tile is in glb
            # get keys
            mem_blocks_decl_flag = True
            # Initialize memory array
            # Get an array that allows us to get the cordinates from the reference values
            # Called ref_to_crd_convertor to be able to use the sizes_dict_level0
            # Get seg and crd arrays of mem tiles in mem_arr
            flag = True
            in_ref_B_ = [0, 'D']
            in_ref_C_ = [0, 'D']
            B_crd0 = dict_arr_B0[mem_model_b.token()]
            B_seg0 = dict_seg_arr_B0[mem_model_b.token()]
            B_crd1 = dict_arr_B1[mem_model_b.token()]
            B_seg1 = dict_seg_arr_B1[mem_model_b.token()]
            B_vals = dict_vals_B[mem_model_b.token()]
            C_crd0 = dict_arr_C0[mem_model_c.token()]
            C_seg0 = dict_seg_arr_C0[mem_model_c.token()]
            C_crd1 = dict_arr_C1[mem_model_c.token()]
            C_seg1 = dict_seg_arr_C1[mem_model_c.token()]
            C_vals = dict_vals_C[mem_model_c.token()]
            

            #print("B: ", B_crd0, B_crd1, B_vals)
            #print("B: ", C_crd0, C_crd1, C_vals)
            #print("starting tile ")
            fiberlookup_Bi_19 = CompressedCrdRdScan(name="Bi", crd_arr=B_crd0, seg_arr=B_seg0,
                                                    debug=debug_sim2, statistics=report_stats,
                                                    back_en=backpressure, depth=depth)
            fiberlookup_Bk_14 = CompressedCrdRdScan(name="Bk", crd_arr=B_crd1, seg_arr=B_seg1,
                                                    debug=debug_sim2, statistics=report_stats,
                                                    back_en=backpressure, depth=depth)
            repsiggen_i_17 = RepeatSigGen(debug=debug_sim2, statistics=report_stats,
                                          back_en=backpressure, depth=depth)
            repeat_Ci_16 = Repeat(debug=debug_sim2, statistics=report_stats,
                                  back_en=backpressure, depth=depth)
            fiberlookup_Ck_15 = CompressedCrdRdScan(name="Ck", crd_arr=C_crd0, seg_arr=C_seg0,
                                                    debug=debug_sim2, statistics=report_stats,
                                                    back_en=backpressure, depth=depth)
            intersectk_13 = Intersect2(debug=debug_sim2, statistics=report_stats,
                                       back_en=backpressure, depth=depth)
            crdhold_5 = CrdHold(debug=debug_sim2, statistics=report_stats, back_en=backpressure,
                                depth=depth)
            fiberlookup_Cj_12 = CompressedCrdRdScan(name="Cj", crd_arr=C_crd1, seg_arr=C_seg1,
                                                    debug=debug_sim2, statistics=report_stats,
                                                    back_en=backpressure, depth=depth)
            arrayvals_C_8 = Array(name="C", init_arr=C_vals, debug=debug_sim2, statistics=report_stats,
                                  back_en=backpressure, depth=depth)
            crdhold_4 = CrdHold(debug=debug_sim2, statistics=report_stats, back_en=backpressure, depth=depth)
            repsiggen_j_10 = RepeatSigGen(debug=debug_sim2, statistics=report_stats, back_en=backpressure,
                                          depth=depth)
            repeat_Bj_9 = Repeat(debug=debug_sim2, statistics=report_stats, back_en=backpressure, depth=depth)
            arrayvals_B_7 = Array(name="B", init_arr=B_vals, debug=debug_sim2, statistics=report_stats,
                                  back_en=backpressure, depth=depth)
            mul_6 = Multiply2(debug=debug_sim2, statistics=report_stats, back_en=backpressure, depth=depth)
            spaccumulator1_3 = SparseAccumulator1(debug=debug_sim2, statistics=report_stats,
                                                  back_en=backpressure, depth=depth)
            spaccumulator1_3_drop_crd_inner = StknDrop(debug=debug_sim2, statistics=report_stats,
                                                       back_en=backpressure, depth=depth)
            spaccumulator1_3_drop_crd_outer = StknDrop(debug=debug_sim2, statistics=report_stats,
                                                       back_en=backpressure, depth=depth)
            spaccumulator1_3_drop_val = StknDrop(debug=debug_sim2, statistics=report_stats,
                                                 back_en=backpressure, depth=depth)
            fiberwrite_Xvals_0 = ValsWrScan(name="vals", size=1 * B_shape[0] * C_shape[1], fill=fill,
                                            debug=debug_sim2, statistics=report_stats,
                                            back_en=backpressure, depth=depth)
            fiberwrite_X1_1 = CompressWrScan(name="X1", seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[1],
                                             fill=fill, debug=debug_sim2, statistics=report_stats,
                                             back_en=backpressure, depth=depth)
            fiberwrite_X0_2 = CompressWrScan(name="X0", seg_size=2, size=B_shape[0],
                                             fill=fill, debug=debug_sim2, statistics=report_stats,
                                             back_en=backpressure, depth=depth)
            # print("INITIALIZE compute loop at ", time_cnt)
            initialize_cntr = time_cnt2
            mem_model_b.valid_tile_received()
            mem_model_c.valid_tile_received()


        if mem_blocks_decl_flag and fiberlookup_Bi_19.out_done() and mem_model_bi.valid_tile() and not nxt_tile_present[0]:
            B_crd0 = dict_arr_B0[mem_model_bi.token()]
            B_seg0 = dict_seg_arr_B0[mem_model_bi.token()]
            B_crd1 = dict_arr_B1[mem_model_bi.token()]
            B_seg1 = dict_seg_arr_B1[mem_model_bi.token()]
            B_vals = dict_vals_B[mem_model_bi.token()]
            #print("             ________ Bi DONE  ______________ ", mem_model_bi.token())
            #print(B_crd0, B_seg0)
            in_fifo = fiberlookup_Bi_19.get_fifo()
            in_fifo.append(0)
            in_fifo.append("D")
            #print(in_fifo)
            fiberlookup_Bi_19 = CompressedCrdRdScan(name="Bi", crd_arr=B_crd0, seg_arr=B_seg0,
                                                    debug=debug_sim2, statistics=report_stats, fifo=in_fifo,
                                                    back_en=backpressure, depth=depth)
            mem_model_bi.valid_tile_received()
            nxt_tile_present[0] = True
        if mem_blocks_decl_flag and fiberlookup_Bk_14.out_done() and mem_model_bk.valid_tile() and not nxt_tile_present[1]:
            B_crd0 = dict_arr_B0[mem_model_bk.token()]
            B_seg0 = dict_seg_arr_B0[mem_model_bk.token()]
            B_crd1 = dict_arr_B1[mem_model_bk.token()]
            B_seg1 = dict_seg_arr_B1[mem_model_bk.token()]
            B_vals = dict_vals_B[mem_model_bk.token()]
            in_fifo = fiberlookup_Bk_14.get_fifo()
            
            #print("             ________ Bk DONE  ______________")
            #print(B_crd1, B_seg1)
            
            fiberlookup_Bk_14 = CompressedCrdRdScan(name="Bk", crd_arr=B_crd1, seg_arr=B_seg1,
                                                    debug=debug_sim2, statistics=report_stats, fifo=in_fifo,
                                                    back_en=backpressure, depth=depth)
            mem_model_bk.valid_tile_received()
            nxt_tile_present[1] = True
        if mem_blocks_decl_flag and fiberlookup_Ck_15.out_done() and mem_model_ck.valid_tile() and not nxt_tile_present[2]:

            C_crd0 = dict_arr_C0[mem_model_ck.token()]
            C_seg0 = dict_seg_arr_C0[mem_model_ck.token()]
            C_crd1 = dict_arr_C1[mem_model_ck.token()]
            C_seg1 = dict_seg_arr_C1[mem_model_ck.token()]
            C_vals = dict_vals_C[mem_model_ck.token()]
            
            in_fifo = fiberlookup_Ck_15.get_fifo()

            #print("             ________ Ck DONE  ______________")
            #print(C_crd0, C_seg0)
            
            fiberlookup_Ck_15 = CompressedCrdRdScan(name="Ck", crd_arr=C_crd0, seg_arr=C_seg0,
                                                    debug=debug_sim2, statistics=report_stats, fifo=in_fifo,
                                                    back_en=backpressure, depth=depth)
            mem_model_ck.valid_tile_received()
            repeat_Ci_16.set_in_ref(0, "")
            repeat_Ci_16.set_in_ref("D", "")
            nxt_tile_present[2] = True 
        
        if mem_blocks_decl_flag and fiberlookup_Cj_12.out_done() and mem_model_cj.valid_tile() and not nxt_tile_present[3]:
            
            C_crd0 = dict_arr_C0[mem_model_cj.token()]
            C_seg0 = dict_seg_arr_C0[mem_model_cj.token()]
            C_crd1 = dict_arr_C1[mem_model_cj.token()]
            C_seg1 = dict_seg_arr_C1[mem_model_cj.token()]
            C_vals = dict_vals_C[mem_model_cj.token()] 
            
            #print("             ________ Cj DONE  ______________")
            #print(C_crd1, C_seg1)
            in_fifo = fiberlookup_Cj_12.get_fifo()
            fiberlookup_Cj_12 = CompressedCrdRdScan(name="Cj", crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim2,
                                                    statistics=report_stats, fifo=in_fifo,
                                                    back_en=backpressure, depth=depth)
            mem_model_cj.valid_tile_received()
            nxt_tile_present[3] = True
        if mem_blocks_decl_flag and arrayvals_B_7.out_done() and mem_model_bvals.valid_tile() and not nxt_tile_present[4]:
            
            B_crd0 = dict_arr_B0[mem_model_bvals.token()]
            B_seg0 = dict_seg_arr_B0[mem_model_bvals.token()]
            B_crd1 = dict_arr_B1[mem_model_bvals.token()]
            B_seg1 = dict_seg_arr_B1[mem_model_bvals.token()]
            B_vals = dict_vals_B[mem_model_bvals.token()]

            #print("             ________ Bvals DONE  ______________")
            in_fifo = arrayvals_B_7.get_fifo()
            arrayvals_B_7 = Array(name="Bvals", init_arr=B_vals, debug=debug_sim2, statistics=report_stats, fifo=in_fifo,
                                  back_en=backpressure, depth=depth)
            mem_model_bvals.valid_tile_received()
            nxt_tile_present[4] = True
        if mem_blocks_decl_flag and arrayvals_C_8.out_done() and mem_model_cvals.valid_tile() and not nxt_tile_present[5]:
            
            C_crd0 = dict_arr_C0[mem_model_cvals.token()]
            C_seg0 = dict_seg_arr_C0[mem_model_cvals.token()]
            C_crd1 = dict_arr_C1[mem_model_cvals.token()]
            C_seg1 = dict_seg_arr_C1[mem_model_cvals.token()]
            C_vals = dict_vals_C[mem_model_cvals.token()]
            
            in_fifo = arrayvals_C_8.get_fifo()
            arrayvals_C_8 = Array(name="Cvals", init_arr=C_vals, debug=debug_sim2, statistics=report_stats, fifo=in_fifo,
                                  back_en=backpressure, depth=depth)
            mem_model_cvals.valid_tile_received()
            nxt_tile_present[5] = True

        if mem_blocks_decl_flag and fiberlookup_Bi_19.out_done() and nxt_tile_present[0]:
            mem_model_bi.check_if_done(True)
            nxt_tile_present[0] = False
        else:
            mem_model_bi.check_if_done(False)
        if mem_blocks_decl_flag and fiberlookup_Bk_14.out_done() and nxt_tile_present[1]:
            mem_model_bk.check_if_done(True)
            nxt_tile_present[1] = False
        else:
            mem_model_bk.check_if_done(False)
        if mem_blocks_decl_flag and fiberlookup_Cj_12.out_done() and nxt_tile_present[3]:
            mem_model_cj.check_if_done(True)
            nxt_tile_present[3] = False
        else:
            mem_model_cj.check_if_done(False)
        if mem_blocks_decl_flag and fiberlookup_Ck_15.out_done() and nxt_tile_present[2]:
            mem_model_ck.check_if_done(True)
            nxt_tile_present[2] = False
        else:
            mem_model_ck.check_if_done(False)
        if mem_blocks_decl_flag and arrayvals_C_8.out_done() and nxt_tile_present[5]:
            mem_model_cvals.check_if_done(True)
            nxt_tile_present[5] = False
        else:
            mem_model_cvals.check_if_done(False)
        if mem_blocks_decl_flag and arrayvals_B_7.out_done() and nxt_tile_present[4]:
            mem_model_bvals.check_if_done(True)
            nxt_tile_present[4] = False
        else:
            mem_model_bvals.check_if_done(False)


        if flag:
            if len(in_ref_B_) > 0:
                fiberlookup_Bi_19.set_in_ref(in_ref_B_.pop(0), "")
            fiberlookup_Bk_14.set_in_ref(fiberlookup_Bi_19.out_ref(), fiberlookup_Bi_19)
            repsiggen_i_17.set_istream(fiberlookup_Bi_19.out_crd(), fiberlookup_Bi_19)
            if len(in_ref_C_) > 0:
                repeat_Ci_16.set_in_ref(in_ref_C_.pop(0), "")
            repeat_Ci_16.set_in_repsig(repsiggen_i_17.out_repsig(), repsiggen_i_17)
            fiberlookup_Ck_15.set_in_ref(repeat_Ci_16.out_ref(), repeat_Ci_16)
            intersectk_13.set_in1(fiberlookup_Ck_15.out_ref(), fiberlookup_Ck_15.out_crd(), fiberlookup_Ck_15)
            intersectk_13.set_in2(fiberlookup_Bk_14.out_ref(), fiberlookup_Bk_14.out_crd(), fiberlookup_Bk_14)
            crdhold_5.set_outer_crd(fiberlookup_Bi_19.out_crd(), fiberlookup_Bi_19)
            crdhold_5.set_inner_crd(intersectk_13.out_crd(), intersectk_13)
            fiberlookup_Cj_12.set_in_ref(intersectk_13.out_ref1(), intersectk_13)
            arrayvals_C_8.set_load(fiberlookup_Cj_12.out_ref(), fiberlookup_Cj_12)
            crdhold_4.set_outer_crd(crdhold_5.out_crd_outer(), crdhold_5)
            crdhold_4.set_inner_crd(fiberlookup_Cj_12.out_crd(), fiberlookup_Cj_12)
            repsiggen_j_10.set_istream(fiberlookup_Cj_12.out_crd(), fiberlookup_Cj_12)
            repeat_Bj_9.set_in_ref(intersectk_13.out_ref2(), intersectk_13)
            repeat_Bj_9.set_in_repsig(repsiggen_j_10.out_repsig(), repsiggen_j_10)
            arrayvals_B_7.set_load(repeat_Bj_9.out_ref(), repeat_Bj_9)
            mul_6.set_in1(arrayvals_B_7.out_val(), arrayvals_B_7)
            mul_6.set_in2(arrayvals_C_8.out_val(), arrayvals_C_8)
            spaccumulator1_3_drop_crd_outer.set_in_stream(crdhold_4.out_crd_outer(), crdhold_4)
            spaccumulator1_3_drop_crd_inner.set_in_stream(crdhold_4.out_crd_inner(), crdhold_4)
            spaccumulator1_3_drop_val.set_in_stream(mul_6.out_val(), mul_6)
            spaccumulator1_3.set_crd_outer(spaccumulator1_3_drop_crd_outer.out_val(), spaccumulator1_3_drop_crd_outer)
            spaccumulator1_3.set_crd_inner(spaccumulator1_3_drop_crd_inner.out_val(), spaccumulator1_3_drop_crd_inner)
            spaccumulator1_3.set_val(spaccumulator1_3_drop_val.out_val(), spaccumulator1_3_drop_val)
            # print("Write: ", spaccumulator1_3.out_val(), spaccumulator1_3.out_crd_inner(), spaccumulator1_3.out_crd_outer())
            fiberwrite_Xvals_0.set_input(spaccumulator1_3.out_val(), spaccumulator1_3)
            fiberwrite_X1_1.set_input(spaccumulator1_3.out_crd_inner(), spaccumulator1_3)
            fiberwrite_X0_2.set_input(spaccumulator1_3.out_crd_outer(), spaccumulator1_3)
            if debug_sim2:
                print("TILE NAME: ", mem_model_b.token(), mem_model_c.token())
                print("____________________________________", time_cnt2, tiled_done, tile_signalled)
            # If tile computed on move ahead
            if tiled_done:  # and not tile_signalled:
                mem_model_b.check_if_done(tiled_done)
                mem_model_c.check_if_done(tiled_done)
            else:
                mem_model_b.check_if_done(False)
                mem_model_c.check_if_done(False)

        fiberlookup_Bi00.update()
        fiberlookup_Bk00.update()
        repsiggen_i00.update()
        repeat_Ci00.update()
        fiberlookup_Ck00.update()
        intersect_00.update()
        fiberlookup_Cj00.update()
        repsiggen_j00.update()
        repeat_Bj00.update()
        mem_model_b.update(time_cnt2)
        mem_model_c.update(time_cnt2)
        mem_model_bk.update(time_cnt2)
        mem_model_ck.update(time_cnt2)
        mem_model_bvals.update(time_cnt2)
        mem_model_cvals.update(time_cnt2)
        mem_model_bi.update(time_cnt2)
        mem_model_cj.update(time_cnt2)


        tiled_done = False
        if flag:
            fiberlookup_Bi_19.update()
            fiberlookup_Bk_14.update()
            repsiggen_i_17.update()
            repeat_Ci_16.update()
            fiberlookup_Ck_15.update()
            intersectk_13.update()
            crdhold_5.update()
            fiberlookup_Cj_12.update()
            arrayvals_C_8.update()
            crdhold_4.update()
            repsiggen_j_10.update()
            repeat_Bj_9.update()
            arrayvals_B_7.update()
            mul_6.update()
            spaccumulator1_3_drop_crd_outer.update()
            spaccumulator1_3_drop_crd_inner.update()
            spaccumulator1_3_drop_val.update()
            spaccumulator1_3.update()
            fiberwrite_Xvals_0.update()
            fiberwrite_X1_1.update()
            fiberwrite_X0_2.update()
            tiled_done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()
            if tiled_done:
                # print("working on tile pair: ", mem_model_b.token(), mem_model_c.token())
                if mem_model_c.out_done():
                    if mem_model_b.out_done():
                        done = True
        
        time_cnt2 += 1


    def bench():
        time.sleep(0.01)
    #print("VAL", max_cnt, average / average_len)
    if debug_sim:
        print(out_crd_i_out)
        #print(temp_arr)
        #print("\\")
        print(out_crd_k_out)
        print(out_crd_i)
        print(out_crd)
        print("-")
        print(max_cnt)
        print(average / average_len)

    print("Done and time: ", done, " ", time_cnt, " ", time_cnt2, " ", time_cnt + time_cnt2)

    extra_info = dict()
    extra_info["dataset"] = ssname
    extra_info["nnz"] = nnz
    extra_info["cycles"] = time_cnt + time_cnt2
    extra_info["cycles_tiling"] = time_cnt
    extra_info["cycles_matmul"] = time_cnt2
    extra_info["max_tile_size"] = max_cnt
    extra_info["avg_tile_size"] = average / average_len
    extra_info["total_tile_cnts"] = average_len
    extra_info["tensor_B_shape"] = B_shape
    extra_info["tensor_B/nnz"] = len(B_vals)

    sample_dict = crdscan.return_statistics()
    for k in sample_dict.keys():
        extra_info["reorder_block" + "/" + k] = sample_dict[k]

    samBench(bench, extra_info)
