import copy
import pytest

from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.reorder import Reorder_and_split, repeated_token_dopper
from sam.sim.src.base import remove_emptystr
from sam.onyx.generate_matrices import *
import os
import csv

cwd = os.getcwd()
######################################
# Compressed Read Scanner Unit Tests #
######################################

arr_dict = {"seg": [0, 3, 6, 9, 11], "crd": [1, 3, 4, 0, 1, 2, 0, 4, 5, 3, 5], "in_ref": [0, 1, 2, 3, 'S0', 'D'], "in_crd": [0, 1, 4, 6, 'S0', 'D'],
            "out_crd_k": [1, 3, 'S0', 0, 1, 2, 'S0', 0, 'S0', 3, 'S1', 0, 'S0', 0, 1, 'S0', 1, 'S2', 'D'],
            "out_ref_k": [0, 1, 'S0', 3, 4, 5, 'S0', 6, 'S0', 9, 'S1', 2, 'S0', 7, 8, 'S0', 10, 'S2', 'D'],
            "out_crd_i": [0, 1, 4, 6, 'S0', 0, 4, 6, 'S1', 'D'],
            "out_ref_i": [0, 1, 2, 3, 'S0', 0, 2, 3, 'S1', 'D']}


arr_dict1 = {"seg": [0, 2, 3, 4], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S0', 'D'], "in_crd": [0, 2, 3, 'S0', 'D'],
             "out_crd_k": [0, 'S1', 2, 'S0', 2, 'S0',  2,  'S1', 'D'], "out_ref_k": [0, 'S0', 1, 'S0', 2, 'S0', 3, 'S1', 'D'],
             "out_crd_i": [0, 'S0',  0, 2, 3, 'S1', 'D'], "out_ref_i": [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D']}

arr_dict2 = {"seg": [0, 2, 5,  7, 9, 10, 11, 15], "crd": [0, 2, 2, 2], "in_ref": [0, 1, 2, 'S0', 'D'], "in_crd": [0, 2, 5,  7, 9, 10, 11, 15, 'S0', 'D'],
             "out_crd_k": [0, 'S1', 2, 'S0', 2, 'S0',  2,  'S1', 'D'], "out_ref_k": [0, 'S0', 1, 'S0', 2, 'S0', 3, 'S1', 'D'],
             "out_crd_i": [0, 'S0',  0, 2, 3, 'S1', 'D'], "out_ref_i": [0, 1, 'S0', 2, 'S0', 3, 'S1', 'D']}



arr_dict2 = {"seg": [0, 3, 4, 6], "crd": [0, 2, 3, 0, 2, 3], "in_ref": [0, 0, 'S0', 1, 'S0', 2, 'S1', 'D'],
             "out_crd": [0, 2, 3, 'S0', 0, 2, 3, 'S1', 0, 'S1', 2, 3, 'S2', 'D'],
             "out_ref": [0, 1, 2, 'S0', 0, 1, 2, 'S1', 3, 'S1', 4, 5, 'S2', 'D']}
arr_dict3 = {"seg": [0, 4], "crd": [0, 1, 2, 3], "in_ref": [0, 'D'],
             "out_crd": [0, 1, 2, 3, 'S0', 'D'], "out_ref": [0, 1, 2, 3, 'S0', 'D']}

formatted_dir = os.getenv('SYNTHETIC_TEST_PATH',  default=os.path.join(cwd, 'synthetic'))


def create_array(shape=5, sparsity=0.995, path=""):
   
    os.system("python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity " + str(sparsity) + " --output_dir ${SYNTHETIC_PATH}/matrix/DCSR_transpose/ --name B --shape " + str(shape) + " " + str(shape) + " --output_format CSF --transpose")
    os.system("python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity " + str(sparsity) + " --output_dir ${SYNTHETIC_PATH}/matrix/DCSR/ --name B --shape " + str(shape) + " " + str(shape) + " --output_format CSF")
    arr_dict = {}

    B_dirname = os.path.join(formatted_dir, "matrix/DCSR/B_random_sp_" + str(sparsity) + "/")

    B0_seg_filename = os.path.join(B_dirname, "tensor_B_mode_0_seg")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "tensor_B_mode_0_crd")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B_crd1 = read_inputs(B1_crd_filename)

    C_dirname = os.path.join(formatted_dir, "matrix/DCSR_transpose/B_random_sp_" + str(sparsity) + "/")

    C0_seg_filename = os.path.join(C_dirname, "tensor_B_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_B_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_B_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_B_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename)



    rdB_0 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0)
    rdB_1 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1)

    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time_cnt1 = 0
    arr_dict["in_ref"] =  [] # read level 0 and put in rd scanner
    arr_dict["in_crd"] =  [] # read level 0 and put in rd scanner 
    t1 = []
    t2 = []
    while not done and time_cnt1 < TIMEOUT:
        if len(in_ref_B) > 0:
            rdB_0.set_in_ref(in_ref_B.pop(0))
        rdB_1.set_in_ref(rdB_0.out_ref())
        
        rdB_0.update()
        rdB_1.update()

        if rdB_0.out_crd() != "":
            arr_dict["in_crd"].append(rdB_0.out_crd())
        if rdB_0.out_ref() != "":
            arr_dict["in_ref"].append(rdB_0.out_ref())
        if rdB_1.out_crd() != "":
            t1.append(rdB_1.out_crd())
        if rdB_1.out_ref() != "":
            t2.append(rdB_1.out_ref())
 
        done = rdB_1.out_done()
        time_cnt1 += 1


    
    rdC_0 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0)
    rdC_1 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1)


    arr_dict["seg"] = B_seg1 # read level 0 and put in rd scanner
    arr_dict["crd"] = B_crd1 # read level 0 and put in rd scanner 


    arr_dict["out_crd_k_outer"] = []  # reading transpose
    arr_dict["out_ref_k_outer"] = []
    arr_dict["out_crd_i"] = []
    arr_dict["out_ref_i"] = []

    time_cnt2 = 0
    done = False
    while not done and time_cnt2 < TIMEOUT:
        if len(in_ref_C) > 0:
            rdC_0.set_in_ref(in_ref_C.pop(0))
        rdC_1.set_in_ref(rdC_0.out_ref())
        
        rdC_0.update()
        rdC_1.update()
        # print(rdC_0.out_crd(), rdC_0.out_ref(), rdC_1.out_crd(), rdC_1.out_ref())
        if rdC_0.out_crd() != "":
            arr_dict["out_crd_k_outer"].append(rdC_0.out_crd())
        if rdC_0.out_ref() != "":
            arr_dict["out_ref_k_outer"].append(rdC_0.out_ref())
        if rdC_1.out_crd() != "":
            arr_dict["out_crd_i"].append(rdC_1.out_crd())
        if rdC_1.out_ref() != "":
            arr_dict["out_ref_i"].append(rdC_1.out_ref())

        done = rdC_1.out_done()
        time_cnt2 += 1
    # print(arr_dict)
    # print(t1, t2)
    return arr_dict, time_cnt1, time_cnt2

 


@pytest.mark.synth
def test_reorder_direct_transpose(arrs, debug_sim):
    shape = [1000] #, 2000]
    sparsity = [0.75, 0.9, 0.99, 0.995]
    plot_arr = []
    plot_arr2 = []
    for sp in sparsity:
        for s in shape:
            # print("____________________________________", s, sp)
            arrs, read_num1, read_num2 = create_array(shape=s, sparsity=sp)
            # print(arrs)
            # return
            seg_arr = arrs["seg"]
            crd_arr = arrs["crd"]

            # gold_crd = arrs["out_crd_k"]
            # gold_ref = arrs["out_ref_k"]
            gold_crd_i = arrs["out_crd_i"]
            gold_ref_i = arrs["out_ref_i"]
            # debug_sim = True
            assert (len(gold_crd_i) == len(gold_ref_i))
            crdscan = Reorder_and_split(seg_arr=seg_arr, crd_arr=crd_arr, limit=10, sf=1, debug=debug_sim)
            
            crd_k = repeated_token_dopper(name="crdk")
            ref_k = repeated_token_dopper(name="refk")
            crd_i = repeated_token_dopper(name="crdi")
            ref_i = repeated_token_dopper(name="refi")
            
            crd_k_out = repeated_token_dopper(name="crdkout")
            ref_k_out = repeated_token_dopper(name="refkout")


            
            in_ref = copy.deepcopy(arrs["in_ref"])
            in_crd = copy.deepcopy(arrs["in_crd"])
            done = False
            time = 0
            out_crd = []
            out_ref = []
            out_crd_i = []
            out_ref_i = []
            out_crd_k_out = []
            out_ref_k_out = []
            while not done and time < TIMEOUT:
                if len(in_ref) > 0:
                    crdscan.set_input(in_ref.pop(0), in_crd.pop(0))
                crd_k.add_token(crdscan.out_crd_k())
                ref_k.add_token(crdscan.out_ref_k())
                crd_i.add_token(crdscan.out_crd_i())
                ref_i.add_token(crdscan.out_ref_i())
                crd_k_out.add_token(crdscan.out_crd_k_outer())
                ref_k_out.add_token(crdscan.out_ref_k_outer())
         

                crdscan.update()
                crd_k.update()
                ref_k.update()
                crd_i.update()
                ref_i.update()
                crd_k_out.update()
                ref_k_out.update()

                if crd_k.get_token() != "":
                    out_crd.append(crd_k.get_token())
                    out_ref.append(ref_k.get_token())
                if crd_i.get_token() != "":
                    out_crd_i.append(crd_i.get_token())
                    out_ref_i.append(ref_i.get_token())
                if crd_k_out.get_token() != "":
                    out_crd_k_out.append(crd_k_out.get_token())
                    out_ref_k_out.append(ref_k_out.get_token())
         

                # print("Timestep", time, "\t k_out_crd:", crdscan.out_crd_k_outer(), "\t k_out_ref:", crdscan.out_ref_k_outer(), "\t Crd:", crdscan.out_crd_i(), "\t Ref:", crdscan.out_ref_i(), "\t Crd:", crdscan.out_crd_k(), "\t Ref:", crdscan.out_ref_k())
                # print("______________________________________________________________________")
                
                done = crd_k.done and ref_k.done and crd_i.done and ref_i.done and crd_k_out.done and ref_k_out.done
                time += 1
                if time > 1000000000:
                    break
            
            print("Done and time: ", done, time, " read time: ", read_num1, read_num2)
            
            plot_arr.append(time)

            plot_arr2.append(read_num2)
            # print("Out Crd val (k): ", out_crd)
            ## print("Gold Crd val (k): ", gold_crd)
            # print(out_ref)
            ## print(gold_ref)

            # print("Out Crd Val (i) ", out_crd_i)
            # print("Gold Crd Val (i)", gold_crd_i)
            # print(out_crd_i)
            # print(gold_crd_i)
            # print(gold_ref_i)

            # print("outer crd: ", out_crd_k_out)
            # print(out_ref_k_out)



            assert (out_crd_i == gold_crd_i)
            assert (out_crd_k_out == arrs["out_crd_k_outer"])
    print(plot_arr)
    print(plot_arr2)
