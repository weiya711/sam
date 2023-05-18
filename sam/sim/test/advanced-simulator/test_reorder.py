import copy
import pytest

from sam.sim.test.test import TIMEOUT
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.reorder import ReorderAndSplit, RepeatedTokenDropper
from sam.sim.src.base import remove_emptystr
from sam.sim.test.sparse_transpose import *
from sam.onyx.generate_matrices import *
import os
import csv
import time
cwd = os.getcwd()
######################################
# Compressed Read Scanner Unit Tests #
######################################


formatted_dir = os.getenv('SYNTHETIC_PATH',  default=os.path.join(cwd, 'synthetic'))


def create_array(shape=5, sparsity=0.995, path=""):
   
    os.system("python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity " +
              str(sparsity) + " --output_dir ${SYNTHETIC_PATH}/matrix/DCSC/ --name B --shape " +
              str(shape) + " " + str(shape) + " --output_format CSF --transpose")
    os.system("python ${SRC_PATH}/generate_random_mats.py --seed 0 --sparsity " + str(sparsity) +
              " --output_dir ${SYNTHETIC_PATH}/matrix/DCSR/ --name B --shape " + str(shape) +
              " " + str(shape) + " --output_format CSF")
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
    C_dirname = os.path.join(formatted_dir, "matrix/DCSC/B_random_sp_" + str(sparsity) + "/")
    C0_seg_filename = os.path.join(C_dirname, "tensor_B_mode_0_seg")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "tensor_B_mode_0_crd")
    C_crd0 = read_inputs(C0_crd_filename)
    C1_seg_filename = os.path.join(C_dirname, "tensor_B_mode_1_seg")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_B_mode_1_crd")
    C_crd1 = read_inputs(C1_crd_filename)
    C_vals_filename = os.path.join(C_dirname, "tensor_B_mode_vals")
    C_vals = read_inputs(C_vals_filename)

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
    software_time = 0
    for atts in range(5):
        start_time = time.time()
        transpose_mat, time_arr = sparse_tranpose_pytorch(B_dirname, shape=shape, shots=5, debug_sim=False, out_format="ss10")
        software_time += time.time() - start_time
    software_time = software_time / (5 * 5)
    return arr_dict, time_cnt1, time_cnt2, software_time, time_arr

#FIXME: Figureout formats
@pytest.mark.skipif(
        os.getenv('CI', 'false') == 'true',
        reason='CI lacks datasets',
)
@pytest.mark.synth
@pytest.mark.parametrize("sparsity", [0.5])
def test_reorder_direct_transpose(debug_sim, sparsity, reorder_not_ideal, reorder_block_len):
    #print("BLK_LEN:", reorder_block_len)
    #print(reorder_not_ideal, bool(reorder_not_ideal) == True)
    shape = [128]
    sparsity = [0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.99, 0.995, 0.9975]
    plot_arr = []
    plot_arr2 = []
    software_times = []
    
    load_times = []
    pt_tuple_times = []
    format_conv_time1 = []
    transpose_time = []
    format_conv_time2 = []
    for sp in sparsity:
        for s in shape:
            arrs, read_num1, read_num2, software_time, received_time_arr = create_array(shape=s, sparsity=sp)
            seg_arr = arrs["seg"]
            crd_arr = arrs["crd"]

            gold_crd_i = arrs["out_crd_i"]
            gold_ref_i = arrs["out_ref_i"]
            assert (len(gold_crd_i) == len(gold_ref_i))
            crdscan = ReorderAndSplit(seg_arr=seg_arr, crd_arr=crd_arr, not_idealized=bool(reorder_not_ideal),
                                        block_size_len=int(reorder_block_len), sf=1, debug=debug_sim, statistics=True)
            
            crd_k = RepeatedTokenDropper(name="crdk")
            ref_k = RepeatedTokenDropper(name="refk")
            crd_i = RepeatedTokenDropper(name="crdi")
            ref_i = RepeatedTokenDropper(name="refi")
            
            crd_k_out = RepeatedTokenDropper(name="crdkout")
            ref_k_out = RepeatedTokenDropper(name="refkout")

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
                if ref_k.get_token() != "":
                    out_ref.append(ref_k.get_token())
                if crd_i.get_token() != "":
                    out_crd_i.append(crd_i.get_token())
                if ref_i.get_token() != "":
                    out_ref_i.append(ref_i.get_token())
                if crd_k_out.get_token() != "":
                    out_crd_k_out.append(crd_k_out.get_token())
                if ref_k_out.get_token() != "":
                    out_ref_k_out.append(ref_k_out.get_token())
         
                done = crd_k.done and ref_k.done and crd_i.done and ref_i.done and crd_k_out.done and ref_k_out.done
                time += 1
                if time > 1000000000:
                    break
            
            plot_arr.append(time)
            plot_arr2.append(read_num2)
            software_times.append(software_time)
            load_times.append(received_time_arr[0])
            pt_tuple_times.append(received_time_arr[1]) 
            format_conv_time1.append(received_time_arr[2])
            transpose_time.append(received_time_arr[3])
            format_conv_time2.append(received_time_arr[4])
            assert (out_crd_i == gold_crd_i)
            assert (out_crd_k_out == arrs["out_crd_k_outer"])
            assert (out_ref_k_out == arrs["out_ref_k_outer"])

    print(plot_arr)
    print(plot_arr2)
    print(software_times)
    print("load time: ", load_times)
    print("conv1 time: ", pt_tuple_times)
    print("conv1 time: ", format_conv_time1)
    print("tran time: ", transpose_time)
    print("conv2 time: ", format_conv_time2)
