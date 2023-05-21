import scipy
import os
import copy
import csv
import torch
from sam.onyx.generate_matrices import *
import time
# import torch_geometric.transforms as T
# from torch_geometric.datasets import TUDataset


def sparse_tranpose_scipy(ssdir, ssname, debug_sim, out_format="ss10"):
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    B_scipy = B_tensor
    C_scipy = B_scipy.transpose()
    out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
    out_tup = remove_zeros(out_tup)
    return


def sparse_tranpose_pytorch(dir_name, debug_sim, shots=5, shape=40, out_format="ss10"):
    software_time_file_loads = 0
    for atts in range(shots):
        start_time = time.time()
        B_dirname = dir_name
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
        software_time_file_loads += time.time() - start_time
    software_time_file_loads = software_time_file_loads / shots

    out_crds = [B_crd0, B_crd1]
    out_segs = [B_seg0, B_seg1]
    out_val = B_vals

    pt_tuple_time = 0
    for atts in range(shots):
        start_time = time.time()
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs))
        # , out_val))
        pt_tuple_time += time.time() - start_time
    pt_tuple_time = pt_tuple_time / shots

    software_time_file_format_convert1 = 0
    for atts in range(shots):
        start_time = time.time()
        tensor = torch.sparse_coo_tensor(list(zip(*out_tup)), out_val, size=(shape, shape))
        software_time_file_format_convert1 += time.time() - start_time
    software_time_file_format_convert1 = software_time_file_format_convert1 / shots
    software_time_tranpose = 0
    for atts in range(shots):
        start_time = time.time()
        a = torch.transpose(tensor, 0, 1)
        software_time_tranpose += time.time() - start_time
    software_time_tranpose = software_time_tranpose / shots
    software_time_file_format_convert2 = 0
    for atts in range(shots):
        start_time = time.time()

        a = a.to_sparse_csr()

        software_time_file_format_convert2 += time.time() - start_time
    software_time_file_format_convert2 = software_time_file_format_convert2 / shots
    return a, [software_time_file_loads, pt_tuple_time,
               software_time_file_format_convert1, software_time_tranpose, software_time_file_format_convert2]


def sparse_tranpose_geometrics():
    return
