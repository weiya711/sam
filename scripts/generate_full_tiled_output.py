from scipy.sparse import csr_matrix
import scipy.sparse as sparse
import os
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix
import argparse
import yaml
import pickle 
import numpy as np

parser = argparse.ArgumentParser("Generate output")
parser.add_argument("--yaml_name", type=str, default="memory_config_onyx.yaml")
parser.add_argument("--op_name", type=str, default="matmul_ijk")
args = parser.parse_args()
yaml_name = args.yaml_name
kernel = args.op_name

cwd = os.getcwd()
directory =  os.getenv('TILED_SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd))
sam_home = os.getenv('SAM_HOME', default=os.path.join(cwd))
output_file = sam_home + "/tiles/" + kernel + "/temporary_output/"
print("sam home: ", sam_home)
#  open(os.path.join(sam_home, "tiles/matmul_ijk/tensor_sizes"), "rb"
with open(os.path.join(sam_home, "tiles/matmul_ijk/tensor_sizes"), "rb") as ff:
    sizes_dict_level_full = pickle.load(ff)
with open(os.path.join(sam_home, "sam/sim/src/tiling/" + yaml_name), "r") as stream:
    memory_config = yaml.safe_load(stream)

# Initialize empty COO matrix for output
output_matrix = coo_matrix((sizes_dict_level_full["B"][0], sizes_dict_level_full["C"][1]))
row_offset1 = memory_config["Glb_tile_size"]
row_offset2 = memory_config["Mem_tile_size"]
col_offset1 = memory_config["Glb_tile_size"]
col_offset2 = memory_config["Mem_tile_size"]


for i00 in range(1 + int(sizes_dict_level_full["B"][0]) // (memory_config["Glb_tile_size"] * memory_config["Mem_tile_size"])):
    for j00 in range(1 + int(sizes_dict_level_full["B"][1]) // (memory_config["Glb_tile_size"] * memory_config["Mem_tile_size"])):
        for k00 in range(1 + int(sizes_dict_level_full["C"][1]) // (memory_config["Glb_tile_size"] * memory_config["Mem_tile_size"])):
            for i0 in range(row_offset1):
                for j0 in range(row_offset1):
                    for k0 in range(row_offset1):
                        filename = "coo_matrix"
                        filepath = filename + "_" + str(i00) + "_" + str(k00) + "_"\
                                + str(j00) +  "_" + str(i0) + "_" + str(k0) + "_" + str(j0) +\
                                ".mtx"
                        print(filepath)
                        print(output_file)
                        print(os.path.exists(os.path.join(output_file, filepath)))
                        if os.path.exists(os.path.join(output_file, filepath)):
                            row_offset = (i00 * row_offset1 + i0) * row_offset2
                            col_offset = (j00 * col_offset1 + j0) * col_offset2
                            matrix = mmread(os.path.join(output_file, filepath))
                            dense_matrix = matrix.todense()
                            #print("init dese shape ", dense_matrix.shape)
                            dense_matrix = np.pad(dense_matrix, ((0, sizes_dict_level_full["B"][0]-dense_matrix.shape[0]),
                                                  (0, sizes_dict_level_full["C"][1]-dense_matrix.shape[1])), mode="constant")
                            #print("init dese shape ", dense_matrix.shape, sizes_dict_level_full["B"][0])
                            reshaped = dense_matrix.reshape(sizes_dict_level_full["B"][0], sizes_dict_level_full["C"][1])
                            reshaped = csr_matrix(reshaped)
                            #print(reshaped)
                            matrix = reshaped.tocoo()
                            # coo_matrix((reshaped.data, (reshaped.row, reshaped.col)), shape=(row_offset2, row_offset2))
                            #matrix.reshape(row_offset2, row_offset2)
                            matrix.row += row_offset
                            matrix.col += col_offset
                            output_matrix = output_matrix + matrix

orig_matrix = mmread(sam_home + "/tiles/extensor.mtx")
new_data = [2]*orig_matrix.nnz
coo_mtx2 = sparse.coo_matrix((new_data, (orig_matrix.row, (orig_matrix.col + 1) % (orig_matrix.shape[1]))), shape=orig_matrix.shape)
coo_mtx_T = coo_mtx2.transpose(copy=True).tocoo()
# Multiply the matrices
result = orig_matrix.dot(coo_mtx_T)
result = result.tocsr()
output_matrix = output_matrix.tocsr()

A = result.sort_indices()
B = output_matrix.sort_indices()
# sort the row, column, and data arrays of each matrix

assert (result.data == output_matrix.data).all() and (result.indices == output_matrix.indices).all() and (result.indptr == output_matrix.indptr).all()
# Write output matrix to MTX file
mmwrite(os.path.join(sam_home + "/tiles/" + kernel + "/", "output.mtx"), output_matrix)
