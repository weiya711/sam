import scipy
import scipy.sparse
import os
import scipy.io
import numpy as np


def generate_gold_matmul_tiled(tile_crd_b, tile_crd_c, out_format="ss01"):
    # CSR
    formatted_dir = "/nobackup/rsharma3/Sparsity/simulator/old_sam/sam/tiles/matmul_ikj/mtx"
    B_dir = "tensor_B_tile_"
    for a in tile_crd_b:
        B_dir += str(a) + "_"
    C_dir = "tensor_C_tile_"
    for a in tile_crd_c:
        C_dir += str(a) + "_"
    B_dir = B_dir[0:-1] + ".mtx"
    C_dir = C_dir[0:-1] + ".mtx"
    print(B_dir, " ", C_dir)
    B_filename = os.path.join(formatted_dir, B_dir)
    C_filename = os.path.join(formatted_dir, C_dir)
    print(B_filename)
    print(C_filename)
    if os.path.exists(B_filename) and os.path.exists(C_filename):
        B_scipy = scipy.io.mmread(B_filename)
        itr = 0
        for i, j, v in zip(B_scipy.row, B_scipy.col, B_scipy.data):
            # print(B_scipy.data)
            # print(i, " ", j, " ", v)
            if B_scipy.data[i] < 1:
                B_scipy.data[i] = 1
            else:
                B_scipy.data[i] = int(B_scipy.data[i])
            itr += 1
        B_scipy = B_scipy.tocsr()
        C_scipy = scipy.io.mmread(C_filename)
        for i, j, v in zip(C_scipy.row, C_scipy.col, C_scipy.data):
            if C_scipy.data[i] < 1:
                C_scipy.data[i] = 1
            else:
                C_scipy.data[i] = int(C_scipy.data[i])
            itr += 1
        C_scipy = C_scipy.tocsc()
        gold_nd = (B_scipy @ C_scipy)
        gold_out = gold_nd.tocoo()
        scipy.io.mmwrite("out" + tile_crd_b[0] + "_" + tile_crd_c[1] + "_" + tile_crd_b[2] + "_" + tile_crd_c[3] + ".mtx", gold_out)

if __name__ == "__main__": 
    check_gold_matmul_tiled([0, 0, 0, 0], [0, 0, 0, 0])
