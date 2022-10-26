import scipy
import scipy.sparse
import os
import scipy.io
import numpy as np


def generate_gold_matmul_tiled(tile_crd_b, tile_crd_c, dirname, out_format="ss01"):
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
            if B_scipy.data[itr] < 1:
                B_scipy.data[itr] = 1
            else:
                B_scipy.data[itr] = int(B_scipy.data[itr])
            itr += 1
        B_scipy = B_scipy.tocsr()
        C_scipy = scipy.io.mmread(C_filename)
        itr = 0
        for i, j, v in zip(C_scipy.row, C_scipy.col, C_scipy.data):
            if C_scipy.data[itr] < 1:
                C_scipy.data[itr] = 1
            else:
                C_scipy.data[itr] = int(C_scipy.data[itr])
            itr += 1
        C_scipy = C_scipy.tocsc()
        gold_nd = (B_scipy @ C_scipy)
        gold_out = gold_nd.tocoo()
        assert tile_crd_b[1] == tile_crd_c[0] and tile_crd_b[3] == tile_crd_c[2]
        scipy.io.mmwrite(dirname + "out_" + str(tile_crd_b[0]) + "_" + str(tile_crd_b[1]) + "_" + str(tile_crd_c[1]) + "_" + str(tile_crd_b[2]) + "_" + str(tile_crd_c[2]) + "_" + str(tile_crd_c[3]) + ".mtx", gold_out)

if __name__ == "__main__":
    outdir = "/nobackup/rsharma3/Sparsity/simulator/old_sam/sam/tiles/matmul_ikj/output/"
    for i00 in range(5):
        for k00 in range(5):
            for j00 in range(5):
                for i0 in range(2):
                    for k0 in range(2):
                        for j0 in range(2):
                            generate_gold_matmul_tiled([i00, k00, i0, k0], [k00, j00, k0, j0], outdir)
