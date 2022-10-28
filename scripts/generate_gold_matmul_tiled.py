import scipy
import scipy.sparse
import os
import scipy.io
import numpy as np
import yaml
import math
import pickle


def round_sparse(x):
    if 0.0 <= x < 1:
        return 1
    elif 0.0 > x > -1:
        return -1
    elif x >= 0.0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

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
        #print(B_scipy)
        for i, j, v in zip(B_scipy.row, B_scipy.col, B_scipy.data):
            # print(B_scipy.data)
            # print(i, " ", j, " ", v)
            B_scipy.data[itr] = round_sparse(B_scipy.data[itr])
            #if B_scipy.data[itr] < 1 and B_scipy.data[itr] > 0:
            #    B_scipy.data[itr] = 1
            #elif B_scipy.data[itr] < 0 and B_scipy.data[itr] > -1:
            #    B_scipy.data[itr] = -1
            #else:
            #    B_scipy.data[itr] = int(B_scipy.data[itr])
            itr += 1
        B_scipy = B_scipy.tocsr()
        C_scipy = scipy.io.mmread(C_filename)
        #print("___________________") 
        #print(B_scipy)
        itr = 0
        for i, j, v in zip(C_scipy.row, C_scipy.col, C_scipy.data):
            if C_scipy.data[itr] < 1 and C_scipy.data[itr] > 0:
                C_scipy.data[itr] = 1
            elif C_scipy.data[itr] < 0 and C_scipy.data[itr] > -1:
                C_scipy.data[itr] = -1
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

    #generate_gold_matmul_tiled([0, 1, 2, 9], [1, 0, 9, 0], outdir)
    
    #generate_gold_matmul_tiled([0, 1, 0, 7], [1, 0, 7, 0], outdir)
    #quit()    with open("/nobackup/rsharma3/Sparsity/simulator/old_sam/sam/tiles/matmul_ikj/tensor_sizes", "rb") as ff:
 
    with open("tiles/matmul_ikj/tensor_sizes", "rb") as ff:
        sizes_dict_level_full = pickle.load(ff)

    with open("sam/sim/src/tiling/memory_config_real.yaml", "r") as stream:
        loop_config = yaml.safe_load(stream)

    struct = {"i00": 1 + int(sizes_dict_level_full["B"][0])//(loop_config["Glb_tile_size"]*loop_config["Mem_tile_size"]), "k00": 1 + int(sizes_dict_level_full["B"][1])//(loop_config["Glb_tile_size"]*loop_config["Mem_tile_size"]), "j00": 1 + int(sizes_dict_level_full["C"][1])//(loop_config["Glb_tile_size"]*loop_config["Mem_tile_size"]), "i0": loop_config["Glb_tile_size"], "k0": loop_config["Glb_tile_size"], "j0": loop_config["Glb_tile_size"]}
    print(struct)
    #quit()
    for i00 in range(struct["i00"]):
        for k00 in range(struct["k00"]):
            for j00 in range(struct["j00"]):
                for i0 in range(struct["i0"]):
                    for k0 in range(struct["k0"]):
                        for j0 in range(struct["j0"]):
                            generate_gold_matmul_tiled([i00, k00, i0, k0], [k00, j00, k0, j0], outdir)
