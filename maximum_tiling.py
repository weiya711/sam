import glob
import sys
import numpy as np
import scipy
import os
import re

class EarlyReturn():
    pass

def get_files_from_dir(path, operands):
    operand_files = {}
    for operand in operands:
        operand_files[operand] = glob.glob(os.path.join(path, f"*{operand}*.mtx"))

    return operand_files
def get_tile_id(string):
    indices = [m.start() for m in re.finditer("tile", string)]
    if len(indices) >= 2:
        substring = string[indices[1] + len("tile") + 1:]
        substring = substring.rstrip(".mtx")
        numbers = substring.split("_")
    return numbers

def pair_tiles(app_name):
    path = f"tiles/{app_name}/mtx"
    tile_pairing = {}

    operands = []
    if "matmul" in app_name:
        operands = ["B", "C"]
        operand_files = get_files_from_dir(path, operands)
        b_tensors = operand_files["B"]
        c_tensors = operand_files["C"]
        
        tile = 0
        for b in b_tensors:
            for c in c_tensors:
                b_loc = get_tile_id(b)
                c_loc = get_tile_id(c)
                if (b_loc[1] == c_loc[0] and b_loc[3] == c_loc[2]):
                    tile_pairing[tile] = [b, c]
                    tile += 1

    return tile_pairing

def read_mtx(mtx_path):
    matrix = scipy.io.mmread(mtx_path)
    arr = np.array(matrix.todense())
    return arr

def compute_outputs(tile_pairing, app_name, limit=900):
    for key, value in tile_pairing.items():
        if "matmul" in app_name:
            B_mat = read_mtx(value[0])
            C_mat = read_mtx(value[1])
            C_mat = np.transpose(C_mat)
            out = np.matmul(B_mat, C_mat)
            # if np.count_nonzero(out) > limit:
            if np.any(out):
                print("tile = ", key)
                print("B_tile_ID = ", value[0])
                print("C_tile_ID = ", value[1])
                print("out = ", out)
                print("count = ", np.count_nonzero(out))
                # return EarlyReturn()
                breakpoint()
                break
    return None
            
def find_optimal_tilesize(app_name, datum, initial=30, step_size=10):
    tile_size = initial
    max_tile_size = initial
    prev_tile_pairing = None

    for _ in range(10):
        call_tiling = f"python3 setup_tiling_mat.py {app_name} {datum} {tile_size} > temp.txt"
        os.system(call_tiling)
        print(call_tiling)

        tile_pairing = pair_tiles(app_name)
        exit_status = compute_outputs(tile_pairing, app_name)
        if isinstance(exit_status, EarlyReturn):
            max_tile_size = tile_size - step_size
            return max_tile_size, prev_tile_pairing
        
        tile_size += step_size
        print("***********************")
        print("tile size = ", tile_size)
        print("***********************")
        prev_tile_pairing = tile_pairing
    
    return None
       

if __name__ == "__main__":
    app_name = "matmul_ijk"
    datum = "qiulp"

    tile_pairing = pair_tiles(app_name)
    compute_outputs(tile_pairing, app_name)

    # max_tile_size, tile_pairing = find_optimal_tilesize(app_name, datum)
    # print("-"*20)
    # print(f"MAX TILESIZE for {app_name}, {datum}: {max_tile_size}")
    # print(f"NUMBER OF TILES: {len(tile_pairing.keys())}")
    # print("-"*20)