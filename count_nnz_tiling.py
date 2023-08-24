import glob
def count_nonzeros(matrix_values_file):
    with open(matrix_values_file, 'r') as values_file:
        matrix_values = [float(val) for val in values_file.readlines()]

    nonzeros = sum(1 for val in matrix_values if val != 0)

    return nonzeros


tile_dirs = glob.glob("SPARSE_TESTS/MAT_TMP_DIR/tile*")
num_tiles = len(tile_dirs)
limit = 900
print("there are ", num_tiles, "tiles")


for tile_num in range(0,num_tiles):
    tot_num_nonzeros = 0

    tensor_C_values_file = f'SPARSE_TESTS/MAT_TMP_DIR/tile{tile_num}/tensor_C_mode_vals'

    num_nonzeros = count_nonzeros(tensor_C_values_file)
    tot_num_nonzeros += num_nonzeros

    tensor_C_values_file = f'SPARSE_TESTS/MAT_TMP_DIR/tile{tile_num}/tensor_B_mode_vals'

    num_nonzeros = count_nonzeros(tensor_C_values_file)
    tot_num_nonzeros += num_nonzeros

    tensor_C_values_file = f'SPARSE_TESTS/MAT_TMP_DIR/tile{tile_num}/tensor_D_mode_vals'

    num_nonzeros = count_nonzeros(tensor_C_values_file)
    tot_num_nonzeros += tot_num_nonzeros
    
    if tot_num_nonzeros >= limit:
        print("tot_num_nonzeros: ", tot_num_nonzeros)
        print("error! too many nonzeros in matrices")
        raise Exception