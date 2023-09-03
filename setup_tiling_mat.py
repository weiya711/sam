import subprocess
import glob
import shutil
import os
import re
import sys

from sam.util import SUITESPARSE_PATH

## PARAMS ######################################################################

# data = ['rajat12']

# data = [sys.argv[2]]
# tilesizes = [int(sys.argv[3])]
# app_name = "mat_elemadd"
# app_name = "mat_elemmul"
# app_name = "mat_sddmm"
# app_name = "matmul_ijk"           
app_name = sys.argv[1]
# app_name = "mat_elemmul"
# app_name = "mat_residual"

data = []
data_file = open("onyx_final_eval_mid50_tensor_names.txt")
data_file_lines = data_file.readlines()
for line in data_file_lines:
   data.append(line[:-1])

with open('matmul_tilesize_list.txt', 'r') as file:
    lines = file.readlines()

tilesizes = [int(line.strip()) for line in lines]
print("TILESIZES: ", tilesizes)
print("DATA: ", data)

mode_to_exclude = 0
addition_vector_name = "d" #mattransmul (d) and residual (b) only

other_tensors = ["c"]
samples_directory = f"samples/{app_name}"
docker_path = f"avb03-sparse-tiling"
use_dataset_files = False

###############################################################################

def write_to_line(file_path, line_number, new_content):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    if line_number > len(lines) or line_number < 1:
        # Line number is out of range
        return

    lines[line_number - 1] = new_content + '\n'

    with open(file_path, 'w') as file:
        file.writelines(lines)

def replace_ones_with_zeros(mtx_file):
    with open(mtx_file, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        values = line.split()
        if len(values) >= 3:
            values[2] = '0'
        new_lines.append(' '.join(values))

    with open(mtx_file, 'w') as file:
        file.writelines(new_lines)


i = 0
for datum in data:
   tilesize = tilesizes[i]
   
   yaml_file = "sam/sim/src/tiling/memory_config_onyx.yaml"
   mem_tile_line = f"Mem_tile_size: {tilesize}"
   print(mem_tile_line)
   write_to_line(yaml_file, 19, mem_tile_line)

   rmdir = f"rm -rf tiles/{app_name}"
   os.system(rmdir)

   mtx_file = glob.glob(f"{SUITESPARSE_PATH}/{datum}.mtx")[0]
   os.makedirs("extensor_mtx", exist_ok=True)
   shutil.copy(mtx_file,f"extensor_mtx/{datum}.mtx")
   
   command = f"./scripts/suitesparse_memory_model_runner.sh {datum} {app_name}"
   os.system(command)

   directories = glob.glob(f'tiles/{app_name}/formatted/tensor_[a-z]*')

   #for vectors, do cleanup
   for directory in directories:
      print(directory)
      match = re.search(r'tensor_([a-z])', directory)
      if match:
         lowercase_letter = match.group(1)
      
      crd_file = os.path.join(directory, f"{lowercase_letter}{mode_to_exclude}_crd.txt")
      seg_file = os.path.join(directory, f"{lowercase_letter}{mode_to_exclude}_seg.txt")

      # if os.path.exists(crd_file):
      #   os.remove(crd_file)

      # if os.path.exists(seg_file):
      #    os.remove(seg_file)

   samples_with_addition_vector = None
   
   # dense tile replacement for addition
   if app_name == "mat_mattransmul" or app_name == "mat_residual":
      # samples_with_addition_vector = glob.glob(f"{samples_directory}/*[{addition_vector_name}]*")
      # samples_with_addition_vector = glob.glob(f"{samples_directory}/mtm_w_0_1/tensor_d_tile_0_0")
      samples_with_addition_vector = glob.glob(f"{samples_directory}/mtm_w_0_1_BAK")


      print(samples_with_addition_vector)
      #fill in missing tiles with blanks
      for sample in samples_with_addition_vector:
         file_path = os.path.join(sample, f"{addition_vector_name}_vals.txt")

         with open(file_path, "r") as file:
            file_contents = file.read()
         
         file_contents = file_contents.replace("1", "0")

         with open(file_path, "w") as file:
            file.write(file_contents)

      tile_range = [(0,i) for i in range(8)] + [(1,i) for i in range(4)]

      for i,j in tile_range:
         tile_dir = f"tiles/{app_name}/formatted/tensor_{addition_vector_name}_tile_{i}_{j}"

         if not os.path.exists(tile_dir):
            # replace_ones_with_zeros("samples/mat_mattransmul/tensor_d_dense_mtx.mtx")

            # copy_over_to_mtx_dir = f"cp samples/mat_mattransmul/tensor_d_dense_gold_stash.mtx tiles/{app_name}/mtx/tensor_{addition_vector_name}_tile_{i}_{j}.mtx"
            # os.system(copy_over_to_mtx_dir)

            sample_tile_dir = samples_with_addition_vector[0]

            if os.path.exists(sample_tile_dir):
               shutil.copytree(sample_tile_dir, tile_dir)   

   dump_gold_tiles = f"python3 scripts/tiling/generate_gold_mattransmul.py --yaml_name memory_config_extensor_17M_llb.yaml"
   os.system(dump_gold_tiles)

   # os.makedirs("tiles_compiled", exist_ok=True)
   # copy_rename = f"cp -r tiles/{app_name} tiles_compiled/{app_name}_{datum}"
   # print(copy_rename)
   # os.system(copy_rename)

   docker_clean = f"docker exec {docker_path} rm -r /aha/garnet/tiles_{app_name}_{datum}"
   print(docker_clean)
   os.system(docker_clean)

   docker_copy_command = f"docker cp tiles {docker_path}:/aha/garnet/tiles_{app_name}_{datum}"
   print(docker_copy_command)
   os.system(docker_copy_command) 

   i = i+1
