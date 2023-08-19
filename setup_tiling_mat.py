import subprocess
import glob
import shutil
import os
import re

from sam.util import SUITESPARSE_PATH

## PARAMS ######################################################################

# 'rel5', 'mk9-b1', 
data = ['rel5']
# app_name = "mat_elemadd"
# app_name = "mat_elemmul"
app_name = "mat_sddmm"
# app_name = "matmul_ijk"
# app_name = "mat_elemmul"
# app_name = "mat_vecmul_ij"
# app_name = "mat_residual"
# data_file = open("scripts/tensor_names/suitesparse_valid_mid50.txt")
# data_file_lines = data_file.readlines()
# for line in data_file_lines:
#    data.append(line[:-1])
mode_to_exclude = 0
addition_vector_name = "d" #mattransmul (d) and residual (b) only

other_tensors = ["c"]
samples_directory = f"samples/{app_name}"
docker_path = f"avb03-sparse-tiling"
use_dataset_files = False
matrix_app=True

###############################################################################

os.environ["SUITESPARSE_PATH"] = "/nobackup/owhsu/sparse-datasets/suitesparse/"
os.environ["FROSTT_PATH"] = "/nobackup/owhsu/sparse-datasets/frostt/"
os.environ["SUITESPARSE_FORMATTED_PATH"] = "/home/avb03/sam/SUITESPARSE_FORMATTED"
os.environ["FROSTT_FORMATTED_TACO_PATH"] = "/home/avb03/sam/FROST_FORMATTED_TACO"
os.environ["FROSTT_FORMATTED_PATH"] = "/home/avb03/sam/FROST_FORMATTED"
os.environ["TACO_TENSOR_PATH"] = "/home/avb03/sam/TACO_TENSOR"

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


if(matrix_app):
   for datum in data:
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

      if(use_dataset_files):
         assert os.path.exists("SUITESPARSE_FORMATTED")

         temp_name = app_name
         if app_name == "mat_vecmul_ij":
            temp_name = "mat_vecmul"

         app_path_additional = f"SUITESPARSE_FORMATTED/{datum}/{temp_name}/"

         for tens in other_tensors:
            valid_dirs = glob.glob(f"tiles/{app_name}/formatted/tensor_{tens}*")
            for d in valid_dirs:
               remove_tens = f"rm {d}/*"
               print(remove_tens)
               os.system(remove_tens)

               files_to_cp = glob.glob(f"{app_path_additional}tensor_{tens}*")

               for file in files_to_cp:
                  if "mode_0_crd" in file:
                     copy_rename = f"cp {file} {d}/{tens}0_crd.txt"
                     print(copy_rename)
                     os.system(copy_rename)
                  elif "mode_1_crd" in file:
                     copy_rename = f"cp {file} {d}/{tens}1_crd.txt"
                     print(copy_rename)
                     os.system(copy_rename)
                  elif "mode_0_seg" in file:
                     copy_rename = f"cp {file} {d}/{tens}0_seg.txt"
                     print(copy_rename)
                     os.system(copy_rename)
                  elif "mode_1_seg" in file:
                     copy_rename = f"cp {file} {d}/{tens}1_seg.txt"
                     print(copy_rename)
                     os.system(copy_rename)
                  elif "vals" in file:
                     copy_rename = f"cp {file} {d}/{tens}_vals.txt"
                     print(copy_rename)
                     os.system(copy_rename)
                  elif "shape" in file:
                     copy_rename = f"cp {file} {d}/{tens}_shape.txt"
                     print(copy_rename)
                     os.system(copy_rename)


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
   
