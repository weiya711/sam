import subprocess
import glob
import shutil
import os

<<<<<<< HEAD
#run script like this: python setup_tiling_mat.py
=======
from sam.util import SUITESPARSE_PATH
>>>>>>> origin/avb03-tiling

data = ['rel5']
# app_name = "matmul_ijk"
# app_name = "mat_elemadd"
app_name = "mat_mattransmul"
# data_file = open("scripts/tensor_names/suitesparse_valid_mid50.txt")
# data_file_lines = data_file.readlines()
# for line in data_file_lines:
#    data.append(line[:-1])
os.makedirs("tiles_compiled", exist_ok=True)
os.makedirs("extensor_mtx", exist_ok=True)

os.environ["SUITESPARSE_PATH"] = "/nobackup/owhsu/sparse-datasets/suitesparse/"
os.environ["FROSTT_PATH"] = "/nobackup/owhsu/sparse-datasets/frostt/"
os.environ["SUITESPARSE_FORMATTED_PATH"] = "/home/avb03/sam/SUITESPARSE_FORMATTED"
os.environ["FROSTT_FORMATTED_TACO_PATH"] = "/home/avb03/sam/FROST_FORMATTED_TACO"
os.environ["FROSTT_FORMATTED_PATH"] = "/home/avb03/sam/FROST_FORMATTED"
os.environ["TACO_TENSOR_PATH"] = "/home/avb03/sam/TACO_TENSOR"

for datum in data:
   mtx_file = glob.glob(f"{SUITESPARSE_PATH}/{datum}.mtx")[0]
   os.makedirs("extensor_mtx", exist_ok=True)
   shutil.copy(mtx_file,f"extensor_mtx/{datum}.mtx")
   
   command = f"./scripts/suitesparse_memory_model_runner.sh {datum} {app_name}"
   os.system(command)

   os.makedirs("tiles_compiled", exist_ok=True)
   copy_rename = f"cp -r tiles/{app_name} tiles_compiled/{app_name}_{datum}"
   os.system(copy_rename)
   #Need to change docker name**
   docker_copy_command = f"docker cp tiles_compiled/{app_name}_{datum} jadivara_randommatrix:/aha/garnet/tiles_{app_name}_{datum}"
   os.system(docker_copy_command) 
