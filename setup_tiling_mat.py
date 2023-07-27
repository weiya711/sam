import subprocess
import glob
import shutil
import os

from sam.util import SUITESPARSE_PATH

data = ['rel5']
# app_name = "mat_elemadd"
app_name = "mat_mattransmul"
# data_file = open("scripts/tensor_names/suitesparse_valid_mid50.txt")
# data_file_lines = data_file.readlines()
# for line in data_file_lines:
#    data.append(line[:-1])

for datum in data:
   mtx_file = glob.glob(f"{SUITESPARSE_PATH}/{datum}.mtx")[0]
   os.makedirs("extensor_mtx", exist_ok=True)
   shutil.copy(mtx_file,f"extensor_mtx/{datum}.mtx")
   
   command = f"./scripts/suitesparse_memory_model_runner.sh {datum} {app_name}"
   os.system(command)

   os.makedirs("tiles_compiled", exist_ok=True)
   copy_rename = f"cp -r tiles/{app_name} tiles_compiled/{app_name}_{datum}"
   os.system(copy_rename)

   docker_copy_command = f"docker cp tiles_compiled/{app_name}_{datum} avb03-sparse-tiling:/aha/garnet/tiles_{app_name}_{datum}"
   os.system(docker_copy_command) 
