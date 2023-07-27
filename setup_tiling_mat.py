import subprocess
import glob
import shutil
import os

data = ['rel5']
# app_name = "mat_elemadd"
app_name = "mat_mattransmul"
# data_file = open("scripts/tensor_names/suitesparse_valid_mid50.txt")
# data_file_lines = data_file.readlines()
# for line in data_file_lines:
#    data.append(line[:-1])

if not os.path.exists("extensor_mtx"):
   os.mkdir("extensor_mtx")

if not os.path.exists("tiles_compiled"):
   os.mkdir("tiles_compiled")

for datum in data:
   mtx_file = glob.glob(f"/nobackup/owhsu/sparse-datasets/suitesparse/{datum}.mtx")[0]
   shutil.copy(mtx_file,f"extensor_mtx/{datum}.mtx")
   
   command = f"./scripts/suitesparse_memory_model_runner.sh {datum} {app_name}"
   os.system(command)

   copy_rename = f"cp -r tiles/{app_name} tiles_compiled/{app_name}_{datum}"
   os.system(copy_rename)

   docker_copy_command = f"docker cp tiles_compiled/{app_name}_{datum} avb03-sparse-tiling:/aha/garnet/tiles_{app_name}_{datum}"
   os.system(docker_copy_command) 