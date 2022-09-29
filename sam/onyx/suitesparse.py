import subprocess
import os
import sys
import argparse
import shutil

def change_dir(dir_path):
    assert os.path.isdir(dir_path), dir_path + " is not a valid directory"
    os.chdir(dir_path)

def clean_dir(dir_path):
    for filename in os.listdir(dir_path):
        full_del_path = os.path.join(dir_path, filename)
        if os.path.isfile(full_del_path):
            ret = os.remove(full_del_path)
        elif os.path.isdir(full_del_path):
            ret = shutil.rmtree(full_del_path)

def create_dir(dir_path):
    os.mkdir(dir_path)

def get_suitesparse_online(mtx, mtx_dir):
    cwd = os.getcwd()
    if not os.path.isdir(mtx_dir):
        create_dir(mtx_dir)
    else:
        clean_dir(mtx_dir)

    change_dir(mtx_dir)
    
    tarname = mtx + ".tar.gz"
    wget_cmd = f"wget https://sparse.tamu.edu/MM/HB/{tarname}"
    run_process(wget_cmd, split=True)

    untar_cmd = f"tar -xvf {tarname} --strip=1"
    run_process(untar_cmd, split=True)
    os.remove(tarname)

    change_dir(cwd)

def generate_suitesparse(basedir, benchname, mtx, mtx_dir, matrix_tmp_dir):
    tensorpath = os.path.join(mtx_dir, mtx + ".mtx") 
    os.environ["SUITESPARSE_TENSOR_PATH"] = tensorpath
    gen_suitesparse_cmd = ["python",
            os.path.join(basedir, "sam/scripts/datastructure_suitesparse.py"), "-n", mtx,
            "-b", benchname, "-hw", "--out", matrix_tmp_dir]
    run_process(gen_suitesparse_cmd)

def run_build_tb(basedir, sparse_test_basedir, benchname, matrix_tmp_dir):
    print(basedir)
    print(os.path.join(basedir, "garnet/tests/test_memory_core/build_tb.py"))

    build_tb_command = ["python", os.path.join(basedir,
            "garnet/tests/test_memory_core/build_tb.py"), "--ic_fork",
            "--sam_graph", os.path.join(basedir,
            f"sam/compiler/sam-outputs/dot/{benchname}.gv"), "--seed", f"{0}",
            "--dump_bitstream", "--add_pond", "--combined",
            "--pipeline_scanner", "--base_dir", sparse_test_basedir,  
            "--fiber_access", "--give_tensor", "--matrix_tmp_dir", matrix_tmp_dir]
    run_process(build_tb_command)

def run_process(command, split=False):
    if split:
        command = command.split()

    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()

def run_bench(benchname, args, matrices):
    cwd = os.getcwd()

    basedir = "/aha" if args.docker else cwd

    # Hardcoded by default
    sparse_test_basedir = os.path.join(basedir,
            "garnet/SPARSE_TESTS/")

    # Temporary directory that stores wget suitesparse file
    mtx_dir = args.mtx_dir if args.mtx_dir is not None else os.path.join(basedir, "suitesparse")

    # Directory that the formatted array files need to go to
    matrix_tmp_dir = args.matrix_tmp_dir if args.matrix_tmp_dir is not None else os.path.join(sparse_test_basedir, "MAT_TMP_DIR")

    for mtx in matrices:
        get_suitesparse_online(mtx, mtx_dir)
        generate_suitesparse(basedir, benchname, mtx, mtx_dir, matrix_tmp_dir)
        run_build_tb(basedir, sparse_test_basedir, benchname, matrix_tmp_dir)
        clean_dir(mtx_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Suitesparse Tester for SCGRA')
    parser.add_argument('--benchname', type=str, default="all")
    parser.add_argument('--matrix_file', type=str, default=None)
    parser.add_argument('--docker', action="store_true")
    parser.add_argument('--matrix_tmp_dir', type=str, default=None)
    parser.add_argument('--mtx_dir', type=str, default=None)

    args = parser.parse_args()
    
    benchmarks = ["matmul_ijk", "mat_elemmul", "mat_elemadd", "mat_elemadd3"]

    # Get matrices
    matrices = None
    with open(args.matrix_file, "r") as ff:
        matrices = ff.read().splitlines()
    assert matrices is not None, "Error opening file " + args.matrix_file

    # Run on all benchmarks
    if args.benchname == "all":
        for benchname in benchmarks:
            print("Running for bench", benchname, "...")
            run_bench(benchname, args, matrices)
    # Only run on one
    else:
        benchname = args.benchname
        assert benchname in benchmarks

        print("Running for bench", benchname, "...")
        run_bench(benchname, args, matrices)
