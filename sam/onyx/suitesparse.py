import subprocess
import os
import argparse
import shutil


def log_error(log_file, errorcode, errormsg):
    with open(log_file, "a+") as lf:
        lf.write(str(errorcode) + "," + errormsg + "\n")


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


def get_wget_cmd(log_file, basedir, tarname):
    download_filename = os.path.join(basedir, "sam", "scripts", "download_suitesparse.sh")
    lines = None
    with open(download_filename, "r") as ff:
        # Need to remove last "\n" character
        lines = [line[:-1] for line in ff if line[:-1].endswith(tarname)]
    error = lines is None or lines == []

    if error:
        print("FAILED: Suitesparse wget command not found for " + tarname)
        log_error(log_file, "Custom", "Failed finding wget in download_suitesparse")
    else:
        print("\n".join(lines))
    # Get the first line that matches
    return error, lines


def get_suitesparse_online(log_file, basedir, mtx, mtx_dir, check=True):
    # cwd = os.getcwd()
    if not os.path.isdir(mtx_dir):
        create_dir(mtx_dir)
    else:
        clean_dir(mtx_dir)

    # change_dir(mtx_dir)

    tarname = mtx + ".tar.gz"
    error1, wget_cmd = get_wget_cmd(log_file, basedir, tarname)
    error = error1
    if not error1:
        error |= run_process(wget_cmd[0], log_file, cwd=mtx_dir, split=True, check=check)

    untar_cmd = f"tar -xvf {tarname} --strip=1"
    if not error:
        error |= run_process(untar_cmd, log_file, cwd=mtx_dir, split=True, check=check)

    if not error1:
        os.remove(os.path.join(mtx_dir, tarname))

    # change_dir(cwd)
    return error


def generate_suitesparse(log_file, basedir, benchname, mtx, mtx_dir, matrix_tmp_dir, check=True):
    tensorpath = os.path.join(mtx_dir, mtx + ".mtx")
    os.environ["SUITESPARSE_TENSOR_PATH"] = tensorpath
    gen_suitesparse_cmd = ["python",
                           os.path.join(basedir, "sam/scripts/datastructure_suitesparse.py"), "-n", mtx,
                           "-b", benchname, "-hw", "--out", matrix_tmp_dir]
    error = run_process(gen_suitesparse_cmd, log_file, check=check)

    return error


def run_build_tb(log_file, basedir, sparse_test_basedir, benchname, matrix_tmp_dir, check=True):
    build_tb_command = ["python", os.path.join(basedir,
                                               "garnet/tests/test_memory_core/build_tb.py"), "--ic_fork",
                        "--sam_graph", os.path.join(basedir,
                                                    f"sam/compiler/sam-outputs/dot/{benchname}.gv"), "--seed", f"{0}",
                        "--dump_bitstream", "--add_pond", "--combined",
                        "--pipeline_scanner", "--base_dir", sparse_test_basedir,
                        "--fiber_access", "--give_tensor", "--dump_glb", "--matrix_tmp_dir", matrix_tmp_dir]
                        #"--fiber_access", "--give_tensor", "--gen_verilog", "--gen_pe", "--dump_glb", "--matrix_tmp_dir", matrix_tmp_dir]
                        #"--fiber_access", "--give_tensor", "--trace"]
                        #"--fiber_access", "--give_tensor", "--matrix_tmp_dir", matrix_tmp_dir, "--trace"]
    error, output_txt = run_process(build_tb_command, log_file, check=check, return_stdout=True)

    print("OUTPUT TXT")
    #print(output_txt)

    split_ot = output_txt.split("\n")

    cyc_count = 0

    for line in split_ot:
        if "Cycle" in line:
            print(line)
            ls = line.split()
            cyc_count = int(ls[-1]) 
        #print(line)

    return error, cyc_count


def run_process(command, log_file=None, cwd=None, split=False, check=True, return_stdout=False):
    if split:
        command = command.split()

    run_result = subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True)
    #return_code = subprocess.run(command, cwd=cwd, check=check).returncode
    return_code = run_result.returncode
    output_txt = run_result.stdout

    cycle_count = [line_ for line_ in output_txt if "Cycle Count" in line_]

    print("FOUND CYCLE COUNT")
    print(cycle_count)
    #exit()

    error = return_code != 0
    if error:
        print("FAILED: " + " ".join(command))
        log_file = "./suitesparse_run.log" if log_file is None else log_file
        log_error(log_file, return_code, " ".join(command))

    if return_stdout:
        return error, output_txt
    else:
        return error


def run_bench(benchname, args, matrices, stats):
    cwd = os.getcwd()

    basedir = "/aha" if args.docker else cwd

    # Hardcoded by default
    sparse_test_basedir = os.path.join(basedir,
                                       "garnet/SPARSE_TESTS/")

    # Temporary directory that stores wget suitesparse file
    mtx_dir = args.mtx_dir if args.mtx_dir is not None else os.path.join(basedir, "suitesparse")

    # Directory that the formatted array files need to go to
    matrix_tmp_dir = args.matrix_tmp_dir if args.matrix_tmp_dir is not None else os.path.join(sparse_test_basedir,
                                                                                              "MAT_TMP_DIR")

    log_file = args.log if args.log is not None else os.path.join(cwd, "suitesparse_run.log")
    if not args.append_log:
        os.remove(log_file)

    check = not args.continue_run

    for mtx in matrices:
        with open(log_file, "a+") as lf:
            lf.write(mtx + ":")

        if get_suitesparse_online(log_file, basedir, mtx, mtx_dir, check):
            continue
        if generate_suitesparse(log_file, basedir, benchname, mtx, mtx_dir, matrix_tmp_dir, check):
            continue
        err, cyc_count = run_build_tb(log_file, basedir, sparse_test_basedir, benchname, matrix_tmp_dir, check)
        #if run_build_tb(log_file, basedir, sparse_test_basedir, benchname, matrix_tmp_dir, check):
        if err:
            continue
        else:
            with open(log_file, "a+") as lf:
                lf.write("SUCCESS\n")
            pstr = f"{benchname}, {mtx}, {cyc_count}\n"
            stats.append(pstr)

        clean_dir(mtx_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Suitesparse Tester for SCGRA')
    parser.add_argument('--benchname', type=str, default="all")
    parser.add_argument('--matrix_file', type=str, default=None)
    parser.add_argument('--docker', action="store_true")
    parser.add_argument('--matrix_tmp_dir', type=str, default=None)
    parser.add_argument('--mtx_dir', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--append_log', action="store_true")
    parser.add_argument('--continue_run', action="store_true")
    parser.add_argument('--perf_log', type=str, default=None)
    args = parser.parse_args()

    benchmarks = ["matmul_ijk", "mat_elemmul", "mat_elemadd", "mat_elemadd3"]

    full_stats = ["BENCH, MATRIX, CYCLE_COUNT\n"]

    # Get matrices
    matrices = None
    with open(args.matrix_file, "r") as ff:
        matrices = ff.read().splitlines()
        print(matrices)
    assert matrices is not None, "Error opening file " + args.matrix_file

    # Run on all benchmarks
    if args.benchname == "all":
        for benchname in benchmarks:
            print("Running for bench", benchname, "...")
            run_bench(benchname, args, matrices, full_stats)
    # Only run on one
    else:
        benchname = args.benchname
        assert benchname in benchmarks

        print("Running for bench", benchname, "...")
        run_bench(benchname, args, matrices, full_stats)

    if args.perf_log is not None:
        with open(args.perf_log, "w+") as plog_file:
            plog_file.writelines(full_stats)

