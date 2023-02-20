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


def get_suitesparse_online(log_file, basedir, mtx, mtx_dir, check=True, clean=False):
    # cwd = os.getcwd()
    if not os.path.isdir(mtx_dir):
        create_dir(mtx_dir)
    elif clean:
        clean_dir(mtx_dir)

    # change_dir(mtx_dir)

    path_ = os.path.join(mtx_dir, f"{mtx}.mtx")

    if not os.path.isfile(path_):

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
    else:
        return 0


def generate_suitesparse(log_file, basedir, benchname, mtx, mtx_dir, matrix_tmp_dir, check=True):

    # Generate subfolder
    final_mat_path = os.path.join(matrix_tmp_dir, f"{mtx}")
    if not os.path.isdir(matrix_tmp_dir):
        os.makedirs(matrix_tmp_dir)
    if not os.path.isdir(final_mat_path):
        os.makedirs(final_mat_path)

    tensorpath = os.path.join(mtx_dir, mtx + ".mtx")
    os.environ["SUITESPARSE_TENSOR_PATH"] = tensorpath
    gen_suitesparse_cmd = ["python",
                           os.path.join(basedir, "sam/scripts/datastructure_suitesparse.py"), "-n", mtx,
                           "-b", benchname, "-hw", "--out", final_mat_path]
    error = run_process(gen_suitesparse_cmd, log_file, check=check)

    return error


def run_build_tb(log_file, basedir, sparse_test_basedir, benchname, matrix_tmp_dir, check=True, gen_verilog=False):
    build_tb_command = ["python", os.path.join(basedir,
                                               "garnet/tests/test_memory_core/build_tb.py"), "--ic_fork",
                        "--sam_graph", os.path.join(basedir,
                                                    f"sam/compiler/sam-outputs/dot/{benchname}.gv"), "--seed", f"{0}",
                        "--dump_bitstream", "--add_pond", "--combined",
                        "--pipeline_scanner", "--base_dir", sparse_test_basedir,
                        "--fiber_access", "--give_tensor", "--dump_glb", "--tensor_locs", matrix_tmp_dir, "--just_glb"]

    if gen_verilog:
        build_tb_command.append('--gen_verilog')
        build_tb_command.append('--gen_pe')

    error, output_txt = run_process(build_tb_command, log_file, check=check, return_stdout=True)

    cyc_count = 0

    return error, cyc_count


def run_build_tb_all(log_file, basedir, sparse_test_basedir, benchname,
                     matrix_tmp_dir, check=True, tname=None, compile_tb=False,
                     debug=False, trace=False):
    assert tname is not None
    build_tb_command = ["python", os.path.join(basedir,
                                               "garnet/tests/test_memory_core/build_tb.py"), "--ic_fork",
                        "--sam_graph", os.path.join(basedir,
                                                    f"sam/compiler/sam-outputs/dot/{benchname}.gv"), "--seed", f"{0}",
                        "--dump_bitstream", "--add_pond", "--combined",
                        "--pipeline_scanner", "--base_dir", sparse_test_basedir,
                        "--fiber_access", "--give_tensor", "--dump_glb", "--tensor_locs", matrix_tmp_dir, "--run", f"{tname}"]

    if compile_tb:
        build_tb_command.append("--compile_tb")

    if trace:
        build_tb_command.append("--trace")

    error, output_txt = run_process(build_tb_command, log_file, check=check, return_stdout=True, debug=debug)

    split_ot = output_txt.split("\n")

    cyc_count = 0

    for line in split_ot:
        if "Cycle" in line:
            print(line)
            ls = line.split()
            cyc_count = int(ls[-1])

    return error, cyc_count


def run_process(command, log_file=None, cwd=None, split=False, check=True, return_stdout=False, debug=False):
    if split:
        command = command.split()

    run_result = None

    try:
        if debug:
            # Command should die after this...
            run_result = subprocess.run(command, cwd=cwd, check=check)
        else:
            run_result = subprocess.run(command, cwd=cwd, check=check, capture_output=True, text=True)
        return_code = run_result.returncode
        output_txt = run_result.stdout
    except subprocess.CalledProcessError:
        return_code = 1
        output_txt = ""

    error = return_code != 0
    if error:
        print("FAILED: " + " ".join(command))
        log_file = "./suitesparse_run.log" if log_file is None else log_file
        log_error(log_file, return_code, " ".join(command))

    if return_stdout:
        return error, output_txt
    else:
        return error


def run_bench(benchname, args, matrices, stats, gen_verilog, compile_tb=False,
              generate=True, run=True, debug=False, trace=False):
    cwd = os.getcwd()

    basedir = "/aha" if args.docker else os.getenv('BASEDIR')

    # Hardcoded by default
    sparse_test_basedir = os.path.join(basedir,
                                       "garnet/SPARSE_TESTS/")

    # Temporary directory that stores wget suitesparse file
    mtx_dir = args.mtx_dir if args.mtx_dir is not None else os.path.join(basedir, "suitesparse")

    # Directory that the formatted array files need to go to
    matrix_tmp_dir = args.matrix_tmp_dir if args.matrix_tmp_dir is not None else os.path.join(sparse_test_basedir,
                                                                                              "MAT_TMP_DIR")

    log_file = args.log if args.log is not None else os.path.join(cwd, "suitesparse_run.log")
    if not args.append_log and os.path.isfile(log_file):
        os.remove(log_file)

    check = not args.continue_run

    gen_vlog = gen_verilog

    clean_ = False

    for mtx in matrices:
        with open(log_file, "a+") as lf:
            lf.write(mtx + ":")

        if get_suitesparse_online(log_file, basedir, mtx, mtx_dir, check, clean=clean_):
            continue
        if generate_suitesparse(log_file, basedir, benchname, mtx, mtx_dir, matrix_tmp_dir, check):
            continue

        clean_ = False

    if generate:
        err, cyc_count = run_build_tb(log_file, basedir, sparse_test_basedir, benchname, matrix_tmp_dir, check, gen_vlog)
        gen_vlog = False

    comp_tb_ = compile_tb

    if run:
        for mtx in matrices:
            tname = f"{benchname}_{mtx}"
            err, cyc_count = run_build_tb_all(log_file, basedir, sparse_test_basedir, benchname,
                                              matrix_tmp_dir, check, tname=tname, compile_tb=comp_tb_,
                                              debug=debug, trace=trace)
            comp_tb_ = False
            pstr = f"{benchname}, {mtx}, {cyc_count}\n"
            stats.append(pstr)

    with open(log_file, "a+") as lf:
        lf.write("SUCCESS\n")

    # only generate for one matrix
    # clean_dir(mtx_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Suitesparse Tester for SCGRA')
    parser.add_argument('--benchname', type=str, default=None, nargs='+')
    parser.add_argument('--matrix_file', type=str, default=None)
    parser.add_argument('--docker', action="store_true")
    parser.add_argument('--matrix_tmp_dir', type=str, default=None)
    parser.add_argument('--mtx_dir', type=str, default=None)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--append_log', action="store_true")
    parser.add_argument('--continue_run', action="store_true")
    parser.add_argument('--gen_verilog', action="store_true")
    parser.add_argument('--perf_log', type=str, default=None)
    parser.add_argument('--generate', action="store_true")
    parser.add_argument('--run', action="store_true")
    parser.add_argument('--compile_tb', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--trace', action="store_true")
    args = parser.parse_args()

    benchmarks = ["matmul_ijk", "mat_elemmul", "mat_elemadd", "mat_elemadd3"]

    full_stats = ["BENCH, MATRIX, CYCLE_COUNT\n"]

    generate = args.generate
    run = args.run
    trace = args.trace
    dbg = args.debug

    # Get matrices
    matrices = None
    with open(args.matrix_file, "r") as ff:
        matrices = ff.read().splitlines()
        print(matrices)
    assert matrices is not None, "Error opening file " + args.matrix_file

    gen_verilog = args.gen_verilog
    compile_tb = args.compile_tb

    # Process benchname
    bname_arg = args.benchname

    use_bmarks = None

    if bname_arg is None:
        use_bmarks = benchmarks
    else:
        use_bmarks = bname_arg

    # Run on all benchmarks specified
    for benchname in use_bmarks:
        print("Running for bench", benchname, "...")
        run_bench(benchname, args, matrices, full_stats, gen_verilog, compile_tb=compile_tb,
                  generate=generate, run=run, debug=dbg, trace=trace)
        # Don't need to do it more than once.
        gen_verilog = False

    if args.perf_log is not None:
        with open(args.perf_log, "w+") as plog_file:
            plog_file.writelines(full_stats)
