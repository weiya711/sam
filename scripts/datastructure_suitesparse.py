import argparse
import os
import shutil

from pathlib import Path

from util import FormatWriter, SuiteSparseTensor, InputCacheSuiteSparse
from sam.util import SUITESPARSE_FORMATTED_PATH, ScipyTensorShifter

cwd = os.getcwd()
SS_PATH = os.getenv('SUITESPARSE_TENSOR_PATH', default=os.path.join(cwd, 'suitesparse'))

out_dirname = SUITESPARSE_FORMATTED_PATH

all_formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense", "denseT"]
formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense"]
scipy_formats = ["coo", "csr", "csc"]

#
# def get_datastructure_string(format, mode):
#     if format == ['d', 'd'] and mode == [0, 1]:
#         return "dense"
#     elif format == ['d', 'd']:
#         return "denseT"
#     elif format == ['d', 's'] and mode == [0, 1]:
#         return "csr"
#     elif format == ['d', 's']:
#         return "csc"
#     elif format == ['s', 's'] and mode == [0, 1]:
#         return "dcsr"
#     elif format == ['s', 's']:
#         return "dcsc"
#     elif format == ['c', 'q'] and mode == [0, 1]:
#         return "coo"
#     elif format == ['c', 'q']:
#         return "cooT"
#     else:
#         return ""


parser = argparse.ArgumentParser(description="Process some suitesparse matrices into per-level datastructures")
parser.add_argument('-n', '--name', metavar='ssname', type=str, action='store', help='tensor name to run format '
                                                                                     'conversion on one SS tensor')
parser.add_argument('-f', '--format', metavar='ssformat', type=str, action='store', help='The format that the tensor '
                                                                                         'should be converted to')
parser.add_argument('-c', '--combined', action='store_true', default=False, help='Whether the formatted datastructures '
                                                                                 'should be in separate files')
parser.add_argument('-o', '--omit-dense', action='store_true', default=False, help='Do not create fully dense format')
parser.add_argument('-i', '--cast', action='store_true', default=False, help='Safe sparsity cast to int for values')
parser.add_argument('-hw', '--hw', action='store_true', default=False, help='Only generate formats used for hardware '
                                                                            'testing (all sparse levels, concordant)')
parser.add_argument('-b', '--benchname', type=str, action='store', help='test name to run format '
                                                                        'conversion on')
parser.add_argument('--out', type=str, default=None)

parser.add_argument('--dir', type=str, action='store', help='If this is defined use this as the directory,'
                                                            'else use SUITESPARSE_TENSOR_PATH', default=None)

args = parser.parse_args()

inputCache = InputCacheSuiteSparse()
formatWriter = FormatWriter(args.cast)
shifter = ScipyTensorShifter()

out_path = Path(out_dirname)
out_path.mkdir(parents=True, exist_ok=True, mode=0o777)

if args.name is None:
    print("Please enter a matrix name")
    exit()

print("SUITESPARSE_PATH", SS_PATH)
tensor = SuiteSparseTensor(SS_PATH)

if args.format is not None:
    assert args.format in formats
    filename = os.path.join(out_path, args.name + "_" + args.format + ".txt")

    coo = inputCache.load(tensor, False)
    formatWriter.writeout(coo, args.format, filename)
elif args.combined:
    for format_str in formats:
        filename = os.path.join(out_path, args.name + "_" + format_str + ".txt")
        print("Writing " + args.name + " " + format_str + "...")

        coo = inputCache.load(tensor, False)
        formatWriter.writeout(coo, format_str, filename)

        shifted_filename = os.path.join(out_path, args.name + "_shifted_" + format_str + ".txt")
        shifted = shifter.shiftLastMode(coo)
        formatWriter.writeout(shifted, format_str, shifted_filename)

        trans_filename = os.path.join(out_path, args.name + "_trans_shifted_" + format_str + ".txt")
        trans_shifted = shifted.transpose()
        formatWriter.writeout(trans_shifted, format_str, trans_filename)
elif args.hw:
    print("Writing " + args.name + " for test " + args.benchname + "...")

    dirname = args.out if args.out is not None else os.path.join(out_path, args.name, args.benchname)
    dirpath = Path(dirname)
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)

    if "mat_mattransmul" in args.benchname or "mat_residual" in args.benchname:
        tensorname = "C"
    else:
        tensorname = "B"
    coo = inputCache.load(tensor, False)

    # These benchmarks need format_str == "ss10"
    if args.benchname not in ["matmul_kij", "mat_vecmul", "mat_mattransmul"]:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss01")

    if "matmul_ijk" in args.benchname:
        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss10")

    elif "matmul_ikj" in args.benchname:
        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss01")

    elif "matmul_kij" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")

        shifted = shifter.shiftLastMode(coo)

        print("Writing " + args.name + " shifted and transposed...")
        tensorname = "C"
        trans_shifted = shifted.transpose()
        formatWriter.writeout_separate_sparse_only(trans_shifted, dirname, tensorname, format_str="ss01")

    elif "mat_elemadd3" in args.benchname:
        print("Writing " + args.name + " shifted...")
        tensorname = "C"
        shifted = shifter.shiftLastMode(coo)
        formatWriter.writeout_separate_sparse_only(shifted, dirname, tensorname, format_str="ss01")

        print("Writing " + args.name + " shifted2...")
        tensorname = "D"
        shifted2 = shifter.shiftLastMode(shifted)
        formatWriter.writeout_separate_sparse_only(shifted2, dirname, tensorname, format_str="ss01")

    elif "mat_elemadd" in args.benchname or "mat_elemmul" in args.benchname:
        print("Writing " + args.name + " shifted...")
        tensorname = "C"
        shifted = shifter.shiftLastMode(coo)
        formatWriter.writeout_separate_sparse_only(shifted, dirname, tensorname, format_str="ss01")

    elif "mat_mattransmul" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
    elif "mat_vecmul" in args.benchname:
        formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
    elif "mat_sddmm" in args.benchname:
        pass
    elif "mat_residual" in args.benchname:
        pass
    else:
        raise NotImplementedError

else:
    print("Writing " + args.name + " original...")
    dirname = os.path.join(out_path, args.name, "orig")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "B"
    coo = inputCache.load(tensor, False)
    formatWriter.writeout_separate(coo, dirname, tensorname, omit_dense=args.omit_dense)

    print("Writing " + args.name + " shifted...")
    dirname = os.path.join(out_path, args.name, "shift")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "C"
    shifted = shifter.shiftLastMode(coo)
    formatWriter.writeout_separate(shifted, dirname, tensorname, omit_dense=args.omit_dense)

    print("Writing " + args.name + " shifted and transposed...")
    dirname = os.path.join(out_path, args.name, "shift-trans")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "C"
    trans_shifted = shifted.transpose()
    formatWriter.writeout_separate(trans_shifted, dirname, tensorname, omit_dense=args.omit_dense)
