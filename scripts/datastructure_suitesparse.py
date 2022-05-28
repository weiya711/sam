import argparse
import os
import dataclasses
import numpy

from pathlib import Path

from util import FormatWriter, SuiteSparseTensor, InputCacheSuiteSparse, ScipyTensorShifter

cwd = os.getcwd()
SS_PATH = os.getenv('SUITESPARSE_PATH')

out_dirname = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))

all_formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense", "denseT"]
formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense"]
scipy_formats = ["coo", "csr", "csc"]


def get_datastructure_string(format, mode):
    if format == ['d', 'd'] and mode == [0, 1]:
        return "dense"
    elif format == ['d', 'd']:
        return "denseT"
    elif format == ['d', 's'] and mode == [0, 1]:
        return "csr"
    elif format == ['d', 's']:
        return "csc"
    elif format == ['s', 's'] and mode == [0, 1]:
        return "dcsr"
    elif format == ['s', 's']:
        return "dcsc"
    elif format == ['c', 'q'] and mode == [0, 1]:
        return "coo"
    elif format == ['c', 'q']:
        return "cooT"
    else:
        return ""


parser = argparse.ArgumentParser(description="Process some suitesparse matrices into per-level datastructures")
parser.add_argument('-n', '--name', metavar='ssname', type=str, action='store', help='tensor name to run format '
                                                                                     'conversion on one SS tensor')
parser.add_argument('-f', '--format', metavar='ssformat', type=str, action='store', help='The format that the tensor '
                                                                                         'should be converted to')
parser.add_argument('-c', '--combined', action='store_true', default=False, help='Whether the formatted datastructures '
                                                                                 'should be in separate files')
parser.add_argument('-o', '--omit-dense', action='store_true', default=False, help='Do not create fully dense format')
parser.add_argument('-i', '--integer', action='store_false', default=True, help='Safe sparsity cast to int for values')
args = parser.parse_args()

inputCache = InputCacheSuiteSparse()
formatWriter = FormatWriter(args.integer)


out_path = Path(out_dirname)
out_path.mkdir(parents=True, exist_ok=True, mode=0o777)

if args.name is None:
    print("Please enter a matrix name")
    exit()

tensor = SuiteSparseTensor(os.path.join(SS_PATH, args.name))

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
        shifted = ScipyTensorShifter().shiftLastMode(coo)
        formatWriter.writeout(shifted, format_str, shifted_filename)

        trans_filename = os.path.join(out_path, args.name + "_trans_shifted_" + format_str + ".txt")
        trans_shifted = shifted.transpose()
        formatWriter.writeout(trans_shifted, format_str, trans_filename)
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
    shifted = ScipyTensorShifter().shiftLastMode(coo)
    formatWriter.writeout_separate(shifted, dirname, tensorname, omit_dense=args.omit_dense)

    print("Writing " + args.name + " shifted and transposed...")
    dirname = os.path.join(out_path, args.name, "shift-trans")
    dirpath = Path(dirname)
    dirpath.mkdir(parents=True, exist_ok=True, mode=0o777)
    tensorname = "C"
    trans_shifted = shifted.transpose()
    formatWriter.writeout_separate(trans_shifted, dirname, tensorname, omit_dense=args.omit_dense)
