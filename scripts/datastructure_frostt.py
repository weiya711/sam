import argparse
import os
from pathlib import Path
from util import parse_taco_format

cwd = os.getcwd()

outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
taco_format_dirname = os.getenv('FROSTT_FORMATTED_TACO_PATH')

formats = ["sss", "dss", "dds", "ddd", "dsd", "sdd", "sds", "ssd"]

parser = argparse.ArgumentParser(description="Process some Frostt tensors into per-level datastructures")
parser.add_argument('-n', '--name', metavar='fname', type=str, action='store',
                    help='tensor name to run format conversion on one frostt tensor')
parser.add_argument('-f', '--format', metavar='fformat', type=str, action='store',
                    help='The format that the tensor should be converted to')
parser.add_argument('-i', '--int', action='store_false', default=True, help='Safe sparsity cast to int for values')
parser.add_argument('-s', '--shift', action='store_false', default=True, help='Also format shifted tensor')
args = parser.parse_args()

out_path = Path(outdir_name)
out_path.mkdir(parents=True, exist_ok=True)

if args.name is None:
    print("Please enter a tensor name")
    exit()

if taco_format_dirname is None:
    print("Please set the FROSTT_FORMATTED_TACO_PATH environment variable")
    exit()

if args.format is not None:
    assert args.format in formats
    taco_format_orig_filename = os.path.join(taco_format_dirname, args.format, args.name + '.txt')
    taco_format_shift_filename = os.path.join(taco_format_dirname, args.format, args.name + '_shift.txt')

    # Original
    outdir_orig_name = os.path.join(outdir_name, args.name, 'orig', args.format)
    outdir_orig_path = Path(outdir_orig_name)
    outdir_orig_path.mkdir(parents=True, exist_ok=True)

    parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format)

    # Shifted
    if args.shift:
        outdir_shift_name = os.path.join(outdir_name, args.name, 'shifted', args.format)
        outdir_shift_path = Path(outdir_shift_name)
        outdir_shift_path.mkdir(parents=True, exist_ok=True)

        parse_taco_format(taco_format_shift_filename, outdir_shift_name, 'C', args.format)
