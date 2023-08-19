import argparse
import os
import shutil
import scipy.sparse
import numpy as np
import sys
import random
import shutil

from pathlib import Path
from sam.util import parse_taco_format

from scripts.util.util import FormatWriter, SuiteSparseTensor, InputCacheSuiteSparse
# custom_path = '/nobackup/jadivara/sam/sam/util.py'
# sys.path.append(custom_path)
# from  import SUITESPARSE_FORMATTED_PATH, ScipyTensorShifter

cwd = os.getcwd()

formats = ["sss012", "ss01", "dss", "dds", "ddd", "dsd", "sdd", "sds", "ssd"]

parser = argparse.ArgumentParser(description="Process some Frostt tensors into per-level datastructures")
parser.add_argument('-n', '--name', metavar='fname', type=str, action='store',
                    help='tensor name to run format conversion on one frostt tensor')
parser.add_argument('-f', '--format', metavar='fformat', type=str, action='store',
                    help='The format that the tensor should be converted to')
parser.add_argument('-i', '--int', action='store_false', default=True, help='Safe sparsity cast to int for values')
parser.add_argument('-s', '--shift', action='store_false', default=True, help='Also format shifted tensor')
parser.add_argument('-o', '--other', action='store_true', default=False, help='Format other tensor')
parser.add_argument('-ss', '--suitesparse', action='store_true', default=False, help='Format suitesparse other tensor')
parser.add_argument('-hw', '--hw', action='store_true', default=False,
                    help='Format filenames as in AHA SCGRA <tensor_<name>_mode_<n|type>')
parser.add_argument('-np', '--numpy', action='store_true', default=False, help='Format numpy tensors')
parser.add_argument('-b', '--bench', type=str, default=None, help='Name of benchmark')
parser.add_argument('--density', type=int, default=0.25, help='If gen_other, used for density of "other" tensor')
parser.add_argument('-cast', '--cast', action='store_true', default=False, help='Safe sparsity cast to int for values')

args = parser.parse_args()
if args.other:
    if args.suitesparse:
        outdir_name = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    else:
        outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    taco_format_dirname = os.getenv('TACO_TENSOR_PATH')
    if taco_format_dirname is None:
        print("Please set the TACO_TENSOR_PATH environment variable")
        exit()
    taco_format_dirname = os.path.join(taco_format_dirname, "other-formatted-taco")
else:
    outdir_name = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))
    taco_format_dirname = os.getenv('FROSTT_FORMATTED_TACO_PATH')
    if taco_format_dirname is None:
        print("Please set the FROSTT_FORMATTED_TACO_PATH environment variable")
        exit()

out_path = Path(outdir_name)
out_path.mkdir(parents=True, exist_ok=True)

formatWriter = FormatWriter(args.cast)

if args.name is None:
    print("Please enter a tensor name")
    exit()

#breakpoint()
if args.format is not None:
    assert args.format in formats
    levels = args.format[:-3]

    if os.path.exists('sam/FROST_FORMATTED/rand_tensor*'):
        shutil.rmtree('sam/FROST_FORMATTED/rand_tensor*')
    
    if args.bench != "tensor3_elemadd" and args.bench != "tensor3_innerprod":
        assert args.bench is not None
        #$FROSTT_FORMATTED_TACO_PATH
        taco_format_orig_filename = "/home/avb03/sam/FROST_FORMATTED_TACO"
        outdir_other_name = os.path.join(outdir_name, args.name, args.bench)
        # outdir_other_name = os.path.join(outdir_name, args.name, 'other', otherfile[:-4])
        outdir_orig_path = Path(outdir_other_name)
        outdir_orig_path.mkdir(parents=True, exist_ok=True)

        name = None
        taco_format_orig_filename = os.path.join(taco_format_dirname, args.name + "_" + levels + '.txt')

        inputCache = InputCacheSuiteSparse()

        if args.bench == "tensor3_ttv":
            outdir_orig_name = os.path.join(outdir_name, args.name, args.bench, args.format)
            outdir_orig_path = Path(outdir_orig_name)
            outdir_orig_path.mkdir(parents=True, exist_ok=True)

            taco_format_orig_filename = "/home/avb03/sam/FROST_FORMATTED_TACO/" + args.name + "_" + levels + '.txt'
            parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format)
            #Need this line? formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            file_path_name = os.path.join(outdir_orig_name, "tensor_B_mode_shape")
            file1 = open(file_path_name, 'r')
            shape = [0]*3
            lines = file1.readlines()
            count = 0

            # Strips the newline character
            for line in lines:
                shape[count] = int(line)
                count += 1
            # coo = inputCache.load(tensor, False)
            
            # formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            tensorname = 'c'
            vec = scipy.sparse.random(shape[2], 1, density=args.density, data_rvs=np.ones)
            vec = vec.toarray().flatten()
            tensor_out_path = os.path.join(out_path, args.name, args.bench, args.format)
            formatWriter.writeout_separate_vec(vec, tensor_out_path, tensorname)

            # vec = scipy.sparse.random(shape[2], 1, data_rvs=np.ones)
            # vec = vec.toarray().flatten()
            # formatWriter.writeout_separate_vec(vec, out_path, tensorname)
            #FormatWriter.writeout_separate_vec(vec, out_path, tensorname, tensorname)
            #formatWriter.writeout_separate_sparse_only()
        elif args.bench == "tensor3_ttm":
            outdir_orig_name = os.path.join(outdir_name, args.name, args.bench, args.format)
            outdir_orig_path = Path(outdir_orig_name)
            outdir_orig_path.mkdir(parents=True, exist_ok=True)

            taco_format_orig_filename = "/home/avb03/sam/FROST_FORMATTED_TACO/" + args.name + "_" + levels + '.txt'
            parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format)
            #Need this line? formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            file_path_name = os.path.join(outdir_orig_name, "tensor_B_mode_shape")
            file1 = open(file_path_name, 'r')
            shape = [0]*3
            lines = file1.readlines()
            count = 0

            # Strips the newline character
            for line in lines:
                shape[count] = int(line)
                count += 1
            # coo = inputCache.load(tensor, False)
            dimension_k = random.randint(min(shape), 10)
            dimension_l = shape[2] 
            dimension_j = shape[1]

            # formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            tensorname = 'C'
            matrix = scipy.sparse.random(dimension_k, dimension_l, density=args.density, data_rvs=np.ones).toarray()
            tensor_out_path = os.path.join(out_path, args.name, args.bench, args.format)
            formatWriter.writeout_separate_sparse_only(matrix, tensor_out_path, tensorname)

            # vec = scipy.sparse.random(shape[2], 1, data_rvs=np.ones)
            # vec = vec.toarray().flatten()
            # formatWriter.writeout_separate_vec(vec, out_path, tensorname)
            #FormatWriter.writeout_separate_vec(vec, out_path, tensorname, tensorname)
            #formatWriter.writeout_separate_sparse_only()
        elif args.bench == "tensor3_mttkrp":
            outdir_orig_name = os.path.join(outdir_name, args.name, args.bench, args.format)
            outdir_orig_path = Path(outdir_orig_name)
            outdir_orig_path.mkdir(parents=True, exist_ok=True)

            taco_format_orig_filename = "/home/avb03/sam/FROST_FORMATTED_TACO/" + args.name + "_" + levels + '.txt'
            parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format)
            
            file_path_name = os.path.join(outdir_orig_name, "tensor_B_mode_shape")
            file1 = open(file_path_name, 'r')
            shape = [0]*3
            lines = file1.readlines()
            count = 0

            # Strips the newline character
            for line in lines:
                shape[count] = int(line)
                count += 1
            
            dimension_i = shape[0]
            dimension_k = shape[1]
            dimension_l = shape[2]
            dimension_j = random.randint(min(shape), 10)

            # formatWriter.writeout_separate_sparse_only(coo, dirname, tensorname, format_str="ss10")
            tensorname = 'C'
            matrix = scipy.sparse.random(dimension_j, dimension_k, density=args.density, data_rvs=np.ones).toarray()
            tensor_out_path = os.path.join(out_path, args.name, args.bench, args.format)
            formatWriter.writeout_separate_sparse_only(matrix, tensor_out_path, tensorname)

            tensorname = 'D'
            matrix = scipy.sparse.random(dimension_j, dimension_l, density=args.density, data_rvs=np.ones).toarray()
            tensor_out_path = os.path.join(out_path, args.name, args.bench, args.format)
            formatWriter.writeout_separate_sparse_only(matrix, tensor_out_path, tensorname)
        else:
            raise NotImplementedError

        assert tensorname is not None, "Other tensor name was not set properly and is None"
        # parse_taco_format(taco_format_orig_filename, outdir_other_name, tensorname, args.format, hw_filename=args.hw)

    else:
        #this code is used for: tensor3_elemadd, tensor3_innerprod
        taco_format_orig_filename = os.path.join(taco_format_dirname, args.name + "_" + levels + '.txt')
        taco_format_shift_filename = os.path.join(taco_format_dirname, args.name + '_shift_' + levels + '.txt')

        # Original
        outdir_orig_name = os.path.join(outdir_name, args.name, args.bench, args.format)
        outdir_orig_path = Path(outdir_orig_name)
        outdir_orig_path.mkdir(parents=True, exist_ok=True)

        taco_format_orig_filename = "/home/avb03/sam/FROST_FORMATTED_TACO/" + args.name + "_" + levels + '.txt'
        parse_taco_format(taco_format_orig_filename, outdir_orig_name, 'B', args.format)

        # Shifted
        if args.shift:
            outdir_shift_name = os.path.join(outdir_name, args.name, args.bench, args.format)
            outdir_shift_path = Path(outdir_shift_name)
            outdir_shift_path.mkdir(parents=True, exist_ok=True)

            taco_format_shift_filename = "/home/avb03/sam/FROST_FORMATTED_TACO/" + args.name + "_shift_" + levels + '.txt'
            parse_taco_format(taco_format_shift_filename, outdir_shift_name, 'C', args.format)
