import argparse
import os
import dataclasses 
import numpy 

from pathlib import Path 

from util import TensorCollectionSuiteSparse, ScipyTensorShifter, PydataMatrixMarketTensorLoader, ScipyMatrixMarketTensorLoader, SuiteSparseTensor, safeCastPydataTensorToInts, InputCacheSuiteSparse

SS_PATH = os.getenv('SUITESPARSE_PATH')
#SS_PATH = Path("~/aha-sparsity/")

formats = ["coo", "cooT", "csr", "dcsr", "dcsc", "csc", "dense", "denseT"]
scipy_formats = ["coo", "csr", "csc"]

def get_datastructure_string(format, mode):
    if format == ['d','d'] and mode == [0, 1]:
        return "dense"
    elif format == ['d','d']:
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


# UfuncInputCache attempts to avoid reading the same tensor from disk multiple
# times in a benchmark run.
class InputCacheTensor:
    def __init__(self):
        self.lastLoaded = None
        self.lastName = None
        self.tensor = None

    def load(self, tensor, suiteSparse, cast, format_str):
        if self.lastName == str(tensor):
            return self.tensor
        else:
            self.lastLoaded = tensor.load()
            self.lastName = str(tensor)
            if cast:
                self.tensor = safeCastPydataTensorToInts(self.lastLoaded)
            else:
                self.tensor = self.lastLoaded
            return self.tensor
 
inputCache = InputCacheSuiteSparse()

parser = argparse.ArgumentParser(description="Process some suitesparse matrices into per-level datastructures")
parser.add_argument('-n', '--name', metavar='ssname', type=str, action='store', help='tensor name to run format conversion on one SS tensor')
parser.add_argument('-f', '--format', metavar='ssformat', type=str, action='store', help='The format that the tensor should be converted to')

args = parser.parse_args()

out_dirname = './mode_formats/'
out_path = Path(out_dirname)  
out_path.mkdir(parents=True, exist_ok=True)

if args.name is None: 
    print("Please enter a matrix name")
    exit()

tensor = SuiteSparseTensor(os.path.join(SS_PATH, args.name))
if args.format is not None:
    assert(args.format in formats)
    filename = os.path.join(out_path, args.name+"_"+args.format + ".txt")
    inputCache.writeout(tensor, False, args.format, filename)
else:
    for format_str in formats:
        filename = os.path.join(out_path, args.name+"_"+format_str + ".txt")
        print("Writing " + args.name + " " + format_str + "...")
        inputCache.writeout(tensor, False, format_str, filename)


