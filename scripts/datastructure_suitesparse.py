import argparse
import os
import dataclasses 
import numpy 

from pathlib import Path 

from util import TensorCollectionSuiteSparse, ScipyTensorShifter, PydataMatrixMarketTensorLoader, ScipyMatrixMarketTensorLoader, SuiteSparseTensor, safeCastPydataTensorToInts

#SS_PATH = os.getenv('SUITESPARSE_PATH')
SS_PATH = Path("~/aha-sparsity/")

formats = ["coo", "csr", "dcsr", "dcsc", "csc", "dense"]

# UfuncInputCache attempts to avoid reading the same tensor from disk multiple
# times in a benchmark run.
class UfuncInputCache:
    def __init__(self):
        self.lastLoaded = None
        self.lastName = None
        self.tensor = None

    def load(self, tensor, suiteSparse, cast, format_str):
        if self.lastName == str(tensor):
            return self.tensor 
        else:
            if suiteSparse:
                self.lastLoaded = tensor.load(ScipyMatrixMarketTensorLoader(format_str))
            else:
                self.lastLoaded = tensor.load()
            self.lastName  = str(tensor)
            if cast:
              self.tensor = safeCastPydataTensorToInts(self.lastLoaded)
            else:
              self.tensor = self.lastLoaded
            return self.tensor

    def writeout(self, tensor, suiteSparse, cast, format_str, filename):
        with open(filename, "w") as outfile: 
            tensor = self.load(tensor, suiteSparse, cast, format_str)
            match format_str:
                case "csr":
                    outfile.write(tensor.shape)
                    outfile.write(tensor.indptr)
                    outfile.write(tensor.indices)
                    outfile.write(tensor.values)

 
inputCache = UfuncInputCache()

parser = argparse.ArgumentParser(description="Process some suitesparse matrices into per-level datastructures")
parser.add_argument('-n', '--name', metavar='ssname', type=str, action='store', help='tensor name to run tile analysis on one SS tensor')
parser.add_argument('-f', '--format', metavar='ssformat', type=str, action='store', default='format', help='The format that the tensor should be converted to')

args = parser.parse_args()

out_dirname = './mode_formats/'
out_path = Path(out_dirname)  
out_path.mkdir(parents=True, exist_ok=True)

if args.name is None: 
    print("Please enter a matrix name")
    exit()

tensor = SuiteSparseTensor(os.path.join(SS_PATH, args.name))
for format_str in formats:
    filename = os.join(out_path, args.name+format_str + ".txt") 
    inputCache.writeout(tensor, True, True, format_str, filename)
    


