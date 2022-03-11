import pandas as pd
import matplotlib.pyplot as plt
import tqdm 
import argparse
import os
import dataclasses 
import numpy 

from pathlib import Path 

from util import TensorCollectionSuiteSparse, ScipyTensorShifter, PydataMatrixMarketTensorLoader, ScipyMatrixMarketTensorLoader, SuiteSparseTensor, safeCastPydataTensorToInts

SS_PATH = os.getenv('SUITESPARSE_PATH')

# UfuncInputCache attempts to avoid reading the same tensor from disk multiple
# times in a benchmark run.
class UfuncInputCache:
    def __init__(self):
        self.lastLoaded = None
        self.lastName = None
        self.tensor = None

    def load(self, tensor, suiteSparse, cast):
        if self.lastName == str(tensor):
            return self.tensor, self.other
        else:
            if suiteSparse:
                self.lastLoaded = tensor.load(ScipyMatrixMarketTensorLoader("coo"))
            else:
                self.lastLoaded = tensor.load()
            self.lastName  = str(tensor)
            if cast:
              self.tensor = safeCastPydataTensorToInts(self.lastLoaded)
            else:
              self.tensor = self.lastLoaded
            return self.tensor
inputCache = UfuncInputCache()


#def save_images(folder, name, imgs, grid_size=2):
#    f, axarr = plt.subplots(grid_size, grid_size, figsize=(15,15))
#    for i in range(grid_size):
#        for j in range(grid_size):
#            axarr[i,j].imshow(imgs[i*grid_size + j]
#    name_path = os.path.join(folder, name)
#    plt.savefig(name_path)

def main():
    parser = argparse.ArgumentParser(description="Process some suitesparse matrix statistics")
    parser.add_argument('-n', '--num', metavar='N', type=int, action='store')
    
    args = parser.parse_args()

    full_out_filepath = Path('./logs/out.csv')  
    full_out_filepath.parent.mkdir(parents=True, exist_ok=True)

    ssMtx = dataclasses.make_dataclass("SS", [("name", str), ("nnz", int), ("dim1", int), ("dim2", int), ("sparsity", float)])
    tensor_list = []
    filenames = [f for f in os.listdir(SS_PATH) if f.endswith(".mtx")][:args.num]
    pbar = tqdm.tqdm(filenames)
    for filename in pbar:
        pbar.set_description("Processing %s" % filename)
        tensor = SuiteSparseTensor(os.path.join(SS_PATH, filename))
        ssTensor = inputCache.load(tensor, True, False)
        if not isinstance(ssTensor, numpy.ndarray):
          tensor_list.append(ssMtx(str(tensor), ssTensor.nnz, ssTensor.shape[0], ssTensor.shape[1], float(ssTensor.nnz)/float(ssTensor.size)))

    dfss = pd.DataFrame(tensor_list)
    
    hist = dfss.hist(bins=100)
    plt.savefig('./logs/figure.pdf')

    dfss.to_csv(full_out_filepath)
    


    

  

  
if __name__=='__main__':
  main()


