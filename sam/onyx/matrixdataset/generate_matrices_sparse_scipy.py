import argparse
import scipy
import scipy.io
import scipy.sparse
import os
import numpy as np

def generate_sparsities(sparsities, dimension, name, path='.', seed=100, rvs=np.ones):
    for i in range(len(sparsities)):
        sparsity = sparsities[i]
        matrix = scipy.sparse.random(dimension[0], dimension[1], density=sparsity, format="coo", random_state=seed, data_rvs=rvs)
        filename = name + "-" + "sp" + str(sparsity).replace(".", "_") + "-" + str(i) + ".mtx"
        filepath = os.path.join(path, filename)
        scipy.io.mmwrite(filepath, matrix)
   
def generate_dimensions(dimensions, sparsity, name, path='.', seed=100, rvs=np.ones): 
    for i in range(len(dimensions)):
        dimension = dimensions[i]
        matrix = scipy.sparse.random(dimension[0], dimension[1], density=sparsity, format="coo", random_state=seed, data_rvs=rvs)
        filename = name + "-" + "dim" + str(dimension[0]) + "_" + str(dimension[1]) + "-" + str(i) + ".mtx"
        filepath = os.path.join(path, filename)
        scipy.io.mmwrite(filepath, matrix)

## Used to generate spmv/gemv crossover point 
# sparsities = [0.00078125, 0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1]
# ss_path = os.environ.get('SUITESPARSE_PATH', './')
# generate_sparsities(sparsities, (1024, 1024), 'mat_vecmul_sweep', path=ss_path)

# Used to generate mask for mask_tri

