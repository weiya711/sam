import scipy.io
import scipy.sparse
import numpy as np

def generate_mat(nnz, dim):
    return scipy.sparse.random(dim, dim, nnz / (dim**2), data_rvs=np.ones)

def write_mtx(path, t):
    scipy.io.mmwrite(path, t)

if __name__ == "__main__":
   dims = list(range(1024, 16390, 668)) 
   nnzs = [5000, 10000, 25000, 50000]

   for nnz in nnzs:
       for dim in dims:
           print("nnz:", nnz, "dim", dim)
           tensor = generate_mat(nnz, dim)
           write_mtx("extensor_" + str(nnz) + "_" + str(dim) + ".mtx", tensor)

    
