import os
import scipy
import glob
import numpy as np

orig_B = scipy.io.mmread('extensor_mtx/rel5.mtx')
orig_C = scipy.io.mmread('tiles/C.mtx')
tot = orig_B + orig_C
tot = tot.todense()
tot_x, tot_y = tot.shape
print(tot_x, tot_y)

x_tsize = 40
y_tsize = 30

numtiles = (tot_x // x_tsize) * (tot_y // y_tsize)

accum = np.zeros((340,80))

B_tensors = glob.glob('tiles/mtx/tensor_B_*.mtx')
C_tensors = glob.glob('tiles/mtx/tensor_C_*.mtx')
paired = {}

for b in B_tensors:
    for c in C_tensors:
        b_loc = b.split("_")
        c_loc = c.split("_")
        
        b_loc = b_loc[3:]
        b_loc[-1] = b_loc[-1][:-4]
        c_loc = c_loc[3:]
        c_loc[-1] = c_loc[-1][:-4]
        # breakpoint()
        
        if b_loc == c_loc:
            tileB = scipy.io.mmread(b)
            tileC = scipy.io.mmread(c)
            
            tile = tileB + tileC
            paired["".join(b_loc)] = tile
            
print(paired)


for k,v in paired.items():
    id_nums = list(map(int, k))    
    # mat = v
    # mat = v.reshape(accum[id_nums[0] * x_tsize:id_nums[0] * x_tsize + x_tsize, id_nums[3] * y_tsize:id_nums[3] * y_tsize + y_tsize].shape)  
    # mat = mat.todense()
    accum[id_nums[0] * x_tsize:id_nums[0] * x_tsize + x_tsize, id_nums[3] * y_tsize:id_nums[3] * y_tsize + y_tsize] = v.todense()
    

np.set_printoptions(threshold=np.inf)   
accum = accum[:,0:35] 
print(np.allclose(accum, tot))