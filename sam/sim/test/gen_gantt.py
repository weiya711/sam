import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

color_dict = {'fiberlookup': 'green', 'intersect': 'blueviolet', 'array':''}

def gen_gantt(extra_info,testname):
    block_list = []
    start_list = []
    duration_list = []
    
    start_c=''
    finish_c=''
    sam_name=''
    
    for k in extra_info.keys():
        if "done_cycles" in k:
            sam_name=k.split('/')[0]
            finish_c=extra_info[k]
        elif ("start_cycle" in k) and (sam_name in k.split('/')[0]):
            start_c=extra_info[k]
            if isinstance(start_c, int):                
                block_list.append(sam_name)
                start_list.append(start_c)
                duration_list.append(finish_c-start_c)

    plt.barh(y=block_list, width=duration_list, left=start_list)
    
    file_name = testname + '_' + extra_info["dataset"] + ".png"
    plt.savefig(file_name,bbox_inches = "tight")
