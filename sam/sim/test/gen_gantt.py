import matplotlib.pyplot as plt


def gen_gantt(extra_info, testname):
    block_list = []
    start_list = []
    duration_list = []

    start_c = ''
    finish_c = ''
    sam_name = ''

    for k in extra_info.keys():
        if "done_cycles" in k:
            sam_name = k.split('/')[0]
            finish_c = extra_info[k]
        elif ("start_cycle" in k) and (sam_name in k.split('/')[0]):
            start_c = extra_info[k]
            # We assume that the info to extra_info is added in the same order
            # each block is updated
            block_list.insert(0, sam_name)
            # block_list.append(sam_name)
            if not isinstance(start_c, int):
                # start_list.append(int(start_c))
                # duration_list.append(finish_c - int(start_c))
                start_list.insert(0, int(start_c))
                duration_list.insert(0, finish_c - int(start_c))
            else:
                # start_list.append(start_c)
                # duration_list.append(finish_c - start_c)
                start_list.insert(0, start_c)
                duration_list.insert(0, finish_c - start_c)

    plt.barh(y=block_list, width=duration_list, left=start_list)
    
    file_name = testname + '_' + extra_info["dataset"] + ".png"
    plt.savefig(file_name, bbox_inches="tight")
