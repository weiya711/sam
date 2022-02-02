import numpy as np

#################
# Helper Functions
#################

def nestLst(slist, tkn):
    result = []
    tmp = []
    tkn_found = False
    for elem in slist: 
        if tkn == elem or (isinstance(elem, list) and tkn in elem):
            tkn_found = True
        elif tkn_found and isinstance(elem, int):
            result.append(tmp)
            tmp = []
            tkn_found = False
            tmp.append(elem)
        else:
            tmp.append(elem)

    result.append(tmp)

    return result
 
"""
:param slist: hierarchical stream 
:return: stream as a multi-dim list 
""" 
def convertStream(slist, order=1):
    ltkn = ["s"+str(x) for x in range(order)]
    for tkn in ltkn:
        result = nestLst(slist, tkn)
        slist = result

    return result 
    
    
    

#################
# Primitives
#################

"""

:param : 
:param : 
:return:    (out_val, out_addr) 
""" 
def rdScan(in_ref, start=0, stop=16, repeat=0, crd_only=False):
    
    pass


"""

:param : 
:param : 
:return:    (out_val, out_addr) 
""" 
def wrScan(in_ref, start=0, repeat=0):
    pass


"""
intersects two (crd, pos) elements

:param in1: (crd, pos) tuple for input stream
:param in2: (crd, pos) tuple for input stream
:return:    (upd1, upd2, crd, pos1, pos2) tuple for output stream where
            upd1, upd2 = whether to update input streams, respectively
            crd = output coordinate
            pos1, pos2 = corresponding input positions, respectively
""" 
def intersect(in1, in2):
    in1_crd = in1[0]
    in1_pos = in1[1]
    in2_crd = in2[0]
    in2_pos = in2[1]

    if in1_crd == in2_crd:
        return (1, 1, in1_crd, in1_pos, in2_pos)
    elif in1_crd < in2_crd:
        # increment in1 stream
        (1, 0, 'X', 'X', 'X')
    else:
        # increment in2 stream
        (0, 1, 'X', 'X', 'X')

"""
intersects two (crd, pos) elements

:param in1: (crd, pos) tuple for input stream
:param in2: (crd, pos) tuple for input stream
:return:    (upd1, upd2, crd, pos1, pos2) tuple for output stream where
            upd1, upd2 = whether to update input streams, respectively
            crd = output coordinate
            pos1, pos2 = corresponding input positions, respectively
""" 
def union(in1, in2):
    pass

def mul(in1, in2):
    return in1 * in2

def add(in1, in2): 
    return in1 + in2

# FIXME
def reduce(in1, lvl=0):
    red = []
    tkn = 's'+str(lvl)
    return np.sum(in1[0:tkn])

def valArr(addrs, vals):
    return vals[addrs]
