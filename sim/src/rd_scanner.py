from abc import ABC, abstractmethod
from .base import Primitive
    

#################
# Read Scanners
#################


class RdScan(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = 'S'
        self.curr_crd = 'S'

    def set_in_ref(self, in_ref):
        if in_ref != '':
            self.in_ref.append(in_ref)


    def out_ref(self):
        return self.curr_ref

    def out_crd(self):
        return self.curr_crd

"""

:param : 
:param : 
:return:    (out_val, out_addr) 
""" 
class UncompressRdScan(RdScan):
    def __init__(self, dim=0, **kwargs):
        super().__init__(**kwargs)

        self.start_addr = 0
        self.stop_addr = dim

        self.in_ref = []
        self.curr_in_ref = 0

        self.meta_dim = dim

    def update(self):

        # run out of coordinates, move to next input reference
        if self.curr_crd == 'S':
            self.curr_in_ref = self.in_ref.pop(0)
            if self.curr_in_ref == 'D':
                self.curr_crd = 'D'
                self.curr_ref = 'D'
                self.done = True
                return
            elif self.curr_in_ref == 'S':
                self.curr_crd = 'S'
                self.curr_ref = 'S'
            else:
                self.curr_crd = 0
                print(self.curr_in_ref, self.meta_dim)
                self.curr_ref = self.curr_crd + (self.curr_in_ref * self.meta_dim)
        elif self.curr_crd >= self.meta_dim-1:
            self.curr_crd = 'S'
            self.curr_ref = 'S'
        else:
            self.curr_crd += 1
            self.curr_ref = self.curr_crd + self.curr_in_ref * self.meta_dim



class CompressedRdScan(RdScan):
    def __init__(self, crd_arr=[], seg_arr=[], **kwargs):
        super().__init__(**kwargs)

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr

        self.start_addr = 0
        self.stop_addr = 0

        self.in_ref = []
        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = None
        self.curr_crd = None

        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)

    
    def update(self):
        if self.curr_crd == 'D' or self.curr_ref == 'D':
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.curr_crd = ''
            self.curr_ref = ''

        # There exists another input reference at the segment and
        # either at the start of computation or end of fiber
        elif len(self.in_ref) > 0 and (self.end_fiber or (self.curr_crd is None or self.curr_ref is None)):

                if self.curr_crd is None or self.curr_ref is None:
                    assert(self.curr_crd == self.curr_ref)
                self.end_fiber = False

                curr_in_ref = self.in_ref.pop(0)
                if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_slen:
                    raise Exception('Not enough elements in seg array')
                if curr_in_ref == 'S' or curr_in_ref == 'D':
                    self.curr_addr = 0
                    self.stop_addr = 0
                    self.start_addr = 0
                    self.curr_crd = curr_in_ref
                    self.curr_ref = curr_in_ref
                    self.end_fiber = True
                    if curr_in_ref == 'D':
                        self.done = True
                else:
                    self.start_addr = self.seg_arr[curr_in_ref]
                    self.stop_addr = self.seg_arr[curr_in_ref + 1]
                    self.curr_addr = self.start_addr
                    self.curr_crd = self.crd_arr[self.curr_addr]
                    self.curr_ref = self.curr_addr
        # End of fiber, get next input reference
        elif self.curr_addr == self.stop_addr - 1 or self.curr_addr == self.meta_clen - 1:
            self.end_fiber = True
            self.curr_crd = 'S'
            self.curr_ref = 'S'
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
        else:
            # Base case: increment address and reference by 1 and get next coordinate
            self.curr_addr += 1
            self.curr_ref = self.curr_addr
            self.curr_crd = self.crd_arr[self.curr_addr]

        # TODO: Need to add in repeat functionality for ref_count

        if self.debug:
            print("Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                  "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr)
''' 
"""

:param : 
:param : 
:return:    (out_val, out_addr) 
""" 
class WrScan(Primitive):
    @abstractmethod 
    def out_wraddr():
        pass
    
    @abstractmethod
    def out_wrdata():
        pass

    @abstractmethod 
    def out_done():
        pass
def rScan(in_ref, start=0, repeat=0):
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

def Intersect(Primitive):
    def __init__(self, *args):
        self.out_crd = 0
        self.out_ref1 = 0
        self.out_ref2 = 0

        self.upd1 = True
        self.upd2 = True
        self.in1 = []
        self.in2 = []

        self.curr_in1 = 0
        self.curr_in2 = 0
        
    def update(in1, in2)
        if in1 != '':
            self.in1.push(in1)
        if in2 != '':
            self.in2.push(in2)

        if self.upd1:
            self.curr_in1 = self.in1.pop(0)
        if self.upd2:
            self.curr_in2 = self.in2.pop(0)

        if self.curr_in1 == 'S' and self.curr_in2 == 'S':
            self.upd1 = True
            self.upd2 = True
        elif self.curr_in1 == 'S':
            self.upd1 = False
            self.upd2 = True
        elif self.curr_in2 == 'S':
            self.upd1 = False
            self.upd2 = True
        else:
            self.upd1 = self.curr_in1 < self.curr_in2
            self.upd2 = self.curr_in2 < self.curr_in1

    def out_crd(self):
        return self.out_crd

    def 
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
    
'''

#################
# Helper Functions
#################

# def nestLst(slist, tkn):
#     result = []
#     tmp = []
#     tkn_found = False
#     for elem in slist:
#         if tkn == elem or (isinstance(elem, list) and tkn in elem):
#             tkn_found = True
#         elif tkn_found and isinstance(elem, int):
#             result.append(tmp)
#             tmp = []
#             tkn_found = False
#             tmp.append(elem)
#         else:
#             tmp.append(elem)
#
#     result.append(tmp)
#
#     return result
#
# """
# :param slist: hierarchical stream
# :return: stream as a multi-dim list
# """
# def convertStream(slist, order=1):
#     ltkn = ["s"+str(x) for x in range(order)]
#     for tkn in ltkn:
#         result = nestLst(slist, tkn)
#         slist = result
#
#     return result
#
