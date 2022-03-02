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
        if self.curr_crd == '' or self.curr_crd == 'D':
            self.curr_crd = ''
            self.curr_ref = ''
        elif self.curr_crd == 'S':
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
                self.curr_ref = self.curr_crd + (self.curr_in_ref * self.meta_dim)
        elif self.curr_crd >= self.meta_dim-1:
            self.curr_crd = 'S'
            self.curr_ref = 'S'
        else:
            self.curr_crd += 1
            self.curr_ref = self.curr_crd + self.curr_in_ref * self.meta_dim

        if self.debug:
            print("DEBUG: U RD SCAN: \t "
                  "Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref)


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

        if self.debug:
            print("DEBUG: C RD SCAN: \t "
                  "Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                  "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr)
