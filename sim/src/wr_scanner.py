from abc import ABC, abstractmethod
from .base import Primitive, valid_tkns
from sim.src.array import Array


class WrScan(Primitive):
    def __init__(self, size=1024, fill=0, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.fill = fill

        self.vals = []

        self.arr = Array(size=size, fill=fill)

    def set_val(self, val):
        # Make sure streams have correct token type
        assert(isinstance(val, int) or val in valid_tkns)

        if val != '':
            self.vals.append(val)

class UncompressWrScan(WrScan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_addr = 0

    def update(self):
        if (len(self.vals) > 0):
            val = self.vals.pop(0)

            if val != 'S' and val != 'D':
                self.arr.set_store(self.curr_addr, val)
                self.curr_addr += 1
            else:
                self.arr.set_store(val, val)

            self.arr.update()
            self.done = self.arr.out_done()

    def reset(self):
        self.curr_addr = 0

    def clear_arr(self):
        self.arr.clear(self.fill)

    def resize_arr(self, size):
        self.size = size
        self.arr.resize(self.size)

    def get_arr(self):
        return self.arr.get_arr()


'''
class CompressedRdScan(RdScan):
    def __init__(self, crd_arr=[], seg_arr=[], **kwargs):
        super().__init__(self, **kwargs)

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr

        self.start_addr = 0
        self.stop_addr = 0

        self.in_ref = []
        self.curr_addr = 0
        self.curr_ref = 0
        self.curr_crd = 0
        self.done = False

        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)

    
    def update(self):
        # End of segment, get next input reference
        if self.curr_addr == self.stop_addr:
            # There exists another input reference at the segment
            if len(self.in_ref) > 0:
                curr_in_ref = self.in_ref.pop(0)
                if (curr_in_ref + 1) > self.meta_slen:
                    raise Exception('Not enough elements in seg array')
                if curr_in_ref == 'S':
                    self.curr_addr = 0
                    self.stop_addr = 0
                    self.start_addr = 0
                    self.curr_crd = 'S'
                    self.curr_ref = 'S'
                else:
                    self.start_addr = self.seg_arr[curr_in_ref]
                    self.stop_addr = self.seg_arr[curr_in_ref+1]
                    self.curr_addr = self.start_addr
            # There does not exist another input reference at the segment
            else:
                self.done = True
                self.curr_crd = ''
                self.curr_ref = ''
        # There are no more coordinates
        elif self.curr_addr == self.meta_clen:
            self.curr_crd = 'S'
            self.curr_ref = 'S'
        # Base case: increment address and reference by 1 and get next coordinate
        else:
            self.curr_crd = self.crd_arr[self.curr_addr]
            self.curr_ref += 1
            self.curr_addr += 1

    def set_in_ref(self, in_ref):
        if in_ref != '':
            self.in_ref.append(in_ref)

    def out_crd(self):
        return self.curr_crd
    
    def out_ref(self):
        return self.curr_ref
        '''

