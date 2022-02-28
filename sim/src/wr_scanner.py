from abc import ABC, abstractmethod
from .base import Primitive, valid_tkns
from sim.src.array import Array


class WrScan(Primitive, ABC):
    def __init__(self, size=1024, fill=0, **kwargs):
        super().__init__(**kwargs)
        self.size = size
        self.fill = fill

        self.input = []

        self.arr = Array(size=size, fill=fill, debug=self.debug)

    def set_input(self, val):
        # Make sure streams have correct token type
        assert(isinstance(val, int) or val in valid_tkns)

        if val != '':
            self.input.append(val)

    def clear_arr(self):
        self.arr.clear(self.fill)

    def resize_arr(self, size):
        self.size = size
        self.arr.resize(self.size)

    def get_arr(self):
        return self.arr.get_arr()

    @abstractmethod
    def reset(self):
        pass


class ValsWrScan(WrScan):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_addr = 0

    def update(self):
        if (len(self.input) > 0):
            val = self.input.pop(0)

            if val != 'S' and val != 'D':
                self.arr.set_store(self.curr_addr, val)
                self.curr_addr += 1
            else:
                self.arr.set_store(val, val)

            self.arr.update()
            self.done = self.arr.out_done()

    def reset(self):
        self.curr_addr = 0



# Unique compressed (not from points)
class CompressWrScan(WrScan):
    def __init__(self, seg_size=0, level=0, **kwargs):
        super().__init__(**kwargs)
        # FIXME: Either use this later or remove
        self.level = level

        self.curr_addr = 0
        self.curr_seg_addr = 1
        self.curr_crd_cnt = 0

        self.end_fiber = False

        self.seg_size = seg_size
        print("Seg Size: ", self.seg_size)
        self.seg_arr = Array(size=self.seg_size, fill=0, debug=self.debug)

    def update(self):
        if len(self.input) > 0:
            in_crd = self.input.pop(0)

            if in_crd != 'S' and in_crd != 'D':
                self.arr.set_store(self.curr_addr, in_crd)
                self.curr_addr += 1
                self.curr_crd_cnt += 1
                self.end_fiber = False
            elif in_crd == 'S' and not self.end_fiber:
                self.seg_arr.set_store(self.curr_seg_addr, self.curr_crd_cnt)
                self.curr_seg_addr += 1
                self.end_fiber = True
                self.arr.set_store(in_crd, in_crd)
            else:
                self.arr.set_store(in_crd, in_crd)

            self.arr.update()
            self.seg_arr.update()
            self.done = self.arr.out_done()

        if self.debug:
            print("DEBUG: WR SCAN: \t "
                  "Curr crd addr:", self.curr_addr, "\t curr crd cnt:", self.curr_crd_cnt, "\t curr seg addr:", self.curr_seg_addr,
                  "\t end fiber:", self.end_fiber)

    def reset(self):
        self.curr_addr = 0
        self.curr_seg_addr = 0

    def clear_seg_arr(self):
        self.seg_arr.clear(self.fill)

    def resize_seg_arr(self, size):
        self.seg_size = size
        self.seg_arr.resize(self.seg_size)

    def get_seg_arr(self):
        return self.seg_arr.get_arr()
