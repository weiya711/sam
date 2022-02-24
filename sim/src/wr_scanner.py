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


class CompressWrScan(WrScan):
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

