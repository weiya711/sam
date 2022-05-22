from .base import *
from functools import reduce


class BV(Primitive):
    def __init__(self, width=4, **kwargs):
        super().__init__(**kwargs)

        # TODO: see if we need this
        # may not need this if we always compress the ENTIRE fiber...hmmm
        self.bv_width = width

        self.in_crd = []
        self.crds = []
        self.curr_bv = None

        self.stkn = None
        self.emit_stkn = False

    def update(self):
        # bin(bv)   # gives 0bx number
        if self.done:
            self.curr_bv = ''
        elif self.emit_stkn:
            self.curr_bv = self.stkn
            self.emit_stkn = False
        elif len(self.in_crd) > 0:
            in_crd = self.in_crd.pop(0)
            if isinstance(in_crd, int):
                self.curr_bv = ''
                self.crds.append(in_crd)
            elif is_stkn(in_crd):
                self.curr_bv = reduce(lambda a, b: a | b, [0b1 << c for c in self.crds])
                self.emit_stkn = True
                self.stkn = in_crd
                self.crds = []
            else:
                self.curr_bv = 'D'
                self.done = True
        else:
            self.curr_bv = ''

    def set_in_crd(self, crd):
        if crd != '':
            self.in_crd.append(crd)

    def out_bv_int(self):
        return self.curr_bv

    def out_bv(self):
        result = bin(self.curr_bv) if isinstance(self.curr_bv, int) else self.curr_bv
        return result
