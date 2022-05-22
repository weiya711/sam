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


class BVDrop(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outer_bv = []
        self.inner_bv = []

        self.curr_obv = ''
        self.curr_bv = ''
        self.has_bv = False
        self.get_next_ibv = False
        self.get_next_obv = True

    def update(self):
        ibv = ""
        if self.debug:
            print("Outer BV:", self.outer_bv)
            print("Inner BV:", self.inner_bv)

        if self.done:
            self.curr_bv = ''
            return

        if len(self.outer_bv) > 0 and self.get_next_obv:
            self.curr_obv = self.outer_bv.pop(0)
            if isinstance(self.curr_obv, int):
                self.get_next_ibv = True
                self.get_next_obv = False
            else:
                self.curr_bv = self.curr_obv
                self.get_next_ibv = False
                self.get_next_obv = True
                if self.curr_obv == 'D':
                    self.done = True
            self.has_bv = False
        elif self.get_next_obv:
            self.curr_bv = ''

        if len(self.inner_bv) > 0 and self.get_next_ibv:
            ibv = self.inner_bv.pop(0)
            if isinstance(ibv, int):
                self.has_bv = True
                self.curr_bv = ''
                self.get_next_obv = False
                self.get_next_ibv = True
            elif is_stkn(ibv) and is_stkn(self.curr_obv):
                self.get_next_obv = True
                self.curr_bv = self.curr_obv
                self.get_next_ibv = False
            elif is_stkn(ibv):
                self.get_next_obv = True
                self.curr_bv = self.curr_obv if self.has_bv else ''
                self.get_next_ibv = False
            elif self.done:
                assert (ibv == 'D')
                self.curr_bv = 'D'
                self.get_next_ibv = False
                self.get_next_obv = False
            else:
                self.curr_bv = ''
                self.get_next_ibv = False
                self.get_next_obv = True
        elif self.get_next_ibv:
            self.curr_bv = ''

        if self.debug:
            print("DEBUG: bvDROP: Curr Outerbv:", self.curr_obv, "\tCurr Innerbv:", ibv,
                  "\t Curr Outputbv:", self.curr_bv, "\tHasbv", self.has_bv,
                  "\t GetNext Innerbv:", self.get_next_ibv, "\t GetNext Outerbv:", self.get_next_obv)

    def set_outer_bv(self, bv):
        if bv != '':
            self.outer_bv.append(bv)

    def set_inner_bv(self, bv):
        if bv != '':
            self.inner_bv.append(bv)

    def out_bv_outer(self):
        return self.curr_bv
