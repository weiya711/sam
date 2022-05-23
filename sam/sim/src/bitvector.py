from abc import ABC

from .base import *
from .token import EmptyFiberStknDrop

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


class BVDropSuper(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outer_bv = []
        self.inner_bv = []

        self.curr_obv = ''

    def set_outer_bv(self, bv):
        if bv != '':
            self.outer_bv.append(bv)

    def set_inner_bv(self, bv):
        if bv != '':
            self.inner_bv.append(bv)

    def out_bv_outer(self):
        return self.curr_obv


class BVDropOnly(BVDropSuper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.get_ibv_count = 0

        self.running_obv = ''
        self.orig_obv = ''

        # FIXME: finish doing this
        self.curr_ibv = ''

        self.has_bv = False
        self.get_next_ibv = False
        self.get_next_obv = True

    def update(self):
        ibv = ""
        if self.debug:
            print("Outer BV:", self.outer_bv)
            print("Inner BV:", self.inner_bv)

        if self.done:
            self.curr_obv = ''
            return

        if len(self.outer_bv) > 0 and self.get_next_obv:
            obv = self.outer_bv.pop(0)
            if isinstance(obv, int):
                self.running_obv = obv
                self.orig_obv = self.running_obv
                self.get_ibv_count = popcount(self.running_obv)
                self.get_next_ibv = self.get_ibv_count > 0
                self.get_next_obv = False
            else:
                self.running_obv = obv
                self.curr_obv = obv
                self.get_next_ibv = False
                self.get_next_obv = True
                if obv == 'D':
                    self.done = True
                    self.get_next_ibv = True
                    self.curr_obv = 'D'
            self.has_bv = False
        elif self.get_next_obv:
            self.curr_obv = ''

        if self.debug:
            print("\t before getting ibv: GetNext InnerBV:", self.get_next_ibv)

        if len(self.inner_bv) > 0 and self.get_next_ibv:
            ibv = self.inner_bv.pop(0)
            if isinstance(ibv, int):
                self.has_bv = True
                self.curr_obv = ''
                self.get_next_obv = False
                self.get_next_ibv = True
            elif is_stkn(ibv) and is_stkn(self.running_obv):
                self.get_next_obv = True
                self.curr_obv = self.running_obv
                self.get_next_ibv = False
            elif is_stkn(ibv) and self.get_ibv_count and self.has_bv:
                self.get_ibv_count -= 1
                self.get_next_obv = self.get_ibv_count == 0
                self.curr_obv = self.running_obv if self.get_ibv_count == 0 else ''
                self.get_next_ibv = self.get_ibv_count > 0
                self.has_bv = False
            elif is_stkn(ibv) and self.get_ibv_count:
                self.get_ibv_count -= 1
                bitn = popcount(self.orig_obv) - self.get_ibv_count - 1
                print(bitn)
                self.running_obv &= ~get_nth_bit(self.orig_obv, bitn)
                self.get_next_obv = self.get_ibv_count == 0
                self.curr_obv = self.running_obv if self.get_ibv_count == 0 and self.running_obv != 0 else ''
                self.get_next_ibv = self.get_ibv_count > 0
                self.has_bv = False
            elif self.done:
                assert (ibv == 'D')
                self.curr_obv = 'D'
                self.get_next_ibv = False
                self.get_next_obv = False
            else:
                self.curr_obv = ''
                self.get_next_ibv = False
                self.get_next_obv = True
        elif self.get_next_ibv:
            self.curr_obv = ''

        if self.debug:
            print("DEBUG: BVDROP: Curr OuterBV:", self.running_obv, "Orig OuterBV:", self.orig_obv,
                  "\n Curr Output BV:", self.curr_obv, "\t GetNext OuterBV:", self.get_next_obv,
                  "\n HasBV", self.has_bv, "\t BitCount", self.get_ibv_count, "\t GetNext InnerBV:", self.get_next_ibv)


class BVDrop(BVDropSuper):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.bv_drop = BVDropOnly(**kwargs)
        self.outer_stkn_drop = EmptyFiberStknDrop(**kwargs)
        self.inner_stkn_drop = EmptyFiberStknDrop(**kwargs)

        self.curr_ibv = ''

    def update(self):
        if len(self.outer_bv) > 0:
            self.bv_drop.set_outer_bv(self.outer_bv.pop(0))

        if len(self.inner_bv) > 0:
            ibv = self.inner_bv.pop(0)
            self.bv_drop.set_inner_bv(ibv)
            self.inner_stkn_drop.set_in_stream(ibv)

        self.bv_drop.update()

        self.outer_stkn_drop.set_in_stream(self.bv_drop.out_bv_outer())
        self.outer_stkn_drop.update()

        self.inner_stkn_drop.update()

        self.curr_obv = self.outer_stkn_drop.out_val()
        self.curr_ibv = self.inner_stkn_drop.out_val()

        self.done = self.outer_stkn_drop.out_done() and self.inner_stkn_drop.out_done() and self.bv_drop.out_done()

    def out_bv_inner(self):
        return self.curr_ibv
