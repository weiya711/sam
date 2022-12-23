from .base import *
from .repeater import RepeatSigGen, Repeat


class CrdMask(Primitive):
    def __init__(self, drop_predicate=lambda *crds: False, **kwargs):
        # Will drop a coordinate if drop_predicate returns True
        # It takes in some number of current coordinates and returns True/False to drop/not drop
        super().__init__(**kwargs)

        self.outer_crd = []
        self.inner_crd = []
        self.curr_inner_crd = ''
        self.curr_ocrd = ''
        self.curr_crd = ''
        self.has_crd = False
        self.prev_ocrd_stkn = True
        self.get_stkn = False
        self.get_next_icrd = False
        self.get_next_ocrd = True

        self.drop_predicate = drop_predicate

        # statistics info
        if self.get_stats:
            self.inner_crd_fifo = 0
            self.outer_crd_fifo = 0
            self.ocrd_drop_cnt = 0

    def update(self):
        self.update_done()
        if len(self.outer_crd) > 0 or len(self.inner_crd) > 0:
            self.block_start = False

        icrd = ""
        if self.debug:
            print("OuterCrds:", self.outer_crd)
            print("InnerCrds:", self.inner_crd)

        if self.done:
            self.curr_crd = ''
            return

        if len(self.outer_crd) > 0 and self.get_next_ocrd:
            if self.get_stats:
                self.outer_crd_fifo = max(self.outer_crd_fifo, len(self.outer_crd))
            self.curr_ocrd = self.outer_crd.pop(0)
            if isinstance(self.curr_ocrd, int):
                self.getclass LowerTriangular2D(CrdMasker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, lambda *crds: crds[0] >= crds[1])
                self.curr_crd = self.curr_ocrd

                if self.prev_ocrd_stkn:
                    self.get_next_icrd = True
                    self.get_next_ocrd = False
                    self.get_stkn = True
                else:
                    self.get_next_icrd = False
                    self.get_next_ocrd = True
                    self.get_stkn = False

                if self.curr_ocrd == 'D':
                    self.done = True
                self.prev_ocrd_stkn = True

            self.has_crd = False
        elif self.get_next_ocrd:
            self.curr_crd = ''
            if self.get_stats:
                self.ocrd_drop_cnt += 1

        if len(self.inner_crd) > 0 and self.get_next_icrd:
            if self.get_stats:
                self.inner_crd_fifo = max(self.inner_crd_fifo, len(self.inner_crd))
            icrd = self.inner_crd.pop(0)
            self.curr_inner_crd = icrd
            if self.get_stkn:
                assert is_stkn(icrd) == is_stkn(self.curr_ocrd)
                self.get_next_ocrd = True
                self.get_next_icrd = False
                self.get_stkn = False
            if isinstance(icrd, int):
                self.has_crd = True
                self.curr_crd = ''
                self.get_next_ocrd = False
                self.get_next_icrd = True
                if self.get_stats:
                    self.ocrd_drop_cnt += 1
            elif is_stkn(icrd) and is_stkn(self.curr_ocrd):
                self.get_next_ocrd = True
                self.curr_crd = self.curr_ocrd
                self.get_next_icrd = False
            elif is_stkn(icrd):
                self.get_next_ocrd = True
                self.curr_crd = self.curr_ocrd if self.has_crd else ''
                self.get_next_icrd = False
            elif self.done:
                assert (icrd == 'D')
                self.curr_crd = 'D'
                self.get_next_icrd = False
                self.get_next_ocrd = False
            else:
                self.curr_crd = ''
                self.get_next_icrd = False
                self.get_next_ocrd = True
                if self.get_stats:
                    self.ocrd_drop_cnt += 1
        elif self.get_next_icrd:
            self.curr_crd = ''
            self.curr_inner_crd = ''
            if self.get_stats:
                self.ocrd_drop_cnt += 1
        else:
            self.curr_inner_crd = ''

        if self.debug:
            print("DEBUG: CRDMASK: Curr OuterCrd:", self.curr_ocrd, "\tCurr InnerCrd:", icrd,
                  "\t Curr OutputCrd:", self.curr_crd, "\tHasCrd", self.has_crd,
                  "\t GetNext InnerCrd:", self.get_next_icrd, "\t GetNext OuterCrd:", self.get_next_ocrd,
                  "\n Prev Stkn:", self.prev_ocrd_stkn, "\t Get Stkn:", self.get_stkn)

    def set_outer_crd(self, crd):
        if crd != '' and crd is not None:
            self.outer_crd.append(crd)

    def set_inner_crd(self, crd):
        if crd != '' and crd is not None:
            self.inner_crd.append(crd)

    def out_crd_outer(self):
        return self.curr_crd

    def out_crd_inner(self):
        return self.curr_inner_crd

    def print_fifos(self):
        print("CrdMask Inner crd fifos size: ", self.inner_crd_fifo)
        print("CrdMask Outer crd fifo size: ", self.outer_crd_fifo)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"inner_crd_fifo": self.inner_crd_fifo, "outer_crd_fifo":
                          self.outer_crd_fifo, "drop_count": self.ocrd_drop_cnt}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

