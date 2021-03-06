from .base import *


class Compression(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.in_crd = []

        self.curr_crd = ''

    def update(self):
        self.update_done()
        if len(self.in_val) > 0 or len(self.in_crd) > 0:
            self.block_start = False

        icrd = ""

        if self.done:
            self.curr_crd = ''
            return
        elif len(self.in_val) > 0 and len(self.in_crd) > 0:
            icrd = self

        if self.debug:
            print("Curr OuterCrd:", self.curr_ocrd, "\tCurr InnerCrd:", icrd, "\t Curr OutputCrd:", self.curr_crd,
                  "\tHasCrd", self.has_crd,
                  "\t GetNext InnerCrd:", self.get_next_icrd, "\t GetNext OuterCrd:", self.get_next_ocrd)

    def set_val(self, val):
        if val != '' and val is not None:
            self.in_val.append(val)

    def set_crd(self, crd):
        if crd != '' and crd is not None:
            self.in_crd.append(crd)

    def out_crd(self):
        return self.curr_crd
