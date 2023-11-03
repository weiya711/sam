from .base import *


class Compression(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.in_crd = []

        self.curr_crd = ''
        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_crd = True
            self.fifo_avail_val = True

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in_val) > self.depth:
                self.fifo_avail_val = False
            else:
                self.fifo_avail_val = True
            if len(self.in_crd) > self.depth:
                self.fifo_avail_crd = False
            else:
                self.fifo_avail_crd = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
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

    def set_val(self, val, parent=None):
        if val != '' and val is not None:
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_val)

    def set_crd(self, crd, parent=None):
        if crd != '' and crd is not None:
            self.in_crd.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_crd)

    def out_crd(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_crd
