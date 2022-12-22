from .base import *


# Drops tokens
class StknDrop(Primitive):
    def __init__(self, depth=1, **kwargs):
        super().__init__(**kwargs)

        self.in_stream = []

        self.curr_out = ''
        if self.backpressure_en:
            self.backpressure = []
            self.data_ready = True
            self.branch = []
            self.depth = depth

    def check_backpressure(self):
        if self.backpressure_en:
            j = 0
            for i in self.backpressure:
                if not i.fifo_available(self.branch[j]):
                    return False
                j += 1
        return True

    def fifo_available(self, br=""):
        if self.backpressure_en:
            if len(self.in_stream) > self.depth:
                return False
        return True

    def add_child(self, child=None, branch=""):
        if self.backpressure_en and child is not None:
            self.backpressure.append(child)
            self.branch.append(branch)

    def update(self):
        self.update_done()
        self.data_ready = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_ready = True
            if len(self.in_stream) > 0:
                self.block_start = False

            ival = ''

            if self.done:
                self.curr_out = ''
                #return

            if len(self.in_stream) > 0:
                ival = self.in_stream.pop(0)
                if ival == 'D':
                    self.done = True
                    self.curr_out = 'D'
                    return
                else:
                    self.done = False

            self.curr_out = '' if is_stkn(ival) else ival

        if self.debug:
            print("Curr InnerCrd:", ival, "\t Curr OutputCrd:", self.curr_out)

    def set_in_stream(self, val):
        if val != '' and val is not None:
            self.in_stream.append(val)

    def out_val(self):
        if (self.backpressure_en and self.data_ready) or not self.backpressure_en:
            return self.curr_out


class EmptyFiberStknDrop(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_stream = []

        self.largest_stkn = None
        self.prev_stkn = False
        self.leading_stkn = True

        self.emit_ival = False
        self.prev_ival = None
        self.curr_out = ''

    def update(self):
        self.update_done()
        if len(self.in_stream) > 0:
            self.block_start = False

        ival = ''

        if self.done:
            self.curr_out = ''
            # return

        if self.emit_ival:
            self.curr_out = self.prev_ival
            self.emit_ival = False

            if self.prev_ival == 'D':
                self.curr_out = self.prev_ival
                self.done = True
                self.prev_stkn = False
                self.leading_stkn = False
            return

        if len(self.in_stream) > 0:
            ival = self.in_stream.pop(0)
            if is_stkn(ival) and not self.leading_stkn:
                self.largest_stkn = ival if self.largest_stkn is None else larger_stkn(self.largest_stkn, ival)
                self.curr_out = ''
                self.prev_stkn = True
                self.done = False
            elif self.prev_stkn and not self.leading_stkn:
                self.curr_out = self.largest_stkn
                self.largest_stkn = None
                self.prev_stkn = False
                self.prev_ival = ival
                self.emit_ival = True
                self.done = False
            elif ival == 'D':
                self.done = True
                self.curr_out = 'D'
                self.prev_stkn = False
                self.leading_stkn = False
            elif isinstance(ival, int):
                self.leading_stkn = False
                self.curr_out = ival
                sefl.done = False
        else:
            self.curr_out = ''

        if self.debug:
            print("Curr InnerCrd:", ival, "\t Curr OutputCrd:", self.curr_out)

    # This can be both val or crd
    def set_in_stream(self, val):
        if val != '' and val is not None:
            self.in_stream.append(val)

    def out_val(self):
        return self.curr_out
