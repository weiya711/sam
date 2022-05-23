from .base import *


# Drops tokens
class StknDrop(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_stream = []

        self.curr_out = ''

    def update(self):
        ival = ''

        if self.done:
            self.curr_out = ''
            return

        if len(self.in_stream) > 0:
            ival = self.in_stream.pop(0)
            if ival == 'D':
                self.done = True
                self.curr_out = 'D'
                return

        self.curr_out = '' if is_stkn(ival) else ival

        if self.debug:
            print("Curr InnerCrd:", ival, "\t Curr OutputCrd:", self.curr_out)

    def set_in_stream(self, val):
        if val != '':
            self.in_stream.append(val)

    def out_val(self):
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
        ival = ''

        if self.done:
            self.curr_out = ''
            return

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
            elif self.prev_stkn and not self.leading_stkn:
                self.curr_out = self.largest_stkn
                self.largest_stkn = None
                self.prev_stkn = False
                self.prev_ival = ival
                self.emit_ival = True
            elif ival == 'D':
                self.done = True
                self.curr_out = 'D'
                self.prev_stkn = False
                self.leading_stkn = False
            elif isinstance(ival, int):
                self.leading_stkn = False
                self.curr_out = ival

        if self.debug:
            print("Curr InnerCrd:", ival, "\t Curr OutputCrd:", self.curr_out)

    # This can be both val or crd
    def set_in_stream(self, val):
        if val != '':
            self.in_stream.append(val)

    def out_val(self):
        return self.curr_out
