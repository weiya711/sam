from .base import *

# Drops tokens
class TknDrop(Primitive):
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
