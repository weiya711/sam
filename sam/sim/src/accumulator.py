from .base import *

class Reduce(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.curr_out = ""
        self.in_val_size = 0
        self.sum = 0
        self.emit_stkn = False
        self.curr_in_val = None

    def update(self):
        curr_in_val = ""
        if self.done:
            self.curr_out = ""
        elif self.emit_stkn:
            self.curr_out = decrement_stkn(self.curr_in_val)
            self.emit_stkn = False
        elif len(self.in_val) > 0:
            self.curr_in_val = self.in_val.pop(0)
            if is_stkn(self.curr_in_val) and stkn_order(self.curr_in_val) == 0:
                self.curr_out = self.sum
                self.sum = 0
            elif is_stkn(self.curr_in_val) and stkn_order(self.curr_in_val) > 0:
                self.curr_out = self.sum
                self.sum = 0
                self.emit_stkn = True
            elif self.curr_in_val == 'D':
                self.done = True
                self.curr_out = 'D'
            else:
                self.sum += self.curr_in_val
                self.curr_out = ""
        else:
            self.curr_out = ""
        self.compute_fifos()
        if self.debug:
            print("DEBUG: REDUCE:", "\t CurrIn:", self.curr_in_val, "\tCurrOut:", self.curr_out,
                  "\t Sum:", self.sum)

    def set_in_val(self, val):
        if val != '':
            self.in_val.append(val)

    def out_val(self):
        return self.curr_out

    def compute_fifos(self):
        self.in_val_size = max(self.in_val_size, len(self.in_val))

    def print_fifos(self):
        print("FiFOo Val size for Reduce block: ", self.in_val_size)

