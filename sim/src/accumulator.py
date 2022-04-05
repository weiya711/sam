from abc import ABC, abstractmethod
from .base import Primitive

class Reduce(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.curr_out = ""

        self.first_stop = True
        self.sum = 0

    def update(self):
        curr_in_val = ""
        if self.done:
            self.curr_out = ""
        elif len(self.in_val) > 0:
            curr_in_val = self.in_val.pop(0)
            if curr_in_val == 'S' and self.first_stop:
                self.curr_out = self.sum
                self.sum = 0
                self.first_stop = False
            elif curr_in_val == 'S':
                self.curr_out = 'S'
            elif curr_in_val == 'D':
                self.done = True
                self.curr_out = 'D'
            else:
                self.first_stop = True
                self.sum += curr_in_val
                self.curr_out = ""
        else:
            self.curr_out = ""

        if self.debug:
            print("DEBUG: REDUCE:", "\t CurrIn:", curr_in_val, "\tCurrOut:", self.curr_out,
                  "\t 1S:", self.first_stop, "\t Sum:", self.sum)

    def set_in_val(self, val):
        if val != '':
            self.in_val.append(val)

    def out_val(self):
        return self.curr_out