from abc import ABC, abstractmethod
from .base import Primitive


class Compute2(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in1 = []
        self.in2 = []

        self.curr_out = None

    @abstractmethod
    def out_val(self):
        pass

    def set_in1(self, in1):
        if in1 != '':
            self.in1.append(in1)

    def set_in2(self, in2):
        if in2 != '':
            self.in2.append(in2)

    def out_val(self):
        return self.curr_out

class Add2(Compute2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self):
        if len(self.in1) > 0 and len(self.in2) > 0:
            curr_in1 = self.in1.pop(0)
            curr_in2 = self.in2.pop(0)
            if curr_in1 == 'D' or curr_in2 == 'D':
                # Inputs are both the same and done tokens
                assert(curr_in1 == curr_in2)
                self.curr_out = curr_in1
                self.done = True
            elif curr_in1 == 'S' or curr_in2 == 'S':
                # Inputs are both the same and stop tokens
                assert (curr_in1 == curr_in2)
                self.curr_out = curr_in1
            else:
                # Both inputs are values
                self.curr_out = curr_in1 + curr_in2

            if self.debug:
                print("DEBUG: Curr Out:", self.curr_out, "\t Curr In1:", curr_in1, "\t Curr In2:", curr_in2)
        else:
            self.curr_out = ''



class Multiply2(Compute2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update(self):
        if len(self.in1) > 0 and len(self.in2) > 0:
            curr_in1 = self.in1.pop(0)
            curr_in2 = self.in2.pop(0)
            if curr_in1 == 'D' or curr_in2 == 'D':
                # Inputs are both the same and done tokens
                assert(curr_in1 == curr_in2)
                self.curr_out = curr_in1
                self.done = True
            elif curr_in1 == 'S' or curr_in2 == 'S':
                # Inputs are both the same and stop tokens
                assert (curr_in1 == curr_in2)
                self.curr_out = curr_in1
            else:
                # Both inputs are values
                self.curr_out = curr_in1 * curr_in2

            if self.debug:
                print("DEBUG: Curr Out:", self.curr_out, "\t Curr In1:", curr_in1, "\t Curr In2:", curr_in2)
        else:
            self.curr_out = ''
