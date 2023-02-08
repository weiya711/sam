from .base import *
import math


class UnaryALU(Primitive, ABC):
    def __init__(self, in2=0, **kwargs):
        super().__init__(**kwargs)

        self.in1 = []
        self.in2 = in2

        if self.get_stats:
            self.in1_size = 0
            # self.in2_size = 0
            self.cycles_operated = 0
        self.curr_out = None

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_in1 = True
            self.fifo_avail_in2 = True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in1) > self.depth:
                self.fifo_avail_in1 = False
            else:
                self.fifo_avail_in1 = True
            # if len(self.in2) > self.depth:
            #     self.fifo_avail_in2 = False
            # else:
            #     self.fifo_avail_in2 = True

    def set_in1(self, in1):
        if in1 != '' and in1 is not None:
            self.in1.append(in1)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_in1)

    # Using in2 as the first param to user-defined function
    # def set_in2(self, in2):
    #     if in2 != 0 and in2 is not None:
    #         self.in2 = in2

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_out

    def compute_fifos(self):
        if self.get_stats:
            self.in1_size = max(self.in1_size, len(self.in1))
            # self.in2_size = max(self.in2_size, len(self.in2))

    def print_fifos(self):
        print("Compute block in 1: ", self.in1_size)
        # print("Compute block in 2: ", self.in2_size)

    def return_statistics(self):
        if self.get_stats:
            dic = {"cycle_operation": self.cycles_operated}
            dic.update(super().return_statistics())
        else:
            dic = {}
        return dic


class Exp(UnaryALU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fill_value = 0

        self.get1 = True
        self.get2 = True

        self.curr_in1 = ''
        self.curr_in2 = ''

    def update(self):
        self.update_done()
        if (len(self.in1) > 0):
            self.block_start = False

        if len(self.in1) > 0:
            if self.get1:
                self.curr_in1 = self.in1.pop(0)
            if self.curr_in1 == 'D':
                # Inputs is done token
                self.curr_out = self.curr_in1
                self.get1 = True
                self.done = True
            elif is_stkn(self.curr_in1):
                # Input is stop token
                self.curr_out = self.curr_in1
                self.get1 = True
            else:
                # Input is value stream
                self.curr_out = math.exp(self.curr_in1)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
                # self.get2 = True
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''

class Sin(UnaryALU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fill_value = 0

        self.get1 = True

        self.curr_in1 = ''

    def update(self):
        self.update_done()
        if (len(self.in1) > 0):
            self.block_start = False

        if len(self.in1) > 0:
            if self.get1:
                self.curr_in1 = self.in1.pop(0)
            if self.curr_in1 == 'D':
                # Inputs is done token
                self.curr_out = self.curr_in1
                self.get1 = True
                self.done = True
            elif is_stkn(self.curr_in1):
                # Input is stop token
                self.curr_out = self.curr_in1
                self.get1 = True
            else:
                # Both inputs are values
                self.curr_out = math.sin(self.curr_in1)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            self.compute_fifos()
            if self.debug:
                print("DEBUG: Sin: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''

class Cos(UnaryALU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fill_value = 0

        self.get1 = True
        self.get2 = True

        self.curr_in1 = ''

    def update(self):
        self.update_done()
        if (len(self.in1) > 0):
            self.block_start = False

        if len(self.in1) > 0:
            if self.get1:
                self.curr_in1 = self.in1.pop(0)
            if self.curr_in1 == 'D':
                # Inputs is done token
                self.curr_out = self.curr_in1
                self.get1 = True
                self.done = True
            elif is_stkn(self.curr_in1):
                # Input is stop token
                self.curr_out = self.curr_in1
                self.get1 = True
            else:
                # Input is value stream
                self.curr_out = math.sin(self.curr_in1)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''

class Max(UnaryALU):
    def __init__(self, in2=0, **kwargs):
        super().__init__(in2, **kwargs)
        self.fill_value = 0

        self.get1 = True
        # self.get2 = True

        self.curr_in1 = ''
        self.curr_in2 = in2

    def update(self):
        self.update_done()
        # if self.out_done():
            # return
        self.update_ready()
        if (len(self.in1) > 0):
            self.block_start = False

        if len(self.in1) > 0:
            if self.get1:
                self.curr_in1 = self.in1.pop(0)
            # if self.get2:
            #     self.curr_in2 = self.in2
            #     self.get2 = False
            if self.curr_in1 == 'D':
                # Inputs is done token
                self.curr_out = self.curr_in1 
                self.get1 = True
                # self.get2 = True
                self.done = True
            elif is_stkn(self.curr_in1):
                # Input is stop token
                self.curr_out = self.curr_in1
                self.get1 = True
                # self.done = True
                # self.get2 = True
            else:
                # Input is value stream
                self.curr_out = max(self.curr_in1, self.curr_in2)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''


class ScalarMult(UnaryALU):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fill_value = 0

        self.get1 = True
        self.get2 = True

        self.curr_in1 = ''
        self.curr_in2 = 0

    def update(self):
        self.update_done()
        if (len(self.in1) > 0):
            self.block_start = False

        if len(self.in1) > 0:
            if self.get1:
                self.curr_in1 = self.in1.pop(0)
            if self.get2:
                self.curr_in2 = self.in2
            if self.curr_in1 == 'D':
                # Inputs is done token
                self.curr_out = self.curr_in1
                self.get1 = True
                # self.get2 = True
                self.done = True
            elif is_stkn(self.curr_in1):
                # Input is stop token
                self.curr_out = self.curr_in1
                self.get1 = True
                # self.get2 = True
            else:
                # Input is value stream
                self.curr_out = self.curr_in1 * self.curr_in2
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''



