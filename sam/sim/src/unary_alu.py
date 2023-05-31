from .base import *
import math
from .compute import Divide2, Add2
from .crd_manager import CrdDrop
from .repeater import Repeat, RepeatSigGen
from .accumulator import Reduce, MaxReduce


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
            elif isinstance(self.curr_in1, float):
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
            elif isinstance(self.curr_in1, float):
                # Input is value stream
                self.curr_out = max(self.curr_in1, self.curr_in2)
                if self.debug:
                    if self.curr_out != self.curr_in1:
                        print("Dropping: ", self.curr_in1)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            # else:
            #     self.curr_out = self.curr_in1
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''


class Square(UnaryALU):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.fill_value = 0

        self.get1 = True
        # self.get2 = True

        self.curr_in1 = ''
        self.curr_in2 = None

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
            elif isinstance(self.curr_in1, float):
                # Input is value stream
                self.curr_out = pow(self.curr_in1, 2)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            # else:
            #     self.curr_out = self.curr_in1
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''


class SquareRoot(UnaryALU):
    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.fill_value = 0

        self.get1 = True
        # self.get2 = True

        self.curr_in1 = ''
        self.curr_in2 = None

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
            elif isinstance(self.curr_in1, float):
                # Input is value stream
                self.curr_out = math.sqrt(self.curr_in1)
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
            # else:
            #     self.curr_out = self.curr_in1
            self.compute_fifos()
            if self.debug:
                print("DEBUG: EXP: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''



class ScalarMult(UnaryALU):
    def __init__(self, in2=1.0, **kwargs):
        super().__init__(**kwargs)
        self.fill_value = 0

        self.get1 = True
        self.get2 = True

        self.curr_in1 = ''
        self.curr_in2 = in2

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
            elif isinstance(self.curr_in1, float):
                # Input is value stream
                # self.curr_out = math.exp(self.curr_in1)
                self.curr_out = self.curr_in1 * self.curr_in2
                if self.get_stats:
                    self.cycles_operated += 1
                self.get1 = True
                # self.get2 = True
            self.compute_fifos()
            if self.debug:
                print("DEBUG: ScalarMult: \t "
                      "Curr Out:", self.curr_out, "\t Curr In1:", self.curr_in1)
        else:
            self.curr_out = ''


class Softmax(Primitive):
    def __init__(self, row_wise=True, block_size=1, debug_sim=False, **kwargs):
        super().__init__(**kwargs)
        self.repeat_siggen = RepeatSigGen(debug=debug_sim)
        self.repeat = Repeat(debug=debug_sim)
        self.repeat1 = Repeat(debug=debug_sim)
        self.exp_1 = Exp(in2=0, debug=debug_sim)
        self.reduce_5 = Reduce(debug=debug_sim)
        self.max_reduce_5 = MaxReduce(debug=debug_sim)
        self.div_6 = Divide2(debug=debug_sim)
        self.add_10 = Add2(debug=debug_sim, neg2=True)
        self.in_val = []
        self.inner_ref = []
        self.curr_val = ''
        self.curr_inner_ref = ''

        self.div_0 = []
        self.div_1 = []
        self.done = False
        self.row_wise = row_wise
        self.block_size = block_size

    def update(self):
        self.update_done()

        if self.done:
            self.curr_inner_ref = ''
            self.curr_val = ''
            return
        if len(self.inner_ref) > 0 and len(self.in_val) > 0:
            self.curr_inner_ref = self.inner_ref.pop(0)
            self.curr_val = self.in_val.pop(0)

        # if self.curr_val == 'D' and self.curr_inner_ref == 'D':
        #     self.done = True
        # if self.row_wise:
        #     self.max_reduce_5.set_in_val(self.curr_val)
        # else:
        #     if self.block_size > 1:
        #         self.max
        self.repeat_siggen.set_istream(self.curr_inner_ref)
        self.repeat.set_in_ref(self.max_reduce_5.out_val())
        self.repeat.set_in_repsig(self.repeat_siggen.out_repsig())
        self.add_10.set_in1(self.curr_val)
        self.add_10.set_in2(self.repeat.out_ref())
        self.exp_1.set_in1(self.add_10.out_val())
        self.reduce_5.set_in_val(self.exp_1.out_val())
        self.repeat1.set_in_ref(self.reduce_5.out_val())
        self.repeat1.set_in_repsig(self.repeat_siggen.out_repsig())
        self.div_6.set_in1(self.exp_1.out_val())
        self.div_6.set_in2(self.repeat1.out_ref())
        if self.div_6.out_val() == 'D':
            self.done = True

        self.div_0.append(self.exp_1.out_val())
        self.div_1.append(self.repeat1.out_ref())

        # print("div 0:", remove_emptystr(self.div_0))
        # print("div 1:", remove_emptystr(self.div_1))

        self.max_reduce_5.update()
        self.repeat_siggen.update()
        self.repeat.update()
        self.add_10.update()
        self.exp_1.update()
        self.reduce_5.update()
        self.repeat1.update()
        self.div_6.update()

    def set_inner_ref(self, ref, parent=None):
        if ref != '' and ref is not None:
            self.inner_ref.append(ref)
        # print("inner ref:", self.inner_ref)
        # if self.backpressure_en:
        #     parent.set_backpressure(self.fifo_avail_outer)

    def set_val(self, val, parent=None):
        if val != '' and val is not None:
            self.in_val.append(val)
        # print("in val", self.in_val)

    def out_val(self):
        if self.done:
            return ''
        return self.div_6.out_val()



