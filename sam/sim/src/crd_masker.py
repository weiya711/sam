from .base import *
from .repeater import RepeatSigGen, Repeat
from .token import EmptyFiberStknDrop, StknDrop
from .crd_manager import CrdDrop
import numpy as np

# From Ritvik's reorder branch
class repeated_token_dropper(Primitive):
    def __init__(self, name="val"):
        self.last_token = ""
        self.new_token = ""
        self.output_token = ""
        self.name = name
        self.done = False
        self.first_stop = True
        self.in_token = []

    def add_input(self, token):
        if token != "" and token is not None:
            self.in_token.append(token)


    def add_token(self, token):
        # if token != "":
        #     self.last_token = self.new_token
        if token != "":
            self.new_token = token

    def get_token(self):
        return self.output_token

    def update(self):
        
        #if self.debug:
        #    print("REPEATED_BLK name:", self.name, self.first_stop, self.new_token)
        if len(self.in_token) > 0:
            self.new_token = self.in_token.pop(0)

        if self.done:
            return
        self.output_token = ""
        if self.first_stop and self.new_token != "" and not is_stkn(self.new_token):
            self.first_stop = False
        elif self.first_stop and is_stkn(self.new_token):
            return
                
        if (self.last_token == "D" or self.new_token != "") and (self.last_token != self.new_token or self.last_token == "D") and not(is_stkn(self.new_token) and is_stkn(self.last_token)):
            # print(self.name, "update1")
            self.output_token = self.last_token
            self.last_token = self.new_token
        elif self.new_token != "" and is_stkn(self.last_token) and is_stkn(self.new_token) and self.new_token != self.last_token:
            #print(self.name, "update2")
            self.last_token = self.new_token
            self.output_token = self.last_token
            self.last_token = ""
            self.new_token = ""
        if self.output_token == "D":
            self.done = True


class CrdMask(Primitive):
    def __init__(self, dimension=2, drop_predicate=lambda crds: False, name="", **kwargs):
        # Will drop a coordinate if drop_predicate returns True
        # drop_predicate takes in some number of current coordinates and returns True/False to drop/not drop

        super().__init__(**kwargs)

        # TODO: innermost dimension is index 0. Perhaps the outermost dimension should be?
        self.name = name
        self.dimension = dimension
        self.in_crd_array = [[] for i in range(self.dimension)]
        self.curr_crd_array = [None for i in range(self.dimension)]
        self.out_crd_array = ['' for i in range(self.dimension)]
        self.inner_ref = ""
        self.inner_ref_arr = []
        self.start = True
        self.out_dropper = repeated_token_dropper(name="out_crd")
        self.prob = []
        self.drop_prob = 0
        self.curr_i = 0
        self.outer = []
        self.inner = []
        self.outer_after = []
        self.inner_after = []
        self.inner_stkn_dropper = EmptyFiberStknDrop()
        self.outer_stkn_dropper = EmptyFiberStknDrop()
        self.dropper = CrdDrop()
        self.actual_done = False

        self.drop_predicate = drop_predicate

        if self.backpressure_en:
            self.ready_backpressure = True
            self.dimension = dimension 
            self.data_valid = True
            self.fifo_avail_inner = True
            self.fifo_avail_outer = True

        # statistics info
        if self.get_stats:
            self.crd_fifos = [0 for i in range(self.dimension)]
            self.crd_drop_cnt = 0

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def set_curr_i(self, i):
        self.curr_i = i

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def update_ready(self):
        if self.backpressure_en:
            if len(self.inner_crd) > self.depth:
                self.fifo_avail_inner = False
            else:
                self.fifo_avail_inner = True
            if len(self.outer_crd) > self.depth:
                self.fifo_avail_outer = False
            else:
                self.fifo_avail_outer = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.get_stats:
            self.crd_fifos = [max(self.crd_fifos[i], len[self.in_crd_array[i]]) for i in range(self.dimension)]

        if self.out_done():
            return

        self.start = True

        for i in range(self.dimension):
            if self.in_crd_array[i] == []:
                self.start = False

        if self.start:
            self.inner_ref = self.inner_ref_arr.pop(0)
            for i in range(self.dimension):
                self.curr_crd_array[i] = self.in_crd_array[i].pop(0)
                # if self.curr_crd_array[i] == None:
                    # initialization: don't skip any
                    # print("Inside first pop")
                    # self.curr_crd_array[i] = self.in_crd_array[i].pop(0)

                # else: 
                    # self.curr_crd_array[i] = self.in_crd_array[i].pop(0)
                    # if not is_stkn(self.out_crd_array[i]):
                        # not a stop token: hold higher dimensions
                        # break

        if self.curr_crd_array[0] is not None:
            self.out_crd_array = self.curr_crd_array

        # if self.out_crd_array[0] == 'D':
        #     self.done = True
        #     return

        print(self.out_crd_array)
        # print(self.drop_prob)
        # print(self.prob)
        #

        if not is_stkn(self.out_crd_array[0]) and self.out_crd_array[0] != 'D' and self.out_crd_array[0] != "":
            # TODO: Added for debugging
            if self.name == "random":
                if len(self.prob) != 0:
                    # print("Dropping", self.prob[self.curr_i, self.out_crd_array[1], self.out_crd_array[0]])
                    self.set_predicate(self.prob[self.curr_i, self.out_crd_array[1], self.out_crd_array[0]], self.drop_prob)
                    # self.set_predicate(self.prob[self.out_crd_array[1], self.out_crd_array[0]], self.drop_prob)
            if self.drop_predicate(self.out_crd_array):
                print("Dropping: ", self.curr_i, *self.out_crd_array)
                # drop (may need to follow up with crd dropper?)
                self.out_crd_array = ['' for i in range(self.dimension)]
                self.inner_ref = ''

        self.inner.append(self.out_crd_array[0])
        self.outer.append(self.out_crd_array[1])

        print("j: ", remove_emptystr(self.outer))
        print("k: ", remove_emptystr(self.inner))

        print()
        # if is_stkn(old_token) and old_token == self.out_dropper.get_token():
        #     self.out_crd_array[0] = ''
        #     self.out_crd_array[1] = ''
        # else:
        # self.out_crd_array[1] = self.out_dropper.get_token()
        # dedup_outer_crd = self.out_dropper.get_token()
        # self.outer_stkn_dropper.set_in_stream(dedup_outer_crd)

        # undo coord hold for outer level crd
        self.out_dropper.add_token(self.out_crd_array[1])
        self.out_dropper.update()
        self.out_crd_array[1] = self.out_dropper.get_token()

        self.inner_stkn_dropper.set_in_stream(self.out_crd_array[0])
        self.outer_stkn_dropper.set_in_stream(self.out_crd_array[1])

        self.inner_stkn_dropper.update()
        self.outer_stkn_dropper.update()

        self.out_crd_array[0] = self.inner_stkn_dropper.out_val()
        self.out_crd_array[1] = self.outer_stkn_dropper.out_val()

        if self.out_crd_array[0] == 'D' and self.out_crd_array[1] == 'D':
            self.done = True
        # self.dropper.set_outer_crd(self.out_crd_array[1])
        # self.dropper.set_inner_crd(self.out_crd_array[0])

        # self.dropper.update()

        # self.out_crd_array[0] = self.dropper.out_crd_inner()
        # self.out_crd_array[1] = self.dropper.out_crd_outer()

        self.inner_after.append(self.out_crd_array[0])
        self.outer_after.append(self.out_crd_array[1])
        print("j after: ", remove_emptystr(self.outer_after))
        print("k after: ", remove_emptystr(self.inner_after))

        print()


    def print_fifos(self):
        for i in range(self.dimension):
            print("CrdMask crd fifo ", i, " size: ", self.crd_fifos[i])

    def set_prob(self, prob, drop_prob):
        self.prob = prob
        self.drop_prob = drop_prob

    # For debug purposes
    def set_predicate(self, prob, drop_prob):
        self.drop_predicate=lambda crds: ~(prob < (1 - drop_prob))
        # self.drop_predicate=lambda crds: False

    def set_inner_crd(self, crd):
        # hard code inner dimension to 0
        inner_dim = 0
        if crd != '' and crd is not None:
            self.in_crd_array[inner_dim].append(crd)

    def set_outer_crd(self, crd):
        # hard code outer dimension to 1
        outer_dim = 1
        if crd != '' and crd is not None:
            self.in_crd_array[outer_dim].append(crd)

    def set_inner_ref(self, ref):
        if ref != '' and ref is not None:
            self.inner_ref_arr.append(ref)

    def out_ref(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.inner_ref

    def out_crd(self, dimension):
        # if dimension == 0:
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.out_crd_array[dimension]
        # else:
        #     self.out_dropper.add_token(self.out_crd_array[dimension])
        #     self.out_dropper.update()
            # return self.out_dropper.get_token()


            # # if self.out_crd_array[dimension] == "":
            #     # return ''
            # self.new_token = self.out_crd_array[dimension]
            # if self.first_stop and self.new_token != "" and not is_stkn(self.new_token):
            #     self.first_stop = False
            # # elif self.first_stop and is_stkn(self.new_token) or (is_stkn(self.last_token) and is_stkn(self.new_token)) or self.last_token == "D":
            # elif self.first_stop and is_stkn(self.new_token):
            #     return ""
            # if (self.last_token == "D" or self.new_token != "") and (self.last_token != self.new_token or self.last_token == "D") and not(is_stkn(self.new_token) and is_stkn(self.last_token)):
            #     # print(self.name, "update1")
            #     self.output_token = self.last_token
            #     self.last_token = self.new_token
            # elif self.new_token != "" and is_stkn(self.last_token) and is_stkn(self.new_token) and self.new_token != self.last_token:
            #     #print(self.name, "update2")
            #     self.last_token = self.new_token
            #     self.output_token = self.last_token
            #     self.last_token = ""
            #     self.new_token = ""
            # elif is_stkn(self.last_token) and is_stkn(self.new_token):
            #     self.output_token = ""
            # return self.output_token
            # if new_token == self.last_token and new_token != "":
            #     return ""
            # self.last_token = new_token
            # return new_token

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"crd_fifos": self.crd_fifos, "drop_count": self.crd_drop_cnt}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


class RandomDropout(CrdMask):
    def __init__(self, dimension=2, drop_probability=0.5, **kwargs):
        super().__init__(dimension, lambda crds: np.random.rand() < (1 - drop_probability), name="random", **kwargs)

class LowerTriangular(CrdMask):
    def __init__(self, dimension=2, **kwargs):
        super().__init__(dimension, lambda crds: crds[0] > crds[1], name="tril", **kwargs)

class UpperTriangular(CrdMask):
    def __init__(self, dimension=2, **kwargs):
        super().__init__(dimension, lambda crds: crds[0] < crds[1], name="triu", **kwargs)

class Diagonal(CrdMask):
    def __init__(self, dimension=2, **kwargs):
        super().__init__(dimension, lambda crds: crds[0] != crds[1], name="diag", **kwargs)
