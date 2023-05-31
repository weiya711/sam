from .base import *
from .repeater import RepeatSigGen, Repeat
from .token import EmptyFiberStknDrop, StknDrop
from .crd_manager import CrdDrop, CrdHold
import numpy as np


# From Ritvik's reorder branch
class CrdPtCompressor(Primitive):
    def __init__(self, name="val", **kwargs):
        super().__init__(**kwargs)
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

        # if self.debug:
        #    print("REPEATED_BLK name:", self.name, self.first_stop, self.new_token)
        if len(self.in_token) > 0:
            self.new_token = self.in_token.pop(0)

        if is_stkn(self.new_token):
            self.new_token = decrement_stkn(self.new_token)

        if self.done:
            return
        self.output_token = ""
        if self.first_stop and self.new_token != "" and not is_stkn(self.new_token):
            self.first_stop = False
        elif self.first_stop and is_stkn(self.new_token):
            return

        if (self.last_token == "D" or self.new_token != "") and (
                self.last_token != self.new_token or self.last_token == "D") and not (
                is_stkn(self.new_token) and is_stkn(self.last_token)):
            # print(self.name, "update1")
            self.output_token = self.last_token
            self.last_token = self.new_token
        elif self.new_token != "" and is_stkn(self.last_token) and is_stkn(
                self.new_token) and self.new_token != self.last_token:
            # print(self.name, "update2")
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
        self.out_dropper = CrdPtCompressor(name="out_crd")
        self.prob = []
        self.drop_prob = 0
        self.curr_i = 0
        self.outer = []
        self.inner = []
        self.outer_after = []
        self.inner_after = []
        self.inner_stkn_dropper = EmptyFiberStknDrop()
        self.inner_ref_stkn_dropper = EmptyFiberStknDrop()
        self.outer_stkn_dropper = EmptyFiberStknDrop()
        self.dropper = CrdDrop()
        self.get_inner = True
        self.get_outer = True
        self.random_dropped = []
        self.debug_arr = []
        self.crd_outer = []
        self.crd_inner = []
        self.crd_outer1 = []
        self.crd_inner1 = []

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

        if self.done:
            self.out_crd_array[0] = ''
            self.out_crd_array[1] = ''
            self.inner_ref = ''
            return

        self.start = True

        for i in range(self.dimension):
            if not self.in_crd_array[i]:
                self.start = False

        # if self.out_crd_array[0] == 'D' and self.out_crd_array[1] == 'D':
        #     self.start = False
        #     self.curr_crd_array[0] == ''
        #     self.curr_crd_array[1] == ''
        #     self.inner_ref = ''
        #

        self.inner_ref = ''

        if self.start:
            if self.inner_ref != 'D':
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

        if self.curr_crd_array[0] is not None and self.done != True:
            self.out_crd_array = self.curr_crd_array
        self.curr_crd_array = ['' for i in range(self.dimension)]

        #     self.done = True
        #     return
        # self.inner.append(self.inner_ref)
        # self.outer.append(self.out_crd_array[1])

        # print("j: ", remove_emptystr(self.outer))
        # print("ref: ", remove_emptystr(self.inner))
        self.inner.append(self.inner_ref)

        self.crd_inner1.append(self.out_crd_array[0])
        self.crd_outer1.append(self.out_crd_array[1])
        # print("crd outer before: ", remove_emptystr(self.crd_outer1))
        # print("crd inner before: ", remove_emptystr(self.crd_inner1))

        if not is_stkn(self.out_crd_array[0]) and self.out_crd_array[0] != 'D' and self.out_crd_array[0] != "":
            # TODO: Added for debugging
            dropped = self.drop_predicate(self.out_crd_array)
            self.random_dropped.append(dropped)
            if dropped:
                # drop (may need to follow up with crd dropper?)
                self.out_crd_array = ['' for i in range(self.dimension)]
                self.inner_ref = ''
        self.debug_arr.append(self.inner_ref)
        self.crd_inner.append(self.out_crd_array[0])
        self.crd_outer.append(self.out_crd_array[1])
        # print("crd outer: ", remove_emptystr(self.crd_outer))
        # print("crd inner: ", remove_emptystr(self.crd_inner))

        # self.outer.append(self.out_crd_array[1])

        # print("j: ", remove_emptystr(self.outer))
        # print("inner ref: ", remove_emptystr(self.inner))

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
        self.inner_ref_stkn_dropper.set_in_stream(self.inner_ref)

        self.inner_stkn_dropper.update()
        self.outer_stkn_dropper.update()
        self.inner_ref_stkn_dropper.update()

        self.out_crd_array[0] = self.inner_stkn_dropper.out_val()
        self.out_crd_array[1] = self.outer_stkn_dropper.out_val()
        self.inner_ref = self.inner_ref_stkn_dropper.out_val()
        self.inner_after.append(self.inner_ref)

        # if self.out_crd_array[0] == 'D' and self.out_crd_array[1] == 'D':
        if self.out_crd_array[0] == 'D' and self.inner_ref == 'D' and self.out_crd_array[1] != 'D':
            self.get_inner = False
        elif self.out_crd_array[0] == 'D' and self.inner_ref == 'D' and self.out_crd_array[1] == 'D':
            self.get_inner = False
            self.get_outer = False
            self.done = True
        elif self.out_crd_array[1] == 'D' and self.get_inner == False:
            self.out_crd_array[0] = ''
            self.inner_ref = ''
            self.done = True
            self.get_outer = False
        elif self.get_inner == False:
            self.out_crd_array[0] = ''
            self.inner_ref = ''
        elif self.get_outer == False:
            self.out_crd_array[1] = ''
        # self.dropper.set_outer_crd(self.out_crd_array[1])
        # self.dropper.set_inner_crd(self.out_crd_array[0])

        # self.dropper.update()

        # self.out_crd_array[0] = self.dropper.out_crd_inner()
        # self.out_crd_array[1] = self.dropper.out_crd_outer()

        # self.inner_after.append(self.out_crd_array[0])
        # self.outer_after.append(self.out_crd_array[1])
        # print("j after: ", remove_emptystr(self.outer_after))
        # print("inner ref before stkndrop: ", remove_emptystr(self.debug_arr))
        # print("inner ref after: ", remove_emptystr(self.inner_after))
        # print(self.inner_ref)

        # print()

    def print_fifos(self):
        for i in range(self.dimension):
            print("CrdMask crd fifo ", i, " size: ", self.crd_fifos[i])

    def set_prob(self, prob, drop_prob):
        self.prob = prob
        self.drop_prob = drop_prob

    # For debug purposes
    def set_predicate(self, prob, drop_prob):
        self.drop_predicate = lambda crds: ~(prob < (1 - drop_prob))
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


class Relu(Primitive):
    def __init__(self, num_dims=3, **kwargs):
        super().__init__(**kwargs)
        self.out_crd_array[num_dims] = []


class Tril(Primitive):
    def __init__(self, debug=False, statistics=False, name="", back_en=False, **kwargs):
        super().__init__(debug, statistics, name, back_en, **kwargs)

        self.crd_hold = CrdHold(debug=debug_sim, statistics=statistics)
        self.drop_1 = CrdDrop(debug=debug_sim, statistics=statistics)
        self.drop_2 = CrdDrop(debug=debug_sim, statistics=statistics)
        self.tril = LowerTriangular(dimension=2, debug=debug_sim, statistics=statistics)
        self.done = False
        self.outer_crd = []
        self.inner_crd = []
        self.inner_ref = []
        self.curr_outer_crd = ''
        self.curr_inner_crd = ''
        self.curr_inner_ref = ''
        self.curr_crd0 = ''
        self.curr_crd1 = ''
        self.crd0 = []
        self.crd1 = []

    def update(self):
        self.update_done()

        if self.done:
            self.curr_outer_crd = ''
            self.curr_inner_crd = ''
            self.curr_inner_ref = ''
            return
        if len(self.outer_crd) > 0 and len(self.inner_crd) > 0 and len(self.inner_ref) > 0:
            self.curr_outer_crd = self.outer_crd.pop(0)
            self.curr_inner_crd = self.inner_crd.pop(0)
            self.curr_inner_ref = self.inner_ref.pop(0)
            self.curr_crd0 = self.crd0.pop(0)
            self.curr_crd1 = self.crd1.pop(0)

        self.crd_hold.set_outer_crd(self.curr_outer_crd)
        self.crd_hold.set_inner_crd(self.curr_inner_crd)

        self.tril.set_inner_crd(crd_hold.out_crd_inner())
        self.tril.set_outer_crd(crd_hold.out_crd_outer())
        self.tril.set_inner_ref(self.curr_inner_ref)

        self.drop_1.set_inner_crd(self.tril.out_crd(1))
        self.drop_1.set_outer_crd(self.curr_crd1)

        self.drop_2.set_inner_crd(self.drop_1.out_crd_outer())
        self.drop_2.set_outer_crd(self.curr_crd0)

        if self.tril.out_ref == 'D':
            self.done = True

        self.crd_hold.update()
        self.tril.update()
        self.drop_1.update()
        self.drop_2.update()

    def set_inner_crd(self, crd):
        if crd != '' and crd is not None:
            self.inner_crd.append(crd)

    def set_outer_crd(self, crd):
        if crd != '' and crd is not None:
            self.outer_crd.append(crd)

    def set_inner_ref(self, ref):
        if ref != '' and ref is not None:
            self.inner_ref.append(ref)

    def set_crd0(self, crd):
        if crd != '' and crd is not None:
            self.crd0.append(crd)

    def set_crd1(self, crd):
        if crd != '' and crd is not None:
            self.crd1.append(crd)

    def out_ref(self):
        return self.tril.out_ref()

    def out_crd_inner(self):
        return self.tril.out_crd(0)
    
    def out_crd_outer(self):
        return self.tril.out_crd(1)
    
    def out_crd0(self):
        return self.drop_2.out_crd_outer()
    
    def out_crd1(self):
        return self.drop_1.out_crd_outer()

class Dropout(Primitive):
    def __init__(self, drop_prob=0.5, debug=False, statistics=False, name="", back_en=False, **kwargs):
        super().__init__(debug, statistics, name, back_en, **kwargs)

        self.crd_hold = CrdHold(debug=debug_sim, statistics=statistics)
        self.drop_1 = CrdDrop(debug=debug_sim, statistics=statistics)
        self.drop_2 = CrdDrop(debug=debug_sim, statistics=statistics)
        self.drop = RandomDropout(dimension=2, drop_probability=drop_prob, debug=debug_sim, statistics=statistics)
        self.done = False
        self.outer_crd = []
        self.inner_crd = []
        self.inner_ref = []
        self.curr_outer_crd = ''
        self.curr_inner_crd = ''
        self.curr_inner_ref = ''
        self.curr_crd0 = ''
        self.curr_crd1 = ''
        self.crd0 = []
        self.crd1 = []

    def update(self):
        self.update_done()

        if self.done:
            self.curr_outer_crd = ''
            self.curr_inner_crd = ''
            self.curr_inner_ref = ''
            return
        if len(self.outer_crd) > 0 and len(self.inner_crd) > 0 and len(self.inner_ref) > 0:
            self.curr_outer_crd = self.outer_crd.pop(0)
            self.curr_inner_crd = self.inner_crd.pop(0)
            self.curr_inner_ref = self.inner_ref.pop(0)
            self.curr_crd0 = self.crd0.pop(0)
            self.curr_crd1 = self.crd1.pop(0)

        self.crd_hold.set_outer_crd(self.curr_outer_crd)
        self.crd_hold.set_inner_crd(self.curr_inner_crd)

        self.drop.set_inner_crd(crd_hold.out_crd_inner())
        self.drop.set_outer_crd(crd_hold.out_crd_outer())
        self.drop.set_inner_ref(self.curr_inner_ref)

        self.drop_1.set_inner_crd(self.drop.out_crd(1))
        self.drop_1.set_outer_crd(self.curr_crd1)

        self.drop_2.set_inner_crd(self.drop_1.out_crd_outer())
        self.drop_2.set_outer_crd(self.curr_crd0)

        if self.drop.out_ref == 'D':
            self.done = True

        self.crd_hold.update()
        self.drop.update()
        self.drop_1.update()
        self.drop_2.update()

    def set_inner_crd(self, crd):
        if crd != '' and crd is not None:
            self.inner_crd.append(crd)

    def set_outer_crd(self, crd):
        if crd != '' and crd is not None:
            self.outer_crd.append(crd)

    def set_inner_ref(self, ref):
        if ref != '' and ref is not None:
            self.inner_ref.append(ref)

    def set_crd0(self, crd):
        if crd != '' and crd is not None:
            self.crd0.append(crd)

    def set_crd1(self, crd):
        if crd != '' and crd is not None:
            self.crd1.append(crd)

    def out_ref(self):
        return self.drop.out_ref()

    def out_crd_inner(self):
        return self.drop.out_crd(0)
    
    def out_crd_outer(self):
        return self.drop.out_crd(1)
    
    def out_crd0(self):
        return self.drop_2.out_crd_outer()
    
    def out_crd1(self):
        return self.drop_1.out_crd_outer()




