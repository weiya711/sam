from .base import *
from .repeater import RepeatSigGen, Repeat


class CrdMask(Primitive):
    def __init__(self, dimension=2, drop_predicate=lambda crds: False, **kwargs):
        # Will drop a coordinate if drop_predicate returns True
        # drop_predicate takes in some number of current coordinates and returns True/False to drop/not drop

        super().__init__(**kwargs)

        # TODO: innermost dimension is index 0. Perhaps the outermost dimension should be?
        self.dimension = dimension
        self.in_crd_array = [[] for i in range(self.dimension)]
        self.curr_crd_array = [None for i in range(self.dimension)]
        self.out_crd_array = ['' for i in range(self.dimension)]

        self.drop_predicate = drop_predicate

        # statistics info
        if self.get_stats:
            self.crd_fifos = [0 for i in range(self.dimension)]
            self.crd_drop_cnt = 0

    def update(self):
        if self.get_stats:
            self.crd_fifos = [max(self.crd_fifos[i], len[self.in_crd_array[i]]) for i in range(self.dimension)]

        if self.out_done():
            return

        print(self.in_crd_array, self.curr_crd_array, self.out_crd_array)

        # if len(self.in_crd_array) > 0:
        for i in range(self.dimension):
            if self.curr_crd_array[i] == None:
                # initialization: don't skip any
                self.curr_crd_array[i] = self.in_crd_array[i].pop(0)

            else: 
                self.curr_crd_array[i] = self.in_crd_array[i].pop(0)
                if not is_stkn(self.out_crd_array[i]):
                    # not a stop token: hold higher dimensions
                    break
        # else:
        #     return

        self.out_crd_array = self.curr_crd_array
        

        if self.out_crd_array[0] == 'D':
            self.done = True
            return

        if not is_stkn(self.out_crd_array[0]) :
            if self.drop_predicate(self.out_crd_array):
                # drop (may need to follow up with crd dropper?)
                self.out_crd_array = ['' for i in range(self.dimension)]

    def print_fifos(self):
        for i in range(self.dimension):
            print("CrdMask crd fifo ", i, " size: ", self.crd_fifos[i])

    def set_predicate(self, prob, drop_prob):
        self.drop_predicate=lambda crds: prob >= drop_prob

    def set_crd(self, dimension, crd):
        if crd != '' and crd is not None:
            self.in_crd_array[dimension].append(crd)

    def out_crd(self, dimension):
        return self.out_crd_array[dimension]

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"crd_fifos": self.crd_fifos, "drop_count": self.crd_drop_cnt}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


class RandomDropout(CrdMask):
    def __init__(self, dimension=2, drop_probability=0.5, **kwargs):
        super().__init__(dimension, lambda crds: random.random >= drop_probability, **kwargs)

class LowerTriangular2D(CrdMask):
    def __init__(self, **kwargs):
        super().__init__(2, lambda crds: crds[0] <= crds[1], **kwargs)

class UpperTriangular2D(CrdMask):
    def __init__(self, **kwargs):
        super().__init__(2, lambda crds: crds[0] >= crds[1], **kwargs)

class Diagonal2D(CrdMask):
    def __init__(self, **kwargs):
        super().__init__(2, lambda crds: crds[0] == crds[1], **kwargs)
