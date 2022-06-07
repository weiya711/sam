from .base import *
from .crd_manager import CrdPtConverter


class Reduce(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.curr_out = ""
        self.in_val_size = 0
        self.sum = 0
        self.emit_stkn = False
        self.curr_in_val = None

        self.reduction_count = 0
        self.num_inputs = 0
        self.num_outputs = 0
        self.stop_token_out = 0
        self.drop_token_out = 0
        self.valid_token_out = 0

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
                self.reduction_count += 1
                self.sum += self.curr_in_val
                self.curr_out = ""
        else:
            self.curr_out = ""
        if self.curr_out == "":
            self.drop_token_out += 1
        elif is_stkn(self.curr_out):
            self.stop_token_out += 1
        else:
            self.valid_token_out += 1

        self.compute_fifos()
        if self.debug:
            print("DEBUG: REDUCE:", "\t CurrIn:", self.curr_in_val, "\tCurrOut:", self.curr_out,
                  "\t Sum:", self.sum)

    def set_in_val(self, val):
        if val != '':
            self.num_inputs += 1
            self.in_val.append(val)

    def out_val(self):
        self.num_outputs += 1
        return self.curr_out

    def compute_fifos(self):
        self.in_val_size = max(self.in_val_size, len(self.in_val))

    def print_fifos(self):
        print("Reduction counts- total inputs ", self.num_inputs, " total outputs ", self.num_outputs,
              " reduction values ", self.reduction_count)
        print("FiFO Val size for Reduce block: ", self.in_val_size)

    def return_statistics(self):
        stats_dict = {"red_count": self.reduction_count, "total_inputs": self.num_inputs,
                      "total_outputs": self.num_outputs, "stpkn_outs": self.stop_token_out,
                      "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out}
        return stats_dict


class SparseCrdPtAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, **kwargs):
        super().__init__(**kwargs)
        self.outer_crdpt = []
        self.inner_crdpt = []
        self.in_val = []

        self.out_crd_fifo = 0
        self.in_crd_fifo = 0
        self.in_val_fifo = 0
        self.curr_in_val = None
        self.curr_in_inner_crdpt = None
        self.curr_in_outer_crdpt = None

        self.emit_output = []
        self.curr_inner_crdpt = ''
        self.curr_outer_crdpt = ''
        self.curr_val = ''

        # Maximum possible dimension for this index level
        self.maxdim = maxdim
        self.order = 1

        self.seen_done = False
        # Accumulation scratchpad storage
        self.storage = dict()
        self.valtype = valtype

    def update(self):
        self.out_crd_fifo = max(self.out_crd_fifo, len(self.outer_crdpt))
        self.in_crd_fifo = max(self.in_crd_fifo, len(self.inner_crdpt))
        self.in_val_fifo = max(self.in_val_fifo, len(self.in_val))
        if self.done:
            self.curr_outer_crd = ''
            self.curr_inner_crd = ''
        if self.done:
            self.curr_outer_crdpt = ''
            self.curr_inner_crdpt = ''
            self.curr_val = ''
            return

        if len(self.in_val) > 0 and len(self.outer_crdpt) > 0 and len(self.inner_crdpt) > 0:
            self.curr_in_val = self.in_val.pop(0)
            self.curr_in_inner_crdpt = self.inner_crdpt.pop(0)

            ocrd = self.outer_crdpt.pop(0)
            emit_output = ocrd != self.curr_in_outer_crdpt and self.curr_in_outer_crdpt is not None
            if emit_output:
                self.emit_output.append([self.curr_in_outer_crdpt, -1])
            self.curr_in_outer_crdpt = ocrd

            if self.curr_in_outer_crdpt in self.storage.keys():
                inner_dict = self.storage[self.curr_in_outer_crdpt]
                if self.curr_in_inner_crdpt in inner_dict.keys():
                    inner_dict[self.curr_in_inner_crdpt] += self.valtype(self.curr_in_val)
                else:
                    inner_dict[self.curr_in_inner_crdpt] = self.valtype(self.curr_in_val)
            # If a done token is seen, cannot emit done until all coordinates have been written out
            elif self.curr_in_outer_crdpt == 'D':
                assert self.curr_in_inner_crdpt == 'D' and self.curr_in_val == 'D', \
                    "If one item is a 'D' token, then all inputs must be"
                self.seen_done = True
            else:
                self.storage[self.curr_in_outer_crdpt] = {self.curr_in_inner_crdpt: self.valtype(self.curr_in_val)}

        if len(self.emit_output) > 0:
            fiber = self.emit_output[0]

            self.curr_outer_crdpt = fiber[0]

            self.curr_inner_crdpt = min(
                [item for item in self.storage[self.curr_outer_crdpt].keys() if item > fiber[1]])
            self.curr_val = self.storage[self.curr_outer_crdpt][self.curr_inner_crdpt]

            if not [item for item in self.storage[self.curr_outer_crdpt].keys() if item > self.curr_inner_crdpt]:
                self.emit_output.pop(0)
            else:
                self.emit_output[0][1] = self.curr_inner_crdpt
        elif self.seen_done:
            self.done = True
            self.seen_done = False
            self.curr_outer_crdpt = 'D'
            self.curr_inner_crdpt = 'D'
            self.curr_val = 'D'
        else:
            self.curr_outer_crdpt = ''
            self.curr_inner_crdpt = ''
            self.curr_val = ''

        if self.debug:
            print("Done:", self.out_done(),
                  "\n Curr in ocrd: ", self.curr_in_outer_crdpt, "\t Curr in icrd", self.curr_in_inner_crdpt,
                  "\t Curr in val", self.curr_in_val,
                  "\n Curr out ocrd: ", self.curr_outer_crdpt, "\t Curr out icrd: ", self.curr_inner_crdpt,
                  "\t Curr out val: ", self.curr_val,
                  "\n Emit crds: ", self.emit_output,
                  "\n Storage: ", self.storage)

    def set_inner_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.inner_crdpt.append(crdpt)

    def set_outer_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.outer_crdpt.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '':
            self.in_val.append(val)

    def out_outer_crdpt(self):
        return self.curr_outer_crdpt

    def out_inner_crdpt(self):
        return self.curr_inner_crdpt

    def out_val(self):
        return self.curr_val


# Accumulation into a vector
class SparseAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, **kwargs):
        super().__init__(**kwargs)
        self.in_outer_crdpt = []
        self.in_inner_crdpt = []
        self.in_val = []

        self.in_outer_crd_pt_fifo = 0
        self.in_inner_crd_pt_fifo = 0
        self.in_val_fifo = 0

        self.crdpt_spacc = SparseCrdPtAccumulator1(maxdim=maxdim, valtype=valtype, **kwargs)
        self.crdpt_converter = CrdPtConverter(last_level=last_level, **kwargs)

        self.crdpt_spacc_out_val = []

        self.curr_outer_crd = None
        self.curr_inner_crd = None
        self.curr_val = None

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.val_stkn = val_stkn

    def update(self):
        # What to do for drop tokens?
        self.in_outer_crd_pt_fifo = max(self.in_outer_crd_pt_fifo, len(self.in_outer_crdpt))
        self.in_inner_crd_pt_fifo = max(self.in_inner_crd_pt_fifo, len(self.in_inner_crdpt))
        self.in_val_fifo = max(self.in_val_fifo, len(self.in_val))

        if len(self.in_outer_crdpt) > 0:
            self.crdpt_spacc.set_outer_crdpt(self.in_outer_crdpt.pop(0))

        if len(self.in_inner_crdpt) > 0:
            self.crdpt_spacc.set_inner_crdpt(self.in_inner_crdpt.pop(0))

        if len(self.in_val) > 0:
            self.crdpt_spacc.set_val(self.in_val.pop(0))

        self.crdpt_spacc.update()

        self.crdpt_converter.set_outer_crdpt(self.crdpt_spacc.out_outer_crdpt())
        self.crdpt_converter.set_inner_crdpt(self.crdpt_spacc.out_inner_crdpt())

        if self.crdpt_spacc.out_val() != '':
            self.crdpt_spacc_out_val.append(self.crdpt_spacc.out_val())

        self.crdpt_converter.update()
        self.curr_outer_crd = self.crdpt_converter.out_crd_outer()
        self.curr_inner_crd = self.crdpt_converter.out_crd_inner()

        if self.val_stkn:
            self.curr_val = self.crdpt_spacc_out_val.pop(0) if isinstance(self.curr_inner_crd, int) and \
                len(self.crdpt_spacc_out_val) > 0 else self.curr_inner_crd
        else:
            self.curr_val = self.crdpt_spacc_out_val.pop(0) if len(self.crdpt_spacc_out_val) > 0 else ''

        if self.debug:
            print(self.in_val)

        self.done = self.crdpt_spacc.out_done() and self.crdpt_converter.out_done()

        if self.debug:
            print("Done:", self.done,
                  "\n SpCrdPt Accum Done:", self.crdpt_spacc.out_done(),
                  "\t CrdPtConverter Done:", self.crdpt_converter.out_done()
                  )

    def set_inner_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_inner_crdpt.append(crdpt)

    def set_outer_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_outer_crdpt.append(crdpt)

    def crd_in_inner(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_inner_crdpt.append(crdpt)

    def crd_in_outer(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_outer_crdpt.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '':
            self.in_val.append(val)

    def out_outer_crd(self):
        return self.curr_outer_crd

    def out_inner_crd(self):
        return self.curr_inner_crd

    def out_val(self):
        return self.curr_val

    def out_crd_outer(self):
        return self.curr_outer_crd

    def out_crd_inner(self):
        return self.curr_inner_crd

    def return_statistics(self):
        stats_dict = {}
        stats_dict["in_outer_fifo"] = self.in_outer_crd_pt_fifo
        stats_dict["in_inner_fifo"] = self.in_inner_crd_pt_fifo
        stats_dict["in_val_fifo"] = self.in_val_fifo
        return stats_dict

    def print_fifos(self):
        print("Spaccumulator: None available")


class SparseCrdPtAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, **kwargs):
        super().__init__(**kwargs)
        self.in_crdpt0 = []
        self.in_crdpt1 = []
        self.in_crdpt2 = []
        self.in_val = []

        self.curr_in_val = None
        self.curr_in0_crdpt = None
        self.curr_in1_crdpt = None
        self.curr_in2_crdpt = None

        self.emit_output = []
        self.curr_crdpt0 = ''
        self.curr_crdpt1 = ''
        self.curr_crdpt2 = ''
        self.curr_val = ''

        # Maximum possible dimension for this index level
        self.maxdim = maxdim
        self.order = 1

        self.seen_done = False
        # Accumulation scratchpad storage
        self.storage = dict()
        self.valtype = valtype

    def update(self):
        if self.done:
            self.curr_crdpt0 = ''
            self.curr_crdpt1 = ''
            self.curr_crdpt2 = ''
            self.curr_val = ''
            return

        if len(self.in_val) > 0 and len(self.in_crdpt1) > 0 and len(self.in_crdpt0) > 0 and len(self.in_crdpt2) > 0:
            self.curr_in_val = self.in_val.pop(0)
            self.curr_in0_crdpt = self.in_crdpt0.pop(0)
            self.curr_in1_crdpt = self.in_crdpt1.pop(0)

            crd2 = self.in_crdpt2.pop(0)
            emit_output = crd2 != self.curr_in2_crdpt and self.curr_in2_crdpt is not None
            if emit_output:
                self.emit_output.append([self.curr_in2_crdpt, (-1, -1)])
            self.curr_in2_crdpt = crd2

            key = (self.curr_in1_crdpt, self.curr_in0_crdpt)
            if self.curr_in2_crdpt in self.storage.keys():
                inner_dict = self.storage[self.curr_in2_crdpt]
                if key in inner_dict.keys():
                    inner_dict[key] += self.valtype(self.curr_in_val)
                else:
                    inner_dict[key] = self.valtype(self.curr_in_val)

            # If a done token is seen, cannot emit done until all coordinates have been written out
            elif self.curr_in2_crdpt == 'D':
                assert self.curr_in1_crdpt == 'D' and self.curr_in0_crdpt == 'D' and self.curr_in_val == 'D', \
                    "If one item is a 'D' token, then all inputs must be"
                self.seen_done = True
            else:
                self.storage[self.curr_in2_crdpt] = {key: self.valtype(self.curr_in_val)}

        if len(self.emit_output) > 0:
            fiber = self.emit_output[0]

            self.curr_crdpt2 = fiber[0]

            key = min(
                [item for item in self.storage[self.curr_crdpt2].keys() if item > fiber[1]])
            (self.curr_crdpt1, self.curr_crdpt0) = key
            self.curr_val = self.storage[self.curr_crdpt2][key]

            if not [item for item in self.storage[self.curr_crdpt2].keys() if item > key]:
                self.emit_output.pop(0)
            else:
                self.emit_output[0][1] = key
        elif self.seen_done:
            self.done = True
            self.seen_done = False
            self.curr_crdpt0 = 'D'
            self.curr_crdpt1 = 'D'
            self.curr_crdpt2 = 'D'
            self.curr_val = 'D'
        else:
            self.curr_crdpt0 = ''
            self.curr_crdpt1 = ''
            self.curr_crdpt2 = ''
            self.curr_val = ''

        if self.debug:
            print("Done:", self.out_done(),
                  "\n Curr in crd2: ", self.curr_in2_crdpt, "\t Curr in crd1: ", self.curr_in1_crdpt,
                  "\t Curr in crd0", self.curr_in0_crdpt,
                  "\t Curr in val", self.curr_in_val,
                  "\n Curr out crd2: ", self.curr_crdpt2, "\t Curr out crd1: ", self.curr_crdpt1,
                  "\t Curr out crd0: ", self.curr_crdpt0,
                  "\t Curr out val: ", self.curr_val,
                  "\n Emit crds: ", self.emit_output,
                  "\n Storage: ", self.storage)

    def set_0_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_crdpt0.append(crdpt)

    def set_1_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_crdpt1.append(crdpt)

    def set_2_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in_crdpt2.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '':
            self.in_val.append(val)

    def out_2_crdpt(self):
        return self.curr_crdpt2

    def out_1_crdpt(self):
        return self.curr_crdpt1

    def out_0_crdpt(self):
        return self.curr_crdpt0

    def out_val(self):
        return self.curr_val


# Accumulation into a matrix (2D)
class SparseAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, **kwargs):
        super().__init__(**kwargs)
        self.in2_crdpt = []
        self.in1_crdpt = []
        self.in0_crdpt = []
        self.in_val = []

        self.in1_fifo = 0
        self.in2_fifo = 0
        self.in0_fifo = 0
        self.inval_fifo = 0

        self.crdpt_spacc = SparseCrdPtAccumulator2(maxdim=maxdim, valtype=valtype, **kwargs)
        self.crdpt_converter01 = CrdPtConverter(last_level=True, **kwargs)
        self.crdpt_converter12 = CrdPtConverter(last_level=False, **kwargs)

        self.crdpt_spacc_out_val = []

        self.curr_2_crd = None
        self.curr_1_crd = None
        self.curr_0_crd = None
        self.curr_val = None

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.val_stkn = val_stkn

    def update(self):
        self.compute_fifo()
        if len(self.in2_crdpt) > 0:
            self.crdpt_spacc.set_2_crdpt(self.in2_crdpt.pop(0))

        if len(self.in1_crdpt) > 0:
            self.crdpt_spacc.set_1_crdpt(self.in1_crdpt.pop(0))

        if len(self.in0_crdpt) > 0:
            self.crdpt_spacc.set_0_crdpt(self.in0_crdpt.pop(0))

        if len(self.in_val) > 0:
            self.crdpt_spacc.set_val(self.in_val.pop(0))

        self.crdpt_spacc.update()

        self.crdpt_converter12.set_inner_crdpt(self.crdpt_spacc.out_1_crdpt())
        self.crdpt_converter12.set_outer_crdpt(self.crdpt_spacc.out_2_crdpt())
        self.crdpt_converter12.update()

        self.crdpt_converter01.set_inner_crdpt(self.crdpt_spacc.out_0_crdpt())
        self.crdpt_converter01.set_outer_crdpt(self.crdpt_converter12.out_crd_inner())
        self.crdpt_converter01.update()

        if self.crdpt_spacc.out_val() != '':
            self.crdpt_spacc_out_val.append(self.crdpt_spacc.out_val())

        self.curr_2_crd = self.crdpt_converter12.out_crd_outer()
        self.curr_1_crd = self.crdpt_converter01.out_crd_outer()
        self.curr_0_crd = self.crdpt_converter01.out_crd_inner()

        if self.val_stkn:
            self.curr_val = self.crdpt_spacc_out_val.pop(0) if isinstance(self.curr_0_crd, int) and \
                len(self.crdpt_spacc_out_val) > 0 else self.curr_0_crd
        else:
            self.curr_val = self.crdpt_spacc_out_val.pop(0) if len(self.crdpt_spacc_out_val) > 0 else ''

        if self.debug:
            print(self.in_val)

        self.done = self.crdpt_spacc.out_done() and self.crdpt_converter01.out_done() \
            and self.crdpt_converter12.out_done()

        if self.debug:
            print("Done:", self.done,
                  "\n SpCrdPt Accum Done:", self.crdpt_spacc.out_done(),
                  "\t CrdPtConv 01 Done:", self.crdpt_converter01.out_done(),
                  "\t CrdPtConv 12 Done:", self.crdpt_converter01.out_done()
                  )

    def compute_fifo(self):
        self.in1_fifo = max(self.in1_fifo, len(self.in1_crdpt))
        self.in2_fifo = max(self.in2_fifo, len(self.in2_crdpt))
        self.in0_fifo = max(self.in0_fifo, len(self.in0_crdpt))
        self.inval_fifo = max(self.inval_fifo, len(self.in_val))

    def set_0_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in0_crdpt.append(crdpt)

    def set_1_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in1_crdpt.append(crdpt)

    def set_2_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in2_crdpt.append(crdpt)

    def crd_in_0(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in0_crdpt.append(crdpt)

    def crd_in_1(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in1_crdpt.append(crdpt)

    def crd_in_2(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '':
            self.in2_crdpt.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '':
            self.in_val.append(val)

    def out_2_crd(self):
        return self.curr_2_crd

    def out_1_crd(self):
        return self.curr_1_crd

    def out_0_crd(self):
        return self.curr_0_crd

    def out_crd_2(self):
        return self.curr_2_crd

    def out_crd_1(self):
        return self.curr_1_crd

    def out_crd_0(self):
        return self.curr_0_crd

    def out_val(self):
        return self.curr_val

    def return_statistics(self):
        stats_dict = {"in1_fifo": self.in1_fifo, "in2_fifo": self.in2_fifo, "in0_fifo": self.in0_fifo,
                      "inval_fifo": self.inval_fifo}
        return stats_dict
