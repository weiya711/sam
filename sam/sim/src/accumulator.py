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
        print("FiFO Val size for Reduce block: ", self.in_val_size)


class SparseCrdPtAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, **kwargs):
        super().__init__(**kwargs)
        self.outer_crdpt = []
        self.inner_crdpt = []
        self.in_val = []

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
                assert self.curr_in_inner_crdpt == 'D' and self.curr_in_val, \
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
                len(self.crdpt_spacc_out_val) > 0 \
                else self.curr_inner_crd
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
