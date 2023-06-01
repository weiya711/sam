from .base import *
from .crd_manager import CrdPtConverter
from .accumulator_helpers import SparseCrdPtAccumulator2, SpAcc1NoCrdHold
from .crd_manager import CrdHold
from .token import StknDrop


# Accumulation into a vector
class SparseAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.maxdim = maxdim
        self.valtype = valtype
        self.last_level = last_level
        # Boolean flag of whether or not to output stop tokens on the output val stream
        self.val_stkn = val_stkn

        self.in_crd1 = []
        self.in_crd0 = []
        self.in_val = []

        self.crdhold_01 = CrdHold(**kwargs)

        self.crdhold_out1 = []
        self.crdhold_out0 = []

        self.spacc1_no_crdhold = SpAcc1NoCrdHold(maxdim=maxdim, valtype=valtype,
                                                 name=self.name + "_no_crdhold",
                                                 last_level=last_level, val_stkn=val_stkn,
                                                 **kwargs)

        self.sd_out1 = []
        self.sd_out0 = []
        self.sd_out_val = []

        self.stkndrop_crd1 = StknDrop(**kwargs)
        self.stkndrop_crd0 = StknDrop(**kwargs)
        self.stkndrop_val = StknDrop(**kwargs)

        self.curr_crd1 = None
        self.curr_crd0 = None
        self.curr_val = None

        if self.get_stats:
            self.in_crd0_fifo = 0
            self.in_crd1_fifo = 0
            self.in_val_fifo = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_inner = True
            self.fifo_avail_outer = True
            self.fifo_avail_val = True


    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    # FIXME: (owhsu) This code is unreachable
    def fifo_available(self, br=""):
        assert False
        if self.backpressure_en:
            if br == "inner":
                # and len(self.in_inner_crdpt) > self.depth:
                return self.fifo_avail_inner
            if br == "outer":  # and len(self.in_outer_crdpt) > self.depth:
                return self.fifo_avail_outer  # return False
            if br == "val":  # and len(self.in_val) > self.depth:
                return self.fifo_avail_val  # return False
            # return True
        return True

    def add_child(self, child=None, branch=""):
        if self.backpressure_en:
            if child is not None:
                self.backpressure.append(child)
                self.branch.append(branch)

    def update_ready(self):
        if self.backpressure_en:
            self.fifo_avail_outer = not (len(self.in_crd1) > self.depth)
            self.fifo_avail_inner = not (len(self.in_crd0) > self.depth)
            self.fifo_avail_val = not (len(self.in_val) > self.depth)

    def update(self):
        self.update_done()
        self.update_ready()

        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if self.debug:
                print(self.in_crd1, self.in_crd0, self.in_val)
                print(self.crdhold_out1, self.crdhold_out0)
                print(self.curr_crd1, self.curr_crd0, self.curr_val)
            if self.done:
                fch1, fch2 = self.crdhold_01.return_fifo()
                fs1, fs2, fs3 = self.spacc1_no_crdhold.return_fifo()
                fsd1 = self.stkndrop_crd1.return_fifo()
                fsd2 = self.stkndrop_crd0.return_fifo()
                fsd3 = self.stkndrop_val.return_fifo()
                self.crdhold_01 = CrdHold(fifos=[fch1, fch2], **self.kwargs)

                self.spacc1_no_crdhold = SpAcc1NoCrdHold(maxdim=self.maxdim, valtype=self.valtype,
                                                         name=self.name + "_no_crdhold",
                                                         fifos=[fs1, fs2, fs3], **self.kwargs)
                self.stkndrop_crd1 = StknDrop(fifos=[fsd1], **self.kwargs)
                self.stkndrop_crd0 = StknDrop(fifos=[fsd2], **self.kwargs)
                self.stkndrop_val = StknDrop(fifos=[fsd3], **self.kwargs)

            # FIXME: (owhsu) self.data_ready not defined in init
            if self.backpressure_en:
                self.data_ready = True

            # Set when block counts should start
            if len(self.in_crd1) > 0 or len(self.in_crd0) > 0 or len(self.in_val) > 0:
                self.block_start = False

            # What to do for drop tokens?
            if self.get_stats:
                pass

            if len(self.in_crd0) > 0:
                self.crdhold_01.set_inner_crd(self.in_crd0.pop(0))

            if len(self.in_crd1) > 0:
                self.crdhold_01.set_outer_crd(self.in_crd1.pop(0))

            self.crdhold_01.update()

            self.crdhold_out1.append(self.crdhold_01.out_crd_outer())
            self.crdhold_out0.append(self.crdhold_01.out_crd_inner())

            self.stkndrop_crd1.set_in_val(self.crdhold_01.out_crd_outer())
            self.stkndrop_crd0.set_in_val(self.crdhold_01.out_crd_inner())
            if len(self.in_val) > 0:
                self.stkndrop_val.set_in_val(self.in_val.pop(0))

            self.stkndrop_crd1.update()
            self.stkndrop_crd0.update()
            self.stkndrop_val.update()

            self.sd_out1.append(self.stkndrop_crd1.out_val())
            self.sd_out0.append(self.stkndrop_crd0.out_val())
            self.sd_out_val.append(self.stkndrop_val.out_val())

            self.spacc1_no_crdhold.set_in_crd1(self.stkndrop_crd1.out_val())
            self.spacc1_no_crdhold.set_in_crd0(self.stkndrop_crd0.out_val())
            self.spacc1_no_crdhold.set_val(self.stkndrop_val.out_val())

            self.spacc1_no_crdhold.update()

            self.curr_crd1 = self.spacc1_no_crdhold.out_crd1()
            self.curr_crd0 = self.spacc1_no_crdhold.out_crd0()
            self.curr_val = self.spacc1_no_crdhold.out_val()

            print(self.spacc1_no_crdhold.out_done(), self.stkndrop_crd1.out_done(),
                  self.stkndrop_crd0.out_done(), self.stkndrop_val.out_done(), self.crdhold_01.out_done())
            self.done = self.crdhold_01.out_done() and self.spacc1_no_crdhold.out_done() and \
                        self.stkndrop_crd1.out_done() and self.stkndrop_crd0.out_done() and self.stkndrop_val.out_done()

        if self.debug:
            print(self.in_crd1, self.in_crd0, self.in_val)
            print(self.crdhold_out1, self.crdhold_out0)
            print(self.curr_crd1, self.curr_crd0, self.curr_val)

    def set_in_crd0(self, crd, parent=None):
        if crd != '' and crd is not None:
            self.in_crd0.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def set_in_crd1(self, crd, parent=None):
        if crd != '' and crd is not None:
            self.in_crd1.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def set_val(self, val, parent=None):
        if val != '' and val is not None:
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_val)

    def out_crd1(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_crd1

    def out_crd0(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_crd0

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_val

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {}
            # TODO: Finish adding in other statistics
            stats_dict.update(self.spacc1_no_crdhold.return_statistics())
            stats_dict.update(self.crdhold_01.return_statistics())
            stats_dict.update(self.stkndrop_crd1.return_statistics())
            stats_dict.update(self.stkndrop_crd0.return_statistics())
            stats_dict.update(self.stkndrop_val.return_statistics())
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def print_fifos(self):
        print("Spaccumulator: None available")


# Accumulation into a matrix (2D)
class SparseAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.in1_crdpt = []
        self.in0_crdpt = []
        self.in_val = []

        self.crdpt_spacc = SparseCrdPtAccumulator2(maxdim=maxdim, valtype=valtype, **kwargs)
        self.crdpt_converter = CrdPtConverter(last_level=True, **kwargs)

        self.crdpt_spacc_out_val = []

        self.curr_1_crd = None
        self.curr_0_crd = None
        self.curr_val = None

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.val_stkn = val_stkn
        if self.get_stats:
            self.in1_fifo = 0
            self.in0_fifo = 0
            self.inval_fifo = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_inner = True
            self.fifo_avail_outer = True
            self.fifo_avail_val = True
        self.temp_maxdim = maxdim
        self.temp_valtype = valtype
        self.temp_last_level = last_level

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def add_child(self, child=None, branch=""):
        if self.backpressure_en:
            if child is not None:
                self.backpressure.append(child)
                self.branch.append(branch)

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in0_crdpt) > self.depth:
                self.fifo_avail_inner = False
            else:
                self.fifo_avail_inner = True
            if len(self.in1_crdpt) > self.depth:
                self.fifo_avail_outer = False
            else:
                self.fifo_avail_outer = True
            if len(self.in_val) > self.depth:
                self.fifo_avail_val = False
            else:
                self.fifo_avail_val = True

    def update(self):
        self.update_done()
        self.update_ready()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.done:
                f1, f2, f3 = self.crdpt_spacc.return_fifo()
                f4, f5 = self.crdpt_converter.return_fifo()
                self.crdpt_spacc = SparseCrdPtAccumulator2(maxdim=self.temp_maxdim, valtype=self.temp_valtype,
                                                           fifos=[f1, f2, f3])
                self.crdpt_converter = CrdPtConverter(last_level=self.temp_last_level, fifos=[f4, f5])
            if self.backpressure_en:
                self.data_valid = True
            if (len(self.in1_crdpt) > 0 or len(self.in0_crdpt) > 0 or len(self.in_val) > 0):
                self.block_start = False

            if self.get_stats:
                self.compute_fifo()

            if len(self.in1_crdpt) > 0:
                self.crdpt_spacc.set_outer_crdpt(self.in1_crdpt.pop(0))

            if len(self.in0_crdpt) > 0:
                self.crdpt_spacc.set_inner_crdpt(self.in0_crdpt.pop(0))

            if len(self.in_val) > 0:
                self.crdpt_spacc.set_val(self.in_val.pop(0))

            self.crdpt_spacc.update()

            self.crdpt_converter.set_inner_crdpt(self.crdpt_spacc.out_inner_crdpt())
            self.crdpt_converter.set_outer_crdpt(self.crdpt_spacc.out_outer_crdpt())
            self.crdpt_converter.update()

            if self.crdpt_spacc.out_val() != '':
                self.crdpt_spacc_out_val.append(self.crdpt_spacc.out_val())

            self.curr_1_crd = self.crdpt_converter.out_crd_outer()
            self.curr_0_crd = self.crdpt_converter.out_crd_inner()

            if self.val_stkn:
                self.curr_val = self.crdpt_spacc_out_val.pop(0) if isinstance(self.curr_0_crd, int) and \
                                                                   len(self.crdpt_spacc_out_val) > 0 else self.curr_0_crd
            else:
                self.curr_val = self.crdpt_spacc_out_val.pop(0) if len(self.crdpt_spacc_out_val) > 0 else ''

            if self.debug:
                print(self.in_val)

            self.done = self.crdpt_spacc.out_done() and self.crdpt_converter.out_done()

        if self.debug:
            print("Done:", self.done,
                  "\n SpCrdPt Accum Done:", self.crdpt_spacc.out_done(),
                  "\t CrdPtConv 01 Done:", self.crdpt_converter.out_done()
                  )

    def compute_fifo(self):
        self.in1_fifo = max(self.in1_fifo, len(self.in1_crdpt))
        self.in0_fifo = max(self.in0_fifo, len(self.in0_crdpt))
        self.inval_fifo = max(self.inval_fifo, len(self.in_val))

    def set_crd_inner(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in0_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_inner)

    def set_crd_outer(self, crdpt, parent=None):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in1_crdpt.append(crdpt)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_outer)

    def set_val(self, val, parent=None):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_val)

    def out_crd_outer(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_1_crd

    def out_crd_inner(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_0_crd

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            return self.curr_val

    def return_statistics(self):
        if self.get_stats:
            hits_info = self.crdpt_spacc.return_hits()
            stats_dict = {"in1_fifo": self.in1_fifo, "in0_fifo": self.in0_fifo,
                          "inval_fifo": self.inval_fifo, "max_hits": hits_info[0], "gt_one": hits_info[1],
                          "total_elems": hits_info[2]}
            stats_dict.update(self.crdpt_spacc.return_statistics())
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict