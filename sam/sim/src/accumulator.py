import enum

from .base import *


class Reduce(Primitive):
    def __init__(self, depth=1, **kwargs):
        super().__init__(**kwargs)

        self.in_val = []
        self.curr_out = ""
        self.in_val_size = 0
        self.sum = 0
        self.emit_stkn = False
        self.curr_in_val = None

        if self.get_stats:
            self.reduction_count = 0
            self.num_inputs = 0
            self.num_outputs = 0
            self.stop_token_out = 0
            self.drop_token_out = 0
            self.valid_token_out = 0
            self.zero_out = 0
            self.nonzero_out = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail = True

    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def fifo_available(self, br=""):
        if self.backpressure_en:
            return self.fifo_avail
        return True

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in_val) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def add_child(self, child=None, branch=""):
        if self.backpressure_en and child is not None:
            self.backpressure.append(child)
            self.branch.append(branch)

    def update(self):
        self.update_ready()
        self.update_done()
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            if len(self.in_val) > 0:
                self.block_start = False

            curr_in_val = ""
            if self.done:
                self.curr_out = ""
                # Reset state
                self.in_val_size = 0
                self.sum = 0
                self.emit_stkn = False
                self.done = False
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
                    if self.get_stats:
                        self.reduction_count += 1
                    self.sum += self.curr_in_val
                    self.curr_out = ""
            else:
                self.curr_out = ""

            if self.get_stats:
                if self.curr_out == "":
                    self.drop_token_out += 1
                elif is_stkn(self.curr_out):
                    self.stop_token_out += 1
                else:
                    if (isinstance(self.curr_out, float) or isinstance(self.curr_out, int)) and self.curr_out == 0:
                        self.zero_out += 1
                    else:
                        self.nonzero_out += 1
                    self.valid_token_out += 1

            if self.get_stats:
                self.compute_fifos()
        if self.debug:
            print("DEBUG: REDUCE:", "\t CurrIn:", self.curr_in_val, "\tCurrOut:", self.curr_out,
                  "\t Sum:", self.sum)

    def set_in_val(self, val, parent=None):
        if val != '' and val is not None:
            if self.get_stats:
                self.num_inputs += 1
            self.in_val.append(val)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail)

    def out_val(self):
        if (self.backpressure_en and self.data_valid) or not self.backpressure_en:
            if self.get_stats:
                self.num_outputs += 1
            return self.curr_out

    def compute_fifos(self):
        self.in_val_size = max(self.in_val_size, len(self.in_val))

    def print_fifos(self):
        if self.get_stats:
            print("Reduction counts- total inputs ", self.num_inputs, " total outputs ", self.num_outputs,
                  " reduction values ", self.reduction_count)
            print("FiFO Val size for Reduce block: ", self.in_val_size)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"red_count": self.reduction_count, "total_inputs": self.num_inputs,
                          "total_outputs": self.num_outputs, "stkn_outs": self.stop_token_out,
                          "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out,
                          "zero_outs": self.zero_out, "nonzero_outs": self.nonzero_out}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

class SparseCrdPtAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, fifos=None, **kwargs):
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

        if fifos is not None and len(fifos) == 3:
            self.outer_crdpt = fifos[0]
            self.inner_crdpt = fifos[1]
            self.in_val = fifos[2]

        if self.get_stats:
            self.hits_tracker = {}
            self.stop_token_out = 0
            self.drop_token_out = 0
            self.valid_token_out = 0
            self.zero_out = 0
            self.nonzero_out = 0
            self.out_crd_fifo = 0
            self.in_crd_fifo = 0
            self.in_val_fifo = 0

    def return_fifo(self):
        return self.outer_crdpt, self.inner_crdpt, self.in_val

    def update(self):
        self.update_done()
        if self.debug:
            if self.seen_done or self.done:
                print(self.seen_done, self.done)
                print("@@@", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()
            if len(self.in_val) > 0 and self.in_val[0] == "D":
                print("val", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()
            if len(self.inner_crdpt) > 0 and self.inner_crdpt[0] == "D":
                print("innercrd", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()
            if len(self.outer_crdpt) > 0 and self.outer_crdpt[0] == "D":
                print("outercrd", self.outer_crdpt, self.inner_crdpt, self.in_val, self.emit_output,
                      self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_val)
                self.print_debug()

        if len(self.outer_crdpt) > 0 or len(self.inner_crdpt) > 0:
            self.block_start = False

        if self.get_stats:
            self.out_crd_fifo = max(self.out_crd_fifo, len(self.outer_crdpt))
            self.in_crd_fifo = max(self.in_crd_fifo, len(self.inner_crdpt))
            self.in_val_fifo = max(self.in_val_fifo, len(self.in_val))

        if self.done:
            self.curr_outer_crdpt = ''
            self.curr_inner_crdpt = ''
            self.curr_val = ''
            if self.get_stats:
                self.drop_token_out += 1
            return

        if len(self.in_val) > 0 and len(self.outer_crdpt) > 0 and len(self.inner_crdpt) > 0 and not self.seen_done:
            self.curr_in_val = self.in_val.pop(0)
            self.curr_in_inner_crdpt = self.inner_crdpt.pop(0)

            ocrd = self.outer_crdpt.pop(0)
            # if self.curr_in_val == 'D':
            #     print(self.curr_in_val, self.curr_in_inner_crdpt, ocrd)
            #     assert self.curr_in_val == "D" and self.curr_in_inner_crdpt == "D" and ocrd == "D"
            #     print("######", ocrd,  self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.emit_output)
            # print(self.in_val, self.outer_crdpt, self.inner_crdpt, ocrd
            # self.curr_in_outer_crdpt, self.curr_in_inner_crdpt, self.curr_in_val)
            emit_output = ocrd != self.curr_in_outer_crdpt and self.curr_in_outer_crdpt is not None and \
                self.curr_in_outer_crdpt != "D"
            if emit_output:
                self.emit_output.append([self.curr_in_outer_crdpt, -1])
                # print("@@@@@", self.curr_in_outer_crdpt)
            self.curr_in_outer_crdpt = ocrd
            if self.curr_in_outer_crdpt in self.storage.keys():
                inner_dict = self.storage[self.curr_in_outer_crdpt]
                if self.get_stats:
                    for k in inner_dict.keys():
                        self.hits_tracker[k] = 1
                if self.curr_in_inner_crdpt in inner_dict.keys():
                    if self.get_stats:
                        self.hits_tracker[self.curr_in_inner_crdpt] += 1
                    inner_dict[self.curr_in_inner_crdpt] += self.valtype(self.curr_in_val)
                else:
                    if self.get_stats:
                        self.hits_tracker[self.curr_in_inner_crdpt] = 1
                    inner_dict[self.curr_in_inner_crdpt] = self.valtype(self.curr_in_val)
            # If a done token is seen, cannot emit done until all coordinates have been written out
            elif self.curr_in_outer_crdpt == 'D':
                assert self.curr_in_inner_crdpt == 'D' and self.curr_in_val == 'D', \
                    "If one item is a 'D' token, then all inputs must be"
                self.seen_done = True
            else:
                self.storage[self.curr_in_outer_crdpt] = {self.curr_in_inner_crdpt: self.valtype(self.curr_in_val)}
        # if self.curr_in_outer_crdpt == "D":
        #     print("__________", self.emit_output, self.seen_done)

        if len(self.emit_output) > 0:
            fiber = self.emit_output[0]

            self.curr_outer_crdpt = fiber[0]
            # print("===, ", self.storage)
            # print(fiber)
            # print(self.emit_output)
            # print(self.storage[self.curr_outer_crdpt].keys(), fiber[1])
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
        if self.get_stats:
            if self.curr_val == "":
                self.drop_token_out += 1
            elif is_stkn(self.curr_val):
                self.stop_token_out += 1
            else:
                if (isinstance(self.curr_val, float) or isinstance(self.curr_val, int)) and self.curr_val == 0:
                    self.zero_out += 1
                else:
                    self.nonzero_out += 1
                self.valid_token_out += 1

        if self.debug:
            print("Done ptaccum:", self.out_done(), self.done,
                  "\n Curr in ocrd: ", self.curr_in_outer_crdpt, "\t Curr in icrd", self.curr_in_inner_crdpt,
                  "\t Curr in val", self.curr_in_val,
                  "\n Curr out ocrd: ", self.curr_outer_crdpt, "\t Curr out icrd: ", self.curr_inner_crdpt,
                  "\t Curr out val: ", self.curr_val,
                  "\n Emit crds: ", self.emit_output,
                  "\n Storage: ", self.storage,
                  "\n f: ", self.outer_crdpt, self.inner_crdpt, self.in_val)

    def print_debug(self):
        print("Crdptaccum_debug Done:", self.out_done(), self.done,
              "\n Curr in ocrd: ", self.curr_in_outer_crdpt, "\t Curr in icrd", self.curr_in_inner_crdpt,
              "\t Curr in val", self.curr_in_val,
              "\n Curr out ocrd: ", self.curr_outer_crdpt, "\t Curr out icrd: ", self.curr_inner_crdpt,
              "\t Curr out val: ", self.curr_val,
              "\n Emit crds: ", self.emit_output,
              "\n Storage: ", self.storage,
              "\n Fifos: ", self.outer_crdpt, self.inner_crdpt, self.in_val)

    def set_inner_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.inner_crdpt.append(crdpt)

    def set_outer_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.outer_crdpt.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)

    def out_outer_crdpt(self):
        return self.curr_outer_crdpt

    def out_inner_crdpt(self):
        return self.curr_inner_crdpt

    def out_val(self):
        return self.curr_val

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"stkn_outs": self.stop_token_out,
                          "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out,
                          "zero_outs": self.zero_out, "nonzero_outs": self.nonzero_out}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def return_hits(self):
        i = 0
        cnt_gt_zero = 0
        cnt_total = 0
        total_sum = 0
        if self.get_stats:
            for k in self.hits_tracker.keys():
                if self.hits_tracker[k] > i:
                    i = self.hits_tracker[k]
                if self.hits_tracker[k] > 1:
                    cnt_gt_zero += 1
                total_sum += self.hits_tracker[k]
                cnt_total += 1
        return i, cnt_gt_zero, cnt_total, total_sum


# NEW VERSION: Accumulation into a vector
class SparseAccumulator1(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.maxdim = maxdim
        self.valtype = valtype
        self.last_level = last_level
        # Boolean flag for whether to output stop tokens on the output val stream
        self.val_stkn = val_stkn

        self.in_crd1 = []
        self.in_crd0 = []
        self.in_val = []

        self.storage = {}

        self.curr_crd1 = None
        self.curr_crd0 = None
        self.curr_val = None

        self.curr_in_crd0 = None
        self.curr_in_val = None

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

        self.states = enum.Enum('States', ['READY', 'ACC', 'WR', 'DONE'])
        self.curr_state = self.states.READY
        self.next_state = self.states.READY
        self.writeout = False
        self.writeout_storage = []

        self.seen_done = False
        self.crd1_stkn = None

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

    def ACC_body(self):
        self.curr_in_crd0 = self.in_crd0.pop(0)
        self.curr_in_val = self.in_val.pop(0)
        # In accumulation, accumulate into memory
        if is_nc_tkn(self.curr_in_val, self.valtype):
            assert is_nc_tkn(self.curr_in_crd0, int), "The inner coordinate must be a non-control token"
            if self.curr_in_crd0 in self.storage.keys():
                # Coordinate is in storage, so accumulate
                self.storage[self.curr_in_crd0] += self.curr_in_val
            else:
                # Coordinate is not in storage, so add it in
                self.storage[self.curr_in_crd0] = self.curr_in_val
            self.next_state = self.states.ACC
        # In accumulation, if you see a stop token in the inner level, go back to start
        elif is_stkn(self.curr_in_crd0):
            assert is_stkn(self.curr_in_val) and stkn_order(self.curr_in_crd0) == stkn_order(self.curr_in_val), \
                "Stop tokens must match for inner crd: " + str(self.curr_in_crd0) + " and val: " + str(self.curr_in_val)
            self.next_state = self.states.READY
        elif is_dtkn(self.curr_in_crd0):
            assert False, "Shouldn't have done token for coordinates if in accumulate (ACC) state"
        # Do nothing
        else:
            self.next_state = self.states.ACC

    def update(self):
        self.update_done()
        self.update_ready()

        # Print out debugging statements
        if self.debug:
            print("========== " + self.name + " SPACC1 (NEW) ==========")
            print("Inputs: ", self.in_crd1, self.in_crd0, self.in_val)
            print("Temps: ", self.curr_crd1, self.crd1_stkn, self.curr_in_crd0, self.curr_in_val)
            print("Store/Wr: ", self.storage, self.writeout_storage, self.writeout)
            print("Outputs: ", self.curr_crd0, self.curr_val)
            print("State: ", self.curr_state)

        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            # Update state of state machine to next state
            self.curr_state = self.next_state

            # FIXME: (owhsu) self.data_ready not defined in init
            if self.backpressure_en:
                self.data_valid = True
                self.data_ready = True

            if self.done:
                # Reset things
                # fch1, fch2 = self.crdhold_01.return_fifo()
                # fs1, fs2, fs3 = self.spacc1_no_crdhold.return_fifo()
                # fsd1 = self.stkndrop_crd1.return_fifo()
                # fsd2 = self.stkndrop_crd0.return_fifo()
                # fsd3 = self.stkndrop_val.return_fifo()
                # self.crdhold_01 = CrdHold(fifos=[fch1, fch2], **self.kwargs)
                #
                # self.spacc1_no_crdhold = SpAcc1NoCrdHold(maxdim=self.maxdim, valtype=self.valtype,
                #                                          name=self.name + "_no_crdhold",
                #                                          fifos=[fs1, fs2, fs3], **self.kwargs)
                # self.stkndrop_crd1 = StknDrop(fifos=[fsd1], **self.kwargs)
                # self.stkndrop_crd0 = StknDrop(fifos=[fsd2], **self.kwargs)
                # self.stkndrop_val = StknDrop(fifos=[fsd3], **self.kwargs)
                pass

            # Set when block counts should start
            if len(self.in_crd1) > 0 or len(self.in_crd0) > 0 or len(self.in_val) > 0:
                self.block_start = False

            # Store block statistics if asked to be reported
            if self.get_stats:
                # What to do for drop tokens?
                pass

            # Begin state machine computation
            if self.curr_state == self.states.READY:
                if len(self.in_crd1) > 0:
                    self.curr_crd1 = self.in_crd1.pop(0)
                    if is_nc_tkn(self.curr_crd1):
                        if len(self.in_crd0) > 0 and len(self.in_val) > 0:
                            self.ACC_body()
                        else:
                            self.next_state = self.states.ACC
                    elif is_stkn(self.curr_crd1):
                        # TODO: what to do when we want to writeout but writeout isn't done

                        # Set writeout to be true, move over storage, and clear it.
                        self.crd1_stkn = self.curr_crd1
                        self.writeout = True
                        self.writeout_storage = [item for item in sorted(self.storage.items())]
                        self.storage = {}
                        self.next_state = self.states.READY
                    elif is_dtkn(self.curr_crd1):
                        self.seen_done = True
                        if self.writeout:
                            self.next_state = self.states.WR
                        else:
                            self.next_state = self.states.DONE
                    else:
                        assert False, "Cannot have a coordinate token of this type: " + str(self.curr_crd1)
            # Handle accumulation into storage
            elif self.curr_state == self.states.ACC:
                if len(self.in_crd0) > 0 and len(self.in_val) > 0:
                    self.ACC_body()
            # Finish writeout and then be done.
            elif self.curr_state == self.states.WR:
                if not self.writeout:
                    self.next_state = self.states.DONE
                else:
                    self.next_state = self.states.WR
            # See a done signal AND writeout is done
            # Stay done until reset
            elif self.curr_state == self.states.DONE:
                self.done = True
                self.next_state = self.states.DONE
            # Default to the current state
            else:
                self.next_state = self.curr_state

        # Writeout is true, so writeout the current values for storage
        if self.writeout:
            # Writeout is done when there are no elements left
            if len(self.writeout_storage) == 0:
                self.writeout = False
                assert self.crd1_stkn is not None, "The current writeout stop token should not be None"
                self.curr_crd0 = self.crd1_stkn
                self.curr_val = self.crd1_stkn
            else:
                curr_writeout_elem = self.writeout_storage.pop(0)
                self.curr_val = curr_writeout_elem[1]
                self.curr_crd0 = curr_writeout_elem[0]
        elif self.done:
            self.curr_val = 'D'
            self.curr_crd0 = 'D'
        else:
            self.curr_val = ""
            self.curr_crd0 = ""

        # Update the current state
        self.curr_state = self.next_state

        if self.debug:
            print(self.in_crd1, self.in_crd0, self.in_val)
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
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def print_fifos(self):
        print("Spaccumulator: None available")

class SparseCrdPtAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, **kwargs):
        super().__init__(**kwargs)
        self.in_crdpt0 = []
        self.in_crdpt1 = []
        self.in_val = []

        self.curr_in_val = None
        self.curr_in0_crdpt = None
        self.curr_in1_crdpt = None

        self.emit_output = []
        self.curr_crdpt0 = ''
        self.curr_crdpt1 = ''
        self.curr_val = ''

        # Maximum possible dimension for this index level
        self.maxdim = maxdim
        self.order = 1

        self.seen_done = False
        # Accumulation scratchpad storage
        self.storage = dict()
        self.valtype = valtype

        if self.get_stats:
            self.hits_tracker = {}
            self.stop_token_out = 0
            self.drop_token_out = 0
            self.valid_token_out = 0
            self.zero_out = 0
            self.nonzero_out = 0

    def return_fifo(self):
        return self.in_crdpt0, self.in_crdpt1, self.in_val

    def update(self):
        self.update_done()
        if len(self.in_crdpt0) > 0 or len(self.in_crdpt0) > 0 or len(self.in_val) > 0:
            self.block_start = False

        if self.done:
            self.curr_crdpt0 = ''
            self.curr_crdpt1 = ''
            self.curr_val = ''
            if self.get_stats:
                self.drop_token_out += 1
            return

        if len(self.in_val) > 0 and len(self.in_crdpt1) > 0 and len(self.in_crdpt0) > 0:
            self.curr_in_val = self.in_val.pop(0)
            self.curr_in0_crdpt = self.in_crdpt0.pop(0)
            self.curr_in1_crdpt = self.in_crdpt1.pop(0)

            emit_output = self.curr_in1_crdpt == 'D'
            if emit_output:
                self.emit_output.append([-1, -1])
                assert self.curr_in1_crdpt == 'D' and self.curr_in0_crdpt == 'D' and self.curr_in_val == 'D', \
                    "If one item is a 'D' token, then all inputs must be"
                self.seen_done = True
            else:
                if self.curr_in1_crdpt in self.storage.keys():
                    inner_dict = self.storage[self.curr_in1_crdpt]
                    if self.get_stats:
                        for k in inner_dict.keys():
                            self.hits_tracker[k] = 1
                    if self.curr_in0_crdpt in inner_dict.keys():
                        if self.get_stats:
                            self.hits_tracker[self.curr_in0_crdpt] += 1
                        inner_dict[self.curr_in0_crdpt] += self.valtype(self.curr_in_val)
                    else:
                        if self.get_stats:
                            self.hits_tracker[self.curr_in0_crdpt] = 1
                        inner_dict[self.curr_in0_crdpt] = self.valtype(self.curr_in_val)
                else:
                    self.storage[self.curr_in1_crdpt] = {self.curr_in0_crdpt: self.valtype(self.curr_in_val)}

        if len(self.emit_output) > 0:
            fiber = self.emit_output.pop(0)
            #
            key1 = min(
                [item for item in self.storage.keys() if item > fiber[0]])
            key0 = min(
                [item for item in self.storage[key1].keys() if item > fiber[1]])

            self.curr_crdpt1 = key1
            self.curr_crdpt0 = key0
            self.curr_val = self.storage[key1][key0]

            # Finished inner coordinates, increment outer coordinate
            if not [item for item in self.storage[key1].keys() if item > key0]:
                # Do not increment outer coordinate if it's the last one
                if [item for item in self.storage.keys() if item > key1]:
                    self.emit_output.append([key1, -1])
            # Do inner coordinates
            else:
                self.emit_output.append([fiber[0], key0])

        elif self.seen_done:
            self.done = True
            self.seen_done = False
            self.curr_crdpt0 = 'D'
            self.curr_crdpt1 = 'D'
            self.curr_val = 'D'
        else:
            self.curr_crdpt0 = ''
            self.curr_crdpt1 = ''
            self.curr_val = ''

        if self.get_stats:
            if self.curr_val == "":
                self.drop_token_out += 1
            elif is_stkn(self.curr_val):
                self.stop_token_out += 1
            else:
                if (isinstance(self.curr_val, float) or isinstance(self.curr_val, int)) and self.curr_val == 0:
                    self.zero_out += 1
                else:
                    self.nonzero_out += 1
                self.valid_token_out += 1

        if self.debug:
            print("Done:", self.out_done(),
                  "\n Curr in crd1: ", self.curr_in1_crdpt,
                  "\t Curr in crd0", self.curr_in0_crdpt,
                  "\t Curr in val", self.curr_in_val,
                  "\n Curr out crd1: ", self.curr_crdpt1,
                  "\t Curr out crd0: ", self.curr_crdpt0,
                  "\t Curr out val: ", self.curr_val,
                  "\n Emit crds: ", self.emit_output,
                  "\n Storage: ", self.storage)

    def set_inner_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_crdpt0.append(crdpt)

    def set_outer_crdpt(self, crdpt):
        assert not is_stkn(crdpt), 'Coordinate points should not have stop tokens'
        if crdpt != '' and crdpt is not None:
            self.in_crdpt1.append(crdpt)

    def set_val(self, val):
        assert not is_stkn(val), 'Values associated with points should not have stop tokens'
        if val != '' and val is not None:
            self.in_val.append(val)

    def out_outer_crdpt(self):
        return self.curr_crdpt1

    def out_inner_crdpt(self):
        return self.curr_crdpt0

    def out_val(self):
        return self.curr_val

    def return_hits(self):
        i = 0
        cnt_gt_zero = 0
        cnt_total = 0
        for k in self.hits_tracker.keys():
            if self.hits_tracker[k] > i:
                i = self.hits_tracker[k]
            if self.hits_tracker[k] > 1:
                cnt_gt_zero += 1
            cnt_total += 1
        return i, cnt_gt_zero, cnt_total

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"stkn_outs": self.stop_token_out,
                          "drop_outs": self.drop_token_out, "valid_outs": self.valid_token_out,
                          "zero_outs": self.zero_out, "nonzero_outs": self.nonzero_out}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


# NEW VERSION: Accumulation into a matrix
class SparseAccumulator2(Primitive):
    def __init__(self, maxdim=100, valtype=float, last_level=True, val_stkn=False, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
        self.maxdim = maxdim
        self.valtype = valtype
        self.last_level = last_level
        # Boolean flag for whether to output stop tokens on the output val stream
        self.val_stkn = val_stkn

        self.in_crd2 = []
        self.in_crd1 = []
        self.in_crd0 = []
        self.in_val = []

        self.storage = {}

        self.curr_crd2 = None
        self.curr_crd1 = None
        self.curr_crd0 = None
        self.curr_val = None

        self.curr_in_crd1 = None
        self.curr_in_crd0 = None
        self.curr_in_val = None

        self.states = enum.Enum('States', ['READY', 'ACC1', 'ACC0', 'WR', 'DONE'])
        self.curr_state = self.states.READY
        self.next_state = self.states.READY
        self.writeout0 = False
        self.writeout1 = False
        self.writeout_storage1 = []
        self.writeout_storage0 = []

        self.seen_done = False
        self.crd2_stkn = None

        if self.get_stats:
            self.in_crd0_fifo = 0
            self.in_crd1_fifo = 0
            self.in_val_fifo = 0

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail_2 = True
            self.fifo_avail_1 = True
            self.fifo_avail_0 = True
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

    def print_debug(self):
        print("========== " + self.name + " SPACC2 (NEW) ==========")
        print("Inputs: ", self.in_crd2, self.in_crd1, self.in_crd0, self.in_val)
        print("Temps: ", self.curr_crd2, self.crd2_stkn, self.curr_in_crd1, self.curr_in_crd0, self.curr_in_val)
        print("Store/Wr: ", self.storage, self.writeout_storage1, self.writeout_storage0, self.writeout1,
              self.writeout0)
        print("Outputs: ", self.curr_crd1, self.curr_crd0, self.curr_val)
        print("State: ", self.curr_state, self.next_state)

    def get_writout(self):
        return self.writeout0 or self.writeout1

    def build_writeout(self):
        result = []
        for crd1 in sorted(self.storage.keys()):
            for crd0, val in self.storage[crd1].items():
                result.append((crd0, val))
            result.append(('S0', 'S0'))
        # Remove last stop token
        return result[:-1]

    def update_storage(self, crd1, crd0, val):
        if crd1 not in self.storage.keys():
            self.storage[crd1] = {crd0: val}
        else:
            assert isinstance(self.storage[crd1], dict), "Storage must be a dictionary of dictionaries"
            if crd0 not in self.storage[crd1].keys():
                self.storage[crd1][crd0] = val
            else:
                self.storage[crd1][crd0] += val

    def ACC1_body(self):
        # In accumulation, accumulate into memory
        self.curr_in_crd1 = self.in_crd1.pop(0)

        # All elements are ready
        if is_nc_tkn(self.curr_in_crd1):
            if len(self.in_crd0) > 0 and len(self.in_val) > 0:
                self.ACC0_body()
            else:
                self.next_state = self.states.ACC0
        elif is_stkn(self.curr_in_crd1):
            self.next_state = self.states.READY
        elif is_dtkn(self.curr_in_crd1):
            assert False, "Shouldn't have done token for coordinates if in accumulate (ACC) state"
        else:
            assert False, "Cannot have a coordinate token of this type: " + str(self.curr_in_crd1)

    def ACC0_body(self):
        self.curr_in_crd0 = self.in_crd0.pop(0)
        self.curr_in_val = self.in_val.pop(0)

        if is_nc_tkn(self.curr_in_val, self.valtype):
            assert is_nc_tkn(self.curr_in_crd0, int), "The inner coordinate must be a non-control token"
            self.update_storage(self.curr_in_crd1, self.curr_in_crd0, self.curr_in_val)
            self.next_state = self.states.ACC0
        # In accumulation, if you see a stop token in the inner level, go back to start
        elif is_stkn(self.curr_in_crd0):
            assert is_stkn(self.curr_in_val) and stkn_order(self.curr_in_crd0) == stkn_order(self.curr_in_val), \
                "Stop tokens must match for inner crd: " + str(self.curr_in_crd0) + " and val: " + str(self.curr_in_val)
            self.next_state = self.states.ACC1
        elif is_dtkn(self.curr_in_crd0):
            assert False, "Shouldn't have done token for coordinates if in accumulate (ACC) state"
        # Do nothing
        else:
            assert False, "Cannot have a coordinate token of this type: " + str(self.curr_in_crd0)

    def update(self):
        self.update_done()
        self.update_ready()

        # Print out debugging statements
        if self.debug:
            self.print_debug()

        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            # Update state of state machine to next state
            self.curr_state = self.next_state

            # FIXME: (owhsu) self.data_ready not defined in init
            if self.backpressure_en:
                self.data_valid = True
                self.data_ready = True

            if self.done:
                # Reset things
                # fch1, fch2 = self.crdhold_01.return_fifo()
                # fs1, fs2, fs3 = self.spacc1_no_crdhold.return_fifo()
                # fsd1 = self.stkndrop_crd1.return_fifo()
                # fsd2 = self.stkndrop_crd0.return_fifo()
                # fsd3 = self.stkndrop_val.return_fifo()
                # self.crdhold_01 = CrdHold(fifos=[fch1, fch2], **self.kwargs)
                #
                # self.spacc1_no_crdhold = SpAcc1NoCrdHold(maxdim=self.maxdim, valtype=self.valtype,
                #                                          name=self.name + "_no_crdhold",
                #                                          fifos=[fs1, fs2, fs3], **self.kwargs)
                # self.stkndrop_crd1 = StknDrop(fifos=[fsd1], **self.kwargs)
                # self.stkndrop_crd0 = StknDrop(fifos=[fsd2], **self.kwargs)
                # self.stkndrop_val = StknDrop(fifos=[fsd3], **self.kwargs)
                pass

            # Set when block counts should start
            if len(self.in_crd1) > 0 or len(self.in_crd0) > 0 or len(self.in_val) > 0:
                self.block_start = False

            # Store block statistics if asked to be reported
            if self.get_stats:
                # What to do for drop tokens?
                pass

            # Begin state machine computation
            # READY State
            # Accepts crd2 token
            if self.curr_state == self.states.READY:
                if len(self.in_crd2) > 0:
                    self.curr_crd2 = self.in_crd2.pop(0)
                    if is_nc_tkn(self.curr_crd2):
                        if len(self.in_crd1) > 0:
                            self.ACC1_body()
                        else:
                            self.next_state = self.states.ACC1
                    elif is_stkn(self.curr_crd2) and not self.get_writout():
                        # Set writeout to be true, move over storage, and clear it.
                        self.crd2_stkn = self.curr_crd2
                        self.writeout1 = True
                        self.writeout0 = True
                        self.writeout_storage1 = [k for k in sorted(self.storage.keys())]
                        self.writeout_storage0 = self.build_writeout()
                        self.storage = {}
                        self.next_state = self.states.READY
                    elif is_stkn(self.curr_crd2):
                        # Wait for previous writeout to be done
                        self.next_state = self.states.WR
                    elif is_dtkn(self.curr_crd2):
                        self.seen_done = True
                        if self.get_writout():
                            self.next_state = self.states.WR
                        else:
                            self.next_state = self.states.DONE
                    else:
                        assert False, "Cannot have a coordinate token of this type: " + str(self.curr_crd2)
            # Handle accumulation into storage
            # Accepts crd1 token
            elif self.curr_state == self.states.ACC1:
                if len(self.in_crd1) > 0:
                    self.ACC1_body()
                else:
                    self.next_state = self.states.ACC1
            elif self.curr_state == self.states.ACC0:
                if len(self.in_crd0) > 0 and len(self.in_val) > 0:
                    self.ACC0_body()
                else:
                    self.next_state = self.states.ACC0
            # Finish writeout and then be done or go back to ready state
            elif self.curr_state == self.states.WR:
                if not self.get_writout() and self.seen_done:
                    self.next_state = self.states.DONE
                elif not self.get_writout():
                    self.next_state = self.states.READY
                else:
                    self.next_state = self.states.WR
            # See a done signal AND writeout is done
            # Stay done until reset
            elif self.curr_state == self.states.DONE:
                self.done = True
                self.next_state = self.states.DONE
            # Default to the current state
            else:
                self.next_state = self.curr_state

        # Writeout is true, so writeout the current values for storage
        if self.get_writout():
            # Writeout is done when there are no elements left
            # Writeout outer (crd1) level coordinates
            if self.writeout1:
                if len(self.writeout_storage1) == 0:
                    self.writeout1 = False
                    assert self.crd2_stkn is not None, "The current writeout stop token should not be None"
                    self.curr_crd1 = self.crd2_stkn
                else:
                    curr_writeout_elem1 = self.writeout_storage1.pop(0)
                    self.curr_crd1 = curr_writeout_elem1
            else:
                self.curr_crd1 = ""

            # Writeout inner (crd0) level coordinates
            if self.writeout0:
                if len(self.writeout_storage0) == 0:
                    self.writeout0 = False
                    assert self.crd2_stkn is not None, "The current writeout stop token should not be None"
                    self.curr_crd0 = increment_stkn(self.crd2_stkn)

                    self.curr_val = increment_stkn(self.crd2_stkn)
                else:
                    curr_writeout_elem0 = self.writeout_storage0.pop(0)
                    self.curr_crd0 = curr_writeout_elem0[0]
                    self.curr_val = curr_writeout_elem0[1]
            else:
                self.curr_crd0 = ""
                self.curr_val = ""

        elif self.done:
            self.curr_val = 'D'
            self.curr_crd0 = 'D'
            self.curr_crd1 = 'D'
        else:
            self.curr_val = ""
            self.curr_crd0 = ""
            self.curr_crd1 = ""

        # Update the current state
        self.curr_state = self.next_state

        if self.debug:
            print(self.in_crd1, self.in_crd0, self.in_val)
            print(self.curr_crd1, self.curr_crd0, self.curr_val)
            print("====== SPACC2 END =======")

    def set_in_crd0(self, crd, parent=None):
        if super().valid_token(crd, int):
            self.in_crd0.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_0)

    def set_in_crd1(self, crd, parent=None):
        if super().valid_token(crd, int):
            self.in_crd1.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_1)

    def set_in_crd2(self, crd, parent=None):
        if super().valid_token(crd, int):
            self.in_crd2.append(crd)
        if self.backpressure_en:
            parent.set_backpressure(self.fifo_avail_2)

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
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def print_fifos(self):
        print("Spaccumulator: None available")
