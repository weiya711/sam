from .base import *


class Repeat(Primitive):
    def __init__(self, union=False, **kwargs):
        super().__init__(**kwargs)

        self.in_ref = []
        self.in_repeat = []

        self.in_ref_size = 0
        self.in_repeat_size = 0
        self.curr_out_ref = ''
        self.curr_in_ref = ''
        self.curr_union_other = ''
        self.get_next_ref = True
        self.get_next_rep = True
        self.emit_stkn = False
        self.empty_rep_fiber = True

        self.get_next_ref_union = False
        self.meta_union_mode = union

    def update(self):
        self.update_done()

        if len(self.in_ref) > 0 or len(self.in_repeat) > 0:
            self.block_start = False
        # if len(self.in_ref) > 0 and self.get_next_ref_union:
        #     next_in = self.in_ref.pop(0)
        #     assert isinstance(next_in, int)
        #     if isinstance(next_in, int):
        #         self.curr_out_ref = next_in
        #         self.emit_stkn = True
        #     else:
        #         assert is_0tkn(next_in), "Next ref should only be int or 0tkn but is " + str(next_in)
        #         if len(self.in_ref) > 0:
        #             next_in = self.in_ref[0]
        #             if is_stkn(next_in):
        #                 stkn = increment_stkn(next_in)
        #                 self.in_ref.pop(0)
        #             else:
        #                 stkn = 'S0'
        #             self.curr_out_ref = stkn
        #         else:
        #             self.emit_stkn = True
        #             self.curr_out_ref = ''
        #     self.get_next_ref_union = False

        if len(self.in_ref) > 0 and self.emit_stkn:
            next_in = self.in_ref[0]
            print(self.in_ref)
            if is_stkn(next_in):
                stkn = increment_stkn(next_in)
                self.in_ref.pop(0)
            else:
                stkn = 'S0'
            print("THIS 0")
            self.curr_out_ref = stkn
            self.emit_stkn = False
            return
        elif self.emit_stkn:
            self.curr_out_ref = ''
            return

        if len(self.in_ref) > 0 and self.get_next_ref:
            print("_____HERE________")
            self.curr_in_ref = self.in_ref.pop(0)
            if is_stkn(self.curr_in_ref):
                self.get_next_rep = True
                self.get_next_ref = False
            elif self.curr_in_ref == 'D':
                self.curr_out_ref = 'D'
                self.get_next_rep = True
                self.get_next_ref = False
                self.done = True
            else:
                self.get_next_rep = True
                self.get_next_ref = False
        elif self.get_next_ref:
            self.curr_out_ref = ''
        print("get_next_rep", self.get_next_rep, self.empty_rep_fiber, self.meta_union_mode, self.curr_in_ref)
        repeat = ''
        if len(self.in_repeat) > 0 and self.get_next_rep:
            repeat = self.in_repeat.pop(0)
            if repeat == 'S' and self.empty_rep_fiber and self.meta_union_mode:
                if isinstance(self.curr_in_ref, int):
                    self.curr_out_ref = self.curr_in_ref
                    self.emit_stkn = True
                else:
                    assert is_0tkn(self.curr_in_ref), "Next ref should only be int or 0tkn but is " + str(
                        self.curr_in_ref)
                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            stkn = increment_stkn(next_in)
                            self.in_ref.pop(0)
                        else:
                            stkn = 'S0'
                        print("THIS 1")
                        self.curr_out_ref = stkn
                    else:
                        self.emit_stkn = True
                        self.curr_out_ref = ''
                self.empty_rep_fiber = True
                self.get_next_ref = True
                self.get_next_rep = False
            elif repeat == 'S':
                self.get_next_ref = True
                self.get_next_rep = False
                if isinstance(self.curr_in_ref, int):
                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            stkn = increment_stkn(next_in)
                            self.in_ref.pop(0)
                        else:
                            stkn = 'S0'
                        print("THIS 2")
                        self.curr_out_ref = stkn
                    else:
                        self.emit_stkn = True
                        self.curr_out_ref = ''
                    self.empty_rep_fiber = True
                elif is_stkn(self.curr_in_ref):
                    self.curr_out_ref = increment_stkn(self.curr_in_ref)
                    self.empty_rep_fiber = True
            elif repeat == 'D':
                if self.curr_out_ref != 'D':
                    raise Exception("Both repeat and ref signal need to end in 'D'")
                self.get_next_ref = True
                self.get_next_rep = False
                self.curr_out_ref = 'D'
            elif repeat == 'R':
                self.get_next_ref = False
                self.get_next_rep = True
                self.curr_out_ref = self.curr_in_ref
                self.empty_rep_fiber = False
            else:
                raise Exception('Repeat signal cannot be: ' + str(repeat))
        elif self.get_next_rep:
            self.curr_out_ref = ''
        self.compute_fifos()
        if self.debug:
            print("DEBUG: REPEAT:", "\t Get Ref:", self.get_next_ref, "\tIn Ref:", self.curr_in_ref,
                  "\t Get Rep:", self.get_next_rep, "\t Rep:", repeat,
                  "\t Out Ref:", self.curr_out_ref, "\tEmit Stkn", self.emit_stkn, "\t Streams", self.in_ref, " ",
                  self.in_repeat)

    def set_in_ref(self, ref):
        if ref != '' and ref is not None:
            self.in_ref.append(ref)

    def set_in_repeat(self, repeat):
        if repeat != '' and repeat is not None:
            self.in_repeat.append(repeat)

    # def set_in_union_0tkn(self, union_0tkn):
    #     if union_0tkn != '':
    #         self.in_union_other_0tkn.append(union_0tkn)
    #
    # def set_in_union_other_ref(self, union_0tkn):
    #     if union_0tkn != '':
    #         self.in_union_other_0tkn.append(union_0tkn)

    def set_in_repsig(self, repeat):
        if repeat != '' and repeat is not None:
            self.in_repeat.append(repeat)

    def out_ref(self):
        return self.curr_out_ref

    def compute_fifos(self):
        self.in_ref_size = max(self.in_ref_size, len(self.in_ref))
        self.in_repeat_size = max(self.in_repeat_size, len(self.in_repeat))

    def print_fifos(self):
        print("FIFOs size in the ref for repeat block: ", self.in_ref_size)
        print("Repeat size for repeat block: ", self.in_repeat_size)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"in_ref_size": self.in_ref_size, "in_repeat_size": self.in_repeat_size}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


# Repeat signal generator will take a crd stream and generate repeat, 'R',
# or next coordinate, 'S', signals for broadcasting along a non-existent dimension.
# It essentially snoops on the crd stream

class RepeatSigGen(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.istream = []
        self.curr_repeat = ''
        self.istream_size = 0

    def update(self):
        self.update_done()
        if len(self.istream) > 0:
            self.block_start = False

        istream = ''

        if len(self.istream) > 0:
            istream = self.istream.pop(0)
            if is_stkn(istream):
                self.curr_repeat = 'S'
            elif istream == 'D':
                self.curr_repeat = 'D'
                self.done = True
            else:
                self.curr_repeat = 'R'
        else:
            self.curr_repeat = ''
        self.compute_fifos()
        if self.debug:
            print("DEBUG: REP GEN", "\t In", istream, "\t Out ", self.curr_repeat, "\t INstream", self.istream)

    # input can either be coordinates or references
    def set_istream(self, istream):
        if istream != '' and istream is not None:
            self.istream.append(istream)

    def out_repeat(self):
        return self.curr_repeat

    def out_repsig(self):
        return self.curr_repeat

    def compute_fifos(self):
        self.istream_size = max(self.istream_size, len(self.istream))

    def print_fifos(self):
        print("Repeat sig gen size:", self.istream_size)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"in_repeat_size": self.istream_size}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


class Repeat_back(Primitive):
    def __init__(self, depth=1, union=False, **kwargs):
        super().__init__(**kwargs)

        self.in_ref = []
        self.in_repeat = []

        self.in_ref_size = 0
        self.in_repeat_size = 0
        self.curr_out_ref = ''
        self.curr_in_ref = ''
        self.curr_union_other = ''
        self.get_next_ref = True
        self.get_next_rep = True
        self.emit_stkn = False
        self.empty_rep_fiber = True

        self.get_next_ref_union = False
        self.meta_union_mode = union

        self.backpressure = []
        self.data_ready = True
        self.branches = []
        self.depth = depth

    def check_backpressure(self):
        j = 0
        for i in self.backpressure:
            if not i.fifo_available(self.branches[j]):
                return False
            j += 1
        return True

    def fifo_debug(self):
        print("Repeater: ", self.in_ref, " ", self.in_repeat)

    def fifo_available(self, br=""):
        if br == "ref" and len(self.in_ref) > self.depth:
            return False
        if len(self.in_repeat) > self.depth and (br == "repeat" or br == "repsig"):
            return False
        return True

    def add_child(self, child=None, branch=""):
        self.backpressure.append(child)
        self.branches.append(branch)

    def update(self):
        self.update_done()

        self.data_ready = False
        if self.check_backpressure():

            self.data_ready = True
            if len(self.in_ref) > 0 or len(self.in_repeat) > 0:
                self.block_start = False

        # if len(self.in_ref) > 0 and self.get_next_ref_union:
        #     next_in = self.in_ref.pop(0)
        #     assert isinstance(next_in, int)
        #     if isinstance(next_in, int):
        #         self.curr_out_ref = next_in
        #         self.emit_stkn = True
        #     else:
        #         assert is_0tkn(next_in), "Next ref should only be int or 0tkn but is " + str(next_in)
        #         if len(self.in_ref) > 0:
        #             next_in = self.in_ref[0]
        #             if is_stkn(next_in):
        #                 stkn = increment_stkn(next_in)
        #                 self.in_ref.pop(0)
        #             else:
        #                 stkn = 'S0'
        #             self.curr_out_ref = stkn
        #         else:
        #             self.emit_stkn = True
        #             self.curr_out_ref = ''
        #     self.get_next_ref_union = False

            if len(self.in_ref) > 0 and self.emit_stkn:
                next_in = self.in_ref[0]
                if is_stkn(next_in):
                    stkn = increment_stkn(next_in)
                    self.in_ref.pop(0)
                else:
                    stkn = 'S0'
                self.curr_out_ref = stkn
                self.emit_stkn = False
                return
            elif self.emit_stkn:
                self.curr_out_ref = ''
                return

            if len(self.in_ref) > 0 and self.get_next_ref:
                self.curr_in_ref = self.in_ref.pop(0)
                if is_stkn(self.curr_in_ref):
                    self.get_next_rep = True
                    self.get_next_ref = False
                elif self.curr_in_ref == 'D':
                    self.curr_out_ref = 'D'
                    self.get_next_rep = True
                    self.get_next_ref = False
                    self.done = True
                else:
                    self.get_next_rep = True
                    self.get_next_ref = False
            elif self.get_next_ref:
                self.curr_out_ref = ''

            if self.debug:
                print("DEBUG__Now: REPEAT:", "\t Get Ref:", self.get_next_ref, "\tIn Ref:", self.curr_in_ref,
                      "\t Get Rep:", self.get_next_rep,
                      "\t Out Ref:", self.curr_out_ref, "\tEmit Stkn", self.emit_stkn, "\tStream", self.in_ref,
                      "in_repeat", self.in_repeat, " backstream: ", self.check_backpressure(), " ", self.data_ready)

            repeat = ''
            if len(self.in_repeat) > 0 and self.get_next_rep:

                repeat = self.in_repeat.pop(0)
                if repeat == 'S' and self.empty_rep_fiber and self.meta_union_mode:
                    if isinstance(self.curr_in_ref, int):
                        self.curr_out_ref = self.curr_in_ref
                        self.emit_stkn = True
                    else:
                        assert is_0tkn(self.curr_in_ref), "Next ref should only be int or 0tkn but is " + str(
                            self.curr_in_ref)
                        if len(self.in_ref) > 0:
                            next_in = self.in_ref[0]
                            if is_stkn(next_in):
                                stkn = increment_stkn(next_in)
                                self.in_ref.pop(0)
                            else:
                                stkn = 'S0'
                            self.curr_out_ref = stkn
                        else:
                            self.emit_stkn = True
                            self.curr_out_ref = ''
                    self.empty_rep_fiber = True
                    self.get_next_ref = True
                    self.get_next_rep = False
                elif repeat == 'S':
                    self.get_next_ref = True
                    self.get_next_rep = False
                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            stkn = increment_stkn(next_in)
                            self.in_ref.pop(0)
                        else:
                            stkn = 'S0'
                        self.curr_out_ref = stkn
                    else:
                        self.emit_stkn = True
                        self.curr_out_ref = ''
                    self.empty_rep_fiber = True
                elif repeat == 'D':
                    if self.curr_in_ref == 'D':
                        self.curr_out_ref = 'D'
                    if self.curr_out_ref != 'D':
                        raise Exception("Both repeat and ref signal need to end in 'D'")
                    self.get_next_ref = True
                    self.get_next_rep = False
                    self.curr_out_ref = 'D'
                elif repeat == 'R':
                    self.get_next_ref = False
                    self.get_next_rep = True
                    self.curr_out_ref = self.curr_in_ref
                    self.empty_rep_fiber = False
                else:
                    raise Exception('Repeat signal cannot be: ' + str(repeat))
            elif self.get_next_rep:
                self.curr_out_ref = ''
            self.compute_fifos()
            if self.debug:
                print("DEBUG: REPEAT:", "\t Get Ref:", self.get_next_ref, "\tIn Ref:", self.curr_in_ref,
                      "\t Get Rep:", self.get_next_rep, "\t Rep:", repeat,
                      "\t Out Ref:", self.curr_out_ref, "\tEmit Stkn", self.emit_stkn, "\t Streams", self.in_ref,
                      "\t in_repeat:", self.in_repeat, " backstream: ", self.check_backpressure(), " ", self.data_ready)
        else:
            if self.debug:
                print("DEBUG: REPEAT:", "\t Get Ref:", self.get_next_ref, "\tIn Ref:", self.curr_in_ref,
                      "\t Get Rep:", self.get_next_rep,
                      "\t Out Ref:", self.curr_out_ref, "\tEmit Stkn", self.emit_stkn, "\tStream", self.in_ref,
                      " ", self.in_repeat, " backstream: ", self.check_backpressure(), " ", self.data_ready)

    def set_in_ref(self, ref):
        if ref != '' and ref is not None:
            self.in_ref.append(ref)

    def set_in_repeat(self, repeat):
        if repeat != '' and repeat is not None:
            self.in_repeat.append(repeat)

    # def set_in_union_0tkn(self, union_0tkn):
    #     if union_0tkn != '':
    #         self.in_union_other_0tkn.append(union_0tkn)
    #
    # def set_in_union_other_ref(self, union_0tkn):
    #     if union_0tkn != '':
    #         self.in_union_other_0tkn.append(union_0tkn)

    def set_in_repsig(self, repeat):
        if repeat != '' and repeat is not None:
            self.in_repeat.append(repeat)

    def out_ref(self):
        if self.data_ready:
            return self.curr_out_ref

    def compute_fifos(self):
        self.in_ref_size = max(self.in_ref_size, len(self.in_ref))
        self.in_repeat_size = max(self.in_repeat_size, len(self.in_repeat))

    def print_fifos(self):
        print("FIFOs size in the ref for repeat block: ", self.in_ref_size)
        print("Repeat size for repeat block: ", self.in_repeat_size)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"in_ref_size": self.in_ref_size, "in_repeat_size": self.in_repeat_size}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict


# Repeat signal generator will take a crd stream and generate repeat, 'R',
# or next coordinate, 'S', signals for broadcasting along a non-existent dimension.
# It essentially snoops on the crd stream

class RepeatSigGen_back(Primitive):
    def __init__(self, depth=1, **kwargs):
        super().__init__(**kwargs)
        self.backpressure = []
        self.data_ready = True
        self.branches = []
        self.depth = depth

        self.istream = []
        self.curr_repeat = ''
        self.istream_size = 0

    def update(self):
        self.update_done()
        self.data_ready = False
        if self.check_backpressure():
            self.data_ready = True
            if len(self.istream) > 0:
                self.block_start = False

            istream = ''

            if len(self.istream) > 0:
                istream = self.istream.pop(0)
                if is_stkn(istream):
                    self.curr_repeat = 'S'
                elif istream == 'D':
                    self.curr_repeat = 'D'
                    self.done = True
                else:
                    self.curr_repeat = 'R'
            else:
                self.curr_repeat = ''
            self.compute_fifos()
            if self.debug:
                print("DEBUG: REP GEN:", "\t In:", istream, "\t Out:", self.curr_repeat, "\t Instream", self.istream,
                      " backstream: ", self.check_backpressure(), " ", self.data_ready, " ", self.backpressure,
                      " ", self.branches)
        else:
            if self.debug:
                print("DEBUG: REP GEN", "\t In", "", "\t Out ", self.curr_repeat, "\t INstream", self.istream,
                      " backstream: ", self.check_backpressure(), " ", self.data_ready, " ", self.backpressure,
                      " ", self.branches)

    def check_backpressure(self):
        j = 0
        for i in self.backpressure:
            if not i.fifo_available(self.branches[j]):
                return False
            j += 1
        return True

    def fifo_debug(self):
        print("Repeat sig : ", self.istream)

    def fifo_available(self, br=""):
        if len(self.istream) > self.depth:
            return False
        return True

    def add_child(self, child=None, branch=""):
        self.backpressure.append(child)
        self.branches.append(branch)

    # input can either be coordinates or references
    def set_istream(self, istream):
        if istream != '' and istream is not None:
            self.istream.append(istream)

    def out_repeat(self):
        if self.data_ready:
            return self.curr_repeat

    def out_repsig(self):
        if self.data_ready:
            return self.curr_repeat

    def compute_fifos(self):
        self.istream_size = max(self.istream_size, len(self.istream))

    def print_fifos(self):
        print("Repeat sig gen size:", self.istream_size)

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"in_repeat_size": self.istream_size}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict
