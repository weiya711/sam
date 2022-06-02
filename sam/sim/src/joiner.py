from .base import *


class Joiner2(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.oref1 = 0
        self.oref2 = 0

    def out_ref1(self):
        return self.oref1

    def out_ref2(self):
        return self.oref2


class CrdJoiner2(Joiner2, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ocrd = 0

        self.in_ref1 = []
        self.in_ref2 = []
        self.in_crd1 = []
        self.in_crd2 = []

    def set_in1(self, in_ref1, in_crd1):
        if in_ref1 != '' and in_crd1 != '':
            self.in_ref1.append(in_ref1)
            self.in_crd1.append(in_crd1)

    def set_in2(self, in_ref2, in_crd2):
        if in_ref2 != '' and in_crd2 != '':
            self.in_ref2.append(in_ref2)
            self.in_crd2.append(in_crd2)

    def out_crd(self):
        return self.ocrd


class BVJoiner2(Joiner2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def set_in1(self, in_ref1, in_bv1):
        pass

    @abstractmethod
    def set_in2(self, in_ref2, in_bv2):
        pass

    @abstractmethod
    def out_bv(self):
        pass


class Intersect2(CrdJoiner2):
    def __init__(self, skip=True, **kwargs):
        super().__init__(**kwargs)

        self.size_in_ref1 = 0
        self.size_in_ref2 = 0
        self.size_in_crd1 = 0
        self.size_in_crd2 = 0

        self.difference_in_ref = 0
        self.max_diff_in_ref = 0
        self.drop_token_output = 0

        self.ocrd = 0
        self.oref1 = 0
        self.oref2 = 0
        self.curr_crd1 = None
        self.curr_crd2 = None
        self.curr_ref1 = None
        self.curr_ref2 = None

        self.total_count = 0
        self.count = 0
        self.run_count = 0
        self.max_run_count = 0

        self.skip = skip
        self.curr_skip1 = ''
        self.curr_skip2 = ''
        self.change_crd1 = True
        self.change_crd2 = True

    def _inc2(self):
        self.ocrd = ''
        self.oref1 = ''
        self.oref2 = ''
        self.curr_crd2 = self.in_crd2.pop(0)
        self.curr_ref2 = self.in_ref2.pop(0)
        self.change_crd2 = True
        self.curr_skip2 = self.curr_crd1 if self.change_crd1 else ''  # Skip list
        self.change_crd1 = False
        self.total_count += 1

    def _inc1(self):
        self.ocrd = ''
        self.oref1 = ''
        self.oref2 = ''
        self.curr_crd1 = self.in_crd1.pop(0)
        self.curr_ref1 = self.in_ref1.pop(0)
        self.change_crd1 = True
        self.curr_skip1 = self.curr_crd2 if self.change_crd2 else ''  # Skip list
        self.change_crd2 = False
        self.total_count += 1

    def update(self):
        # Skip list
        self.curr_skip1 = ''
        self.curr_skip2 = ''

        if len(self.in_crd1) > 0 and len(self.in_crd2) > 0:
            # FIXME: See when only one 'D' signal is present
            if self.curr_crd1 == 'D' or self.curr_crd2 == 'D':
                assert self.curr_crd1 == self.curr_crd2, "Both coordinates need to be done tokens"
                self.done = True
                self.ocrd = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
            elif self.curr_crd2 == self.curr_crd1:
                # Skip list
                if is_stkn(self.curr_crd2):
                    self.curr_skip2 = self.curr_crd1 if self.change_crd1 else ''
                    self.curr_skip1 = self.curr_crd2 if self.change_crd2 else ''

                self.ocrd = '' if self.curr_crd2 is None else self.curr_crd1
                self.oref1 = '' if self.curr_ref1 is None else self.curr_ref1
                self.oref2 = '' if self.curr_ref2 is None else self.curr_ref2
                self.curr_crd1 = self.in_crd1.pop(0)
                self.curr_crd2 = self.in_crd2.pop(0)
                self.curr_ref1 = self.in_ref1.pop(0)
                self.curr_ref2 = self.in_ref2.pop(0)
                self.change_crd1 = True
                self.change_crd2 = True
                self.total_count += 1
                self.count += 1
                self.run_count = 0
                self.max_run_count = max(self.max_run_count, abs(self.run_count))
            elif is_stkn(self.curr_crd1):
                self._inc2()
            elif is_stkn(self.curr_crd2):
                self._inc1()
            elif self.curr_crd1 < self.curr_crd2:
                self._inc1()
                if self.run_count >= 0:
                    self.run_count += 1
                    self.max_run_count = max(self.max_run_count, abs(self.run_count))
                else:
                    self.run_count = 0
            elif self.curr_crd1 > self.curr_crd2:
                self._inc2()
                if self.run_count < 0:
                    self.run_count -= 1
                    self.max_run_count = max(self.max_run_count, abs(self.run_count))
                else:
                    self.run_count = 0
            else:
                raise Exception('Intersect2: should not enter this case')
        else:
            # Do Nothing if no inputs are detected
            if self.curr_crd1 == 'D' or self.curr_crd2 == 'D':
                self.done = True
                self.ocrd = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
                self.curr_crd1 = ''
                self.curr_crd2 = ''
                self.curr_ref1 = ''
                self.curr_ref2 = ''
            else:
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
        self.compute_fifos()

        if self.debug:
            print("DEBUG: INTERSECT: ",
                  "\n OutCrd:", self.ocrd, "\t Out Ref1:", self.oref1, "\t Out Ref2:", self.oref2,
                  "\n Crd1:", self.curr_crd1, "\t Ref1:", self.curr_ref1,
                  "\n Crd2:", self.curr_crd2, "\t Ref2", self.curr_ref2,
                  "\n Skip1:", self.curr_skip1, "\t Skip2:", self.curr_skip2,
                  "\t Change1:", self.change_crd1, "\t Change2:", self.change_crd2,
                  "\n Intersection rate: ",
                  self.return_intersection_rate())

    def out_crd_skip1(self):
        return self.curr_skip1 if self.skip else ''

    def out_crd_skip2(self):
        return self.curr_skip2 if self.skip else ''

    def compute_fifos(self):
        self.size_in_ref1 = max(self.size_in_ref1, len(self.in_ref1))
        self.size_in_ref2 = max(self.size_in_ref2, len(self.in_ref2))
        self.size_in_crd1 = max(self.size_in_crd1, len(self.in_crd1))
        self.size_in_crd2 = max(self.size_in_crd2, len(self.in_crd2))

    def print_fifos(self):
        print("FIFO in ref 1: ", self.size_in_ref1)
        print("FIFO in ref 2: ", self.size_in_ref2)
        print("FIFO in crd 1: ", self.size_in_crd1)
        print("FIFO in crd 2: ", self.size_in_crd2)

    def return_intersection_rate(self):
        return self.count / self.total_count if self.total_count != 0 else 0

    def print_intersection_rate(self):
        return print("Intersection rate: ", self.return_intersection_rate())

    def return_statistics(self):
        stat_dict = {"fifos_ref_1": self.size_in_ref1, "fifos_ref_2": self.size_in_ref2,
                     "fifos_crd_1": self.size_in_crd1, "fifos_crd_2": self.size_in_crd2,
                     "fifo_difference": self.max_diff_in_ref, "intersection_rate": self.count / self.total_count,
                     "drop_count": self.drop_token_output, "valid_output": self.total_count - self.drop_token_output,
                     "run_count": self.max_run_count}
        return stat_dict


class Union2(CrdJoiner2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.size_in_ref1 = 0
        self.size_in_ref2 = 0
        self.size_in_crd1 = 0
        self.size_in_crd2 = 0

        self.curr_crd1 = None
        self.curr_crd2 = None
        self.curr_ref1 = None
        self.curr_ref2 = None

        self.total_count = 0
        self.count = 0

    def update(self):
        if len(self.in_crd1) > 0 and len(self.in_crd2) > 0:
            if self.curr_crd1 == 'D' or self.curr_crd2 == 'D':
                assert self.curr_crd1 == self.curr_ref1 == self.curr_crd2 == self.curr_ref2
                self.done = True
                self.ocrd = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
            elif self.curr_crd2 == self.curr_crd1:
                self.ocrd = '' if self.curr_crd2 is None else self.curr_crd1
                self.oref1 = '' if self.curr_ref1 is None else self.curr_ref1
                self.oref2 = '' if self.curr_ref2 is None else self.curr_ref2
                self.curr_crd1 = self.in_crd1.pop(0)
                self.curr_crd2 = self.in_crd2.pop(0)
                self.curr_ref1 = self.in_ref1.pop(0)
                self.curr_ref2 = self.in_ref2.pop(0)
                self.total_count += 1
                self.count += 1
            elif is_stkn(self.curr_crd1):
                self.ocrd = self.curr_crd2
                self.oref1 = 'N'
                self.oref2 = self.curr_ref2
                self.curr_crd2 = self.in_crd2.pop(0)
                self.curr_ref2 = self.in_ref2.pop(0)
                self.total_count += 1
            elif is_stkn(self.curr_crd2):
                self.ocrd = self.curr_crd1
                self.oref1 = self.curr_ref1
                self.oref2 = 'N'
                self.curr_crd1 = self.in_crd1.pop(0)
                self.curr_ref1 = self.in_ref1.pop(0)
                self.total_count += 1
            elif self.curr_crd1 < self.curr_crd2:
                self.ocrd = self.curr_crd1
                self.oref1 = self.curr_ref1
                self.oref2 = 'N'
                self.curr_crd1 = self.in_crd1.pop(0)
                self.curr_ref1 = self.in_ref1.pop(0)
                self.total_count += 1
            elif self.curr_crd1 > self.curr_crd2:
                self.ocrd = self.curr_crd2
                self.oref1 = 'N'
                self.oref2 = self.curr_ref2
                self.curr_crd2 = self.in_crd2.pop(0)
                self.curr_ref2 = self.in_ref2.pop(0)
                self.total_count += 1
            else:
                raise Exception('Intersect2: should not enter this case')
        else:
            # Do Nothing if no inputs are detected
            if self.curr_crd1 == 'D' or self.curr_crd2 == 'D':
                assert self.curr_crd1 == self.curr_ref1 == self.curr_crd2 == self.curr_ref2
                self.done = True
                self.ocrd = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
                self.curr_crd1 = ''
                self.curr_crd2 = ''
                self.curr_ref1 = ''
                self.curr_ref2 = ''
            else:
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
        self.compute_fifos()

        if self.debug:
            print("DEBUG: INTERSECT: \t OutCrd:", self.ocrd, "\t Out Ref1:", self.oref1, "\t Out Ref2:", self.oref2,
                  "\n Crd1:", self.curr_crd1, "\t Ref1:", self.curr_ref1,
                  "\t Crd2:", self.curr_crd2, "\t Ref2", self.curr_ref2,
                  "\n Union rate: ",
                  self.return_union_rate())

    def set_in1(self, in_ref1, in_crd1):
        if in_ref1 != '' and in_crd1 != '':
            self.in_ref1.append(in_ref1)
            self.in_crd1.append(in_crd1)

    def set_in2(self, in_ref2, in_crd2):
        if in_ref2 != '' and in_crd2 != '':
            self.in_ref2.append(in_ref2)
            self.in_crd2.append(in_crd2)

    def compute_fifos(self):
        self.size_in_ref1 = max(self.size_in_ref1, len(self.in_ref1))
        self.size_in_ref2 = max(self.size_in_ref2, len(self.in_ref2))
        self.size_in_crd1 = max(self.size_in_crd1, len(self.in_crd1))
        self.size_in_crd2 = max(self.size_in_crd2, len(self.in_crd2))

    def print_fifos(self):
        print("FIFO in ref 1: ", self.size_in_ref1)
        print("FIFO in ref 2: ", self.size_in_ref2)
        print("FIFO in crd 1: ", self.size_in_crd1)
        print("FIFO in crd 2: ", self.size_in_crd2)

    def out_crd(self):
        return self.ocrd

    def out_ref1(self):
        return self.oref1

    def out_ref2(self):
        return self.oref2

    def return_union_rate(self):
        return self.count / self.total_count if self.total_count != 0 else 0

    def print_union_rate(self):
        return print("Intersection rate: ", self.return_union_rate())


class IntersectBV2(BVJoiner2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_ref1 = []
        self.in_ref2 = []
        self.in_bv1 = []
        self.in_bv2 = []

        self.size_in_ref1 = 0
        self.size_in_ref2 = 0
        self.size_in_bv1 = 0
        self.size_in_bv2 = 0

        self.obv = ''
        self.oref1 = ''
        self.oref2 = ''

        self.curr_bv1 = None
        self.curr_bv2 = None
        self.curr_ref1 = None
        self.curr_ref2 = None

        self.reflist1 = []
        self.reflist2 = []
        self.emit_refs = False

        self.total_count = 0
        self.count = 0

    def update(self):
        if self.done:
            self.obv = ''
            self.oref1 = ''
            self.oref2 = ''
            return

        if self.emit_refs:
            assert len(self.reflist1) == len(self.reflist2), "Lengths of refs must match"
            self.obv = ''
            self.oref1 = self.reflist1.pop(0)
            self.oref2 = self.reflist2.pop(0)

            self.emit_refs = len(self.reflist1) > 0
            return
        if len(self.in_bv1) > 0 and len(self.in_bv2) > 0:
            self.curr_bv1 = self.in_bv1.pop(0)
            self.curr_bv2 = self.in_bv2.pop(0)
            self.curr_ref1 = self.in_ref1.pop(0)
            self.curr_ref2 = self.in_ref2.pop(0)

            # FIXME: See when only one 'D' signal is present
            if self.curr_bv1 == 'D' or self.curr_bv2 == 'D':
                assert self.curr_bv1 == self.curr_bv2 == self.curr_ref1 == self.curr_ref2
                self.done = True
                self.obv = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
            elif is_stkn(self.curr_bv1) and is_stkn(self.curr_bv2):
                assert self.curr_bv1 == self.curr_bv2 == self.curr_ref1 == self.curr_ref2

                self.obv = self.curr_bv1
                self.oref1 = self.curr_bv1
                self.oref2 = self.curr_bv2
                self.total_count += 1
            elif self.curr_bv1 & self.curr_bv2:
                obv = self.curr_bv1 & self.curr_bv2

                reflist1 = []
                reflist2 = []
                self.obv = obv
                while obv:
                    rbit = right_bit_set(obv)
                    refbits1 = bin(self.curr_bv1 & (rbit - 1)).count('1') + self.curr_ref1
                    refbits2 = bin(self.curr_bv2 & (rbit - 1)).count('1') + self.curr_ref2

                    reflist1.append(refbits1)
                    reflist2.append(refbits2)
                    obv = ~rbit & obv

                self.reflist1 = reflist1
                self.reflist2 = reflist2

                self.oref1 = self.reflist1.pop(0)
                self.oref2 = self.reflist2.pop(0)

                self.emit_refs = len(self.reflist1) > 0

                self.total_count += 1
                self.count += 1
            elif not self.curr_bv1 & self.curr_bv2:
                self.obv = ''
                self.oref1 = ''
                self.oref2 = ''
            else:
                raise Exception('Intersect2: should not enter this case')
        else:
            # Do Nothing if no inputs are detected
            self.obv = ''
            self.oref1 = ''
            self.oref2 = ''
        self.compute_fifos()

        if self.debug:
            print("DEBUG: INTERSECT: \t Outbv:", self.obv, "\t Out Ref1:", self.oref1, "\t Out Ref2:", self.oref2,
                  "\t bv1:", self.curr_bv1, "\t Ref1:", self.curr_ref1,
                  "\t bv2:", self.curr_bv2, "\t Ref2", self.curr_ref2, "\t Intersection rate: ",
                  self.return_intersection_rate())

    def set_in1(self, in_ref1, in_bv1):
        if in_ref1 != '' and in_bv1 != '':
            self.in_ref1.append(in_ref1)
            self.in_bv1.append(in_bv1)

    def set_in2(self, in_ref2, in_bv2):
        if in_ref2 != '' and in_bv2 != '':
            self.in_ref2.append(in_ref2)
            self.in_bv2.append(in_bv2)

    def compute_fifos(self):
        self.size_in_ref1 = max(self.size_in_ref1, len(self.in_ref1))
        self.size_in_ref2 = max(self.size_in_ref2, len(self.in_ref2))
        self.size_in_bv1 = max(self.size_in_bv1, len(self.in_bv1))
        self.size_in_bv2 = max(self.size_in_bv2, len(self.in_bv2))

    def print_fifos(self):
        print("FIFO in ref 1: ", self.size_in_ref1)
        print("FIFO in ref 2: ", self.size_in_ref2)
        print("FIFO in bv 1: ", self.size_in_bv1)
        print("FIFO in bv 2: ", self.size_in_bv2)

    def out_bv(self):
        return self.obv

    def out_ref1(self):
        return self.oref1

    def out_ref2(self):
        return self.oref2

    def return_intersection_rate(self):
        return self.count / self.total_count if self.total_count > 0 else 0
