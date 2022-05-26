from .base import *


class Joiner2(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def out_ref1(self):
        pass

    @abstractmethod
    def out_ref2(self):
        pass


class CrdJoiner2(Joiner2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def set_in1(self, in_ref1, in_crd1):
        pass

    @abstractmethod
    def set_in2(self, in_ref2, in_crd2):
        pass

    @abstractmethod
    def out_crd(self):
        pass


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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_ref1 = []
        self.in_ref2 = []
        self.in_crd1 = []
        self.in_crd2 = []

        self.size_in_ref1 = 0
        self.size_in_ref2 = 0
        self.size_in_crd1 = 0
        self.size_in_crd2 = 0

        self.ocrd = 0
        self.oref1 = 0
        self.oref2 = 0

        self.curr_crd1 = None
        self.curr_crd2 = None
        self.curr_ref1 = None
        self.curr_ref2 = None

        self.total_count = 0
        self.count = 0

    def update(self):
        flag = 0
        if len(self.in_crd1) > 0 and len(self.in_crd2) > 0:
            # FIXME: See when only one 'D' signal is present
            if self.curr_crd1 == 'D' or self.curr_crd2 == 'D':
                flag = 1
                self.done = True
                self.ocrd = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
            elif self.curr_crd2 == self.curr_crd1: # and is_stkn(self.curr_crd1) == False and is_stkn(self.curr_crd2) == False :
                flag = 2
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
                flag = 3
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
                self.curr_crd2 = self.in_crd2.pop(0)
                self.curr_ref2 = self.in_ref2.pop(0)
                self.total_count += 1
            elif is_stkn(self.curr_crd2):
                flag = 4
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
                self.curr_crd1 = self.in_crd1.pop(0)
                self.curr_ref1 = self.in_ref1.pop(0)
                self.total_count += 1
            elif self.curr_crd1 < self.curr_crd2:
                flag = 5
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
                self.curr_crd1 = self.in_crd1.pop(0)
                self.curr_ref1 = self.in_ref1.pop(0)
                self.total_count += 1
            elif self.curr_crd1 > self.curr_crd2:
                flag = 6
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
                self.curr_crd2 = self.in_crd2.pop(0)
                self.curr_ref2 = self.in_ref2.pop(0)
                self.total_count += 1
            else:
                raise Exception('Intersect2: should not enter this case')
        else:
            # Do Nothing if no inputs are detected
            if self.curr_crd1 == 'D' or self.curr_crd2 == 'D':
                flag = 7
                self.done = True
                self.ocrd = 'D'
                self.oref1 = 'D'
                self.oref2 = 'D'
                self.curr_crd1 = ''
                self.curr_crd2 = ''
                self.curr_ref1 = ''
                self.curr_ref2 = ''
            else:
                flag = 8
                self.ocrd = ''
                self.oref1 = ''
                self.oref2 = ''
        self.compute_fifos()

        if self.debug:
            print(flag, "\t DEBUG: INTERSECT: \t OutCrd:", self.ocrd, "\t Out Ref1:", self.oref1, "\t Out Ref2:", self.oref2,
                  "\t Crd1:", self.curr_crd1, "\t Ref1:", self.curr_ref1,
                  "\t Crd2:", self.curr_crd2, "\t Ref2", self.curr_ref2,  "\t Intersection rate: ",
                  self.count / self.total_count  if self.total_count > 1 else 0 )

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

    def return_intersection_rate(self):
        return self.count / self.total_count

    def print_intersection_rate(self):
        print("Total operations for intersection: ", self.total_count)
        return print("Intersection rate: ", self.count / self.total_count)


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

        self.obv = None
        self.oref1 = None
        self.oref2 = None

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
            self.obv = self.curr_bv1
            self.oref1 = self.curr_bv1
            self.oref2 = self.curr_bv2
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
                  self.count / self.total_count if self.total_count > 1 else 0  )

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
        return self.count / self.total_count

    def print_intersection_rate(self):
        print("Total operations for intersection: ", self.total_count)
        return print("Intersection rate: ", self.count / self.total_count)
