from .base import *

from .repeater import RepeatSigGen, Repeat


class CrdDrop(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outer_crd = []
        self.inner_crd = []

        self.curr_ocrd = ''
        self.curr_crd = ''
        self.has_crd = False
        self.get_next_icrd = False
        self.get_next_ocrd = True

    def update(self):
        icrd = ""
        if self.debug:
            print("OuterCrds:", self.outer_crd)
            print("InnerCrds:", self.inner_crd)

        if self.done:
            self.curr_crd = ''
            return

        if len(self.outer_crd) > 0 and self.get_next_ocrd:
            self.curr_ocrd = self.outer_crd.pop(0)
            if isinstance(self.curr_ocrd, int):
                self.get_next_icrd = True
                self.get_next_ocrd = False
            else:
                self.curr_crd = self.curr_ocrd
                self.get_next_icrd = False
                self.get_next_ocrd = True
                if self.curr_ocrd == 'D':
                    self.done = True
            self.has_crd = False
        elif self.get_next_ocrd:
            self.curr_crd = ''

        if len(self.inner_crd) > 0 and self.get_next_icrd:
            icrd = self.inner_crd.pop(0)
            if isinstance(icrd, int):
                self.has_crd = True
                self.curr_crd = ''
                self.get_next_ocrd = False
                self.get_next_icrd = True
            elif is_stkn(icrd) and is_stkn(self.curr_ocrd):
                self.get_next_ocrd = True
                self.curr_crd = self.curr_ocrd
                self.get_next_icrd = False
            elif is_stkn(icrd):
                self.get_next_ocrd = True
                self.curr_crd = self.curr_ocrd if self.has_crd else ''
                self.get_next_icrd = False
            elif self.done:
                assert (icrd == 'D')
                self.curr_crd = 'D'
                self.get_next_icrd = False
                self.get_next_ocrd = False
            else:
                self.curr_crd = ''
                self.get_next_icrd = False
                self.get_next_ocrd = True
        elif self.get_next_icrd:
            self.curr_crd = ''

        if self.debug:
            print("Curr OuterCrd:", self.curr_ocrd, "\tCurr InnerCrd:", icrd, "\t Curr OutputCrd:", self.curr_crd,
                  "\tHasCrd", self.has_crd,
                  "\t GetNext InnerCrd:", self.get_next_icrd, "\t GetNext OuterCrd:", self.get_next_ocrd)

    def set_outer_crd(self, crd):
        if crd != '':
            self.outer_crd.append(crd)

    def set_inner_crd(self, crd):
        if crd != '':
            self.inner_crd.append(crd)

    def out_crd_outer(self):
        return self.curr_crd


# Converts coordinate streams to point streams
class CrdHold(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outer_crd = []
        self.inner_crd = []

        self.repsig = []
        self.curr_crd = ''

        self.RSG = RepeatSigGen(debug=self.debug)
        self.repeat = Repeat(debug=self.debug)

    def update(self):
        if self.done:
            self.curr_crd = ''
            return

        if (len(self.inner_crd) > 0):
            icrd = self.inner_crd.pop(0)
            self.RSG.set_istream(icrd)
        self.RSG.update()
        self.repsig.append(self.RSG.out_repeat())

        if len(self.outer_crd) > 0:
            ocrd = self.outer_crd.pop(0)
            self.repeat.set_in_ref(ocrd)
        if len(self.repsig) > 0:
            self.repeat.set_in_repeat(self.repsig.pop(0))

        self.repeat.update()

        self.curr_crd = self.repeat.out_ref()

        self.done = self.RSG.done and self.repeat.done

    def set_outer_crd(self, crd):
        if crd != '':
            self.outer_crd.append(crd)

    def set_inner_crd(self, crd):
        if crd != '':
            self.inner_crd.append(crd)

    def out_crd_outer(self):
        return self.curr_crd

# Converts point streams back into coordinate streams
# Helper for the sparse accumulator
class CrdPtConverter(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.curr_ocrd = None

    def update(self):
        pass