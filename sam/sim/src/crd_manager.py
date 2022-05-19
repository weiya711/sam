from .base import *

from .repeater import RepeatSigGen, Repeat


class CrdDrop(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.outer_crd = []
        self.inner_crd = []
        self.inner_crd_fifo = 0
        self.outer_crd_fifo = 0
        self.curr_inner_crd = ''
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
            self.outer_crd_fifo = max(self.outer_crd_fifo, len(self.outer_crd))
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
            self.inner_crd_fifo = max(self.inner_crd_fifo, len(self.inner_crd))
            icrd = self.inner_crd.pop(0)
            self.curr_inner_crd = icrd
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
            print("DEBUG: CRDDROP: Curr OuterCrd:", self.curr_ocrd, "\tCurr InnerCrd:", icrd,
                  "\t Curr OutputCrd:", self.curr_crd, "\tHasCrd", self.has_crd,
                  "\t GetNext InnerCrd:", self.get_next_icrd, "\t GetNext OuterCrd:", self.get_next_ocrd)

    def set_outer_crd(self, crd):
        if crd != '':
            self.outer_crd.append(crd)

    def set_inner_crd(self, crd):
        if crd != '':
            self.inner_crd.append(crd)

    def out_crd_outer(self):
        return self.curr_crd

    def out_crd_inner(self):
        return self.curr_inner_crd

    def print_fifos(self):
        print("Crdrop Inner crd fifos size: ", self.inner_crd_fifo)
        print("CrdDrop Outer crd fifo size: ", self.outer_crd_fifo)


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

        if len(self.inner_crd) > 0:
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
    def __init__(self, last_level=False, **kwargs):
        super().__init__(**kwargs)

        self.outer_crdpt = []
        self.inner_crdpt = []

        self.curr_ocrd = None
        self.curr_icrd = None

        self.prev_ocrd = None
        self.prev_ocrdpt = None
        self.prev_icrdpt = None

        self.emit_stkn = False
        self.emit_done = False
        self.prev_stkn = False
        self.waiting_next = False

        self.inner_last_level = last_level



    def update(self):

        if self.curr_ocrd != '':
            self.prev_ocrd = self.curr_ocrd

        if self.done:
            print("case 1")
            self.curr_ocrd = ''
            self.curr_icrd = ''
        # elif self.emit_stkn and len(self.outer_crdpt) > 0 and is_stkn(self.outer_crdpt[0]):
        #     # Increment stop token
        #     curr_ocrdpt = self.outer_crdpt.pop(0)
        #     self.curr_ocrd = self._next_done(curr_ocrdpt)
        #     self.curr_icrd = self._next_done(increment_stkn(curr_ocrdpt)) if self.inner_last_level \
        #         else increment_stkn(curr_ocrdpt)
        #     self.emit_stkn = False
        elif self.waiting_next and len(self.outer_crdpt) > 0:
            print("case 2")
            print(self.prev_ocrd)
            stkn = increment_stkn(self.prev_ocrd)
            self.curr_ocrd = stkn if self.outer_crdpt[0] == 'D' else self.prev_ocrd
            self.curr_icrd = increment_stkn(stkn) if self.inner_last_level and self.outer_crdpt[0] == 'D' else stkn
            self.waiting_next = False
        elif self.emit_stkn:
            print("case 3")

            # Emit innermost level stop token
            self.curr_ocrd = self.prev_ocrdpt
            self.curr_icrd = self.prev_icrdpt

            self.emit_stkn = False
        elif len(self.outer_crdpt) > 0 and is_stkn(self.outer_crdpt[0]):
            print("case 6")

            # Just forward stop token if self.emit_tkn = False
            curr_ocrdpt = self.outer_crdpt.pop(0)

            if len(self.outer_crdpt) > 0:
                next_outer = self.outer_crdpt[0]

                if next_outer == 'D':
                    assert self.inner_crdpt[0] == 'D', "Done tokens must be aligned"
                    self.curr_ocrd = increment_stkn(curr_ocrdpt)
                    self.curr_icrd = increment_stkn(increment_stkn(curr_ocrdpt)) if self.inner_last_level \
                        else increment_stkn(curr_ocrdpt)

                else:
                    self.curr_ocrd = curr_ocrdpt
                    self.curr_icrd = increment_stkn(curr_ocrdpt)
            else:
                self.curr_ocrd = ''
                self.curr_icrd = ''
                self.waiting_next = True
                self.prev_ocrd = curr_ocrdpt

            self.emit_stkn = False
            self.prev_stkn = True
        elif self.emit_done:
            print("case 4")

            self.curr_ocrd = 'D'
            self.curr_icrd = 'D'
            self.done = True
            self.emit_done = False
        elif len(self.outer_crdpt) > 0 and len(self.inner_crdpt) > 0 and \
                isinstance(self.outer_crdpt[0], int) and isinstance(self.inner_crdpt[0], int):

            # Both streams are coordinates
            curr_ocrdpt = self.outer_crdpt.pop(0)
            curr_icrdpt = self.inner_crdpt.pop(0)

            if self.prev_ocrdpt != curr_ocrdpt and self.prev_ocrdpt is not None and not self.prev_stkn:
                print("case 5a")
                self.curr_ocrd = ''
                self.curr_icrd = 'S0'
                self.emit_stkn = True
            elif self.prev_ocrdpt != curr_ocrdpt:
                print("case 5b")
                self.curr_icrd = curr_icrdpt
                self.curr_ocrd = curr_ocrdpt
            else:
                print("case 5v")
                self.curr_icrd = curr_icrdpt
                self.curr_ocrd = ''
            self.prev_stkn = False

            self.prev_icrdpt = curr_icrdpt
            self.prev_ocrdpt = curr_ocrdpt
        elif len(self.inner_crdpt) > 0 and is_stkn(self.inner_crdpt[0]):
            assert False, "The inner crdpt stream should not have stop tokens"

        elif len(self.outer_crdpt) > 0 and len(self.inner_crdpt) > 0 and \
                self.outer_crdpt[0] == 'D' and self.inner_crdpt[0] == 'D':
            print("case 8")

            if isinstance(self.prev_ocrd, int):
                self.curr_ocrd = 'S0'
                self.curr_icrd = increment_stkn('S0') if self.inner_last_level else 'S0'
                self.emit_done = True
            else:
                self.curr_ocrd = 'D'
                self.curr_icrd = 'D'
                self.done = True
            # elif is_stkn(self.prev_ocrd):
            #     stkn = increment_stkn(self.prev_ocrd)
            #     self.curr_ocrd = stkn
            #     self.curr_icrd = increment_stkn(stkn) if self.inner_last_level else stkn

        else:
            print("case 9")
            self.curr_ocrd = ''
            self.curr_icrd = ''

        if self.debug:
            print("DEBUG: CrdPtConverter \t Done:", self.out_done(),
                  "\n Curr in ocrd: ", self.inner_crdpt, "\t Curr in icrd", self.outer_crdpt,
                  "\t Curr in val", self.prev_ocrdpt, "\t Emit Tkn: ", self.emit_stkn)

    def set_outer_crdpt(self, crdpt):
        if crdpt != '':
            self.outer_crdpt.append(crdpt)

    def set_inner_crdpt(self, crdpt):
        if crdpt != '':
            self.inner_crdpt.append(crdpt)

    def out_crd_outer(self):
        return self.curr_ocrd

    def out_crd_inner(self):
        return self.curr_icrd
