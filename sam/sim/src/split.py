from .base import *


class Split(Primitive):
    def __init__(self, split_factor=4, orig_crd=True, **kwargs):
        super().__init__(**kwargs)

        self.in_crd = []
        self.in_crd_size = 0

        self.curr_ocrd = None
        self.curr_icrd = None

        self.prev_ocrd = None
        self.prev_icrd = None

        self.prev_stkn = True
        self.emit_stkn = False
        self.emit_done = False
        self.prev_cntr = None
        self.cntr = 0

        self.split_factor = split_factor
        self.orig_crd = orig_crd

    def update(self):
        if self.done:
            self.curr_ocrd = ''
            self.curr_icrd = ''
        elif self.emit_done:
            self.curr_ocrd = 'D'
            self.curr_icrd = 'D'
            self.done = True
            self.emit_done = False
        elif self.emit_stkn:
            self.curr_ocrd = self.prev_ocrd
            self.curr_icrd = self.prev_icrd
            self.emit_stkn = False
        elif len(self.in_crd) > 0 and is_stkn(self.in_crd[0]):
            crd = self.in_crd.pop(0)
            assert is_stkn(crd)
            self.curr_ocrd = crd
            self.curr_icrd = increment_stkn(crd)
            self.prev_stkn = True
            self.emit_stkn = False
        elif len(self.in_crd) > 0 and not is_stkn(self.in_crd[0]):
            crd = self.in_crd.pop(0)
            if isinstance(crd, int):
                self.cntr = int(crd / self.split_factor)

                if self.prev_cntr != self.cntr and self.prev_cntr is not None and not self.prev_stkn:
                    self.curr_ocrd = ''
                    self.curr_icrd = 'S0'
                    self.prev_ocrd = self.cntr
                    self.prev_icrd = crd if self.orig_crd else crd % self.split_factor
                    self.emit_stkn = True
                elif self.prev_cntr != self.cntr:
                    self.curr_ocrd = self.cntr
                    self.curr_icrd = crd if self.orig_crd else crd % self.split_factor
                else:
                    self.curr_ocrd = ''
                    self.curr_icrd = crd if self.orig_crd else crd % self.split_factor

                self.prev_stkn = False
                self.prev_cntr = self.cntr
            else:
                if self.prev_stkn:
                    self.curr_ocrd = 'D'
                    self.curr_icrd = 'D'
                    self.done = True
                else:
                    self.curr_ocrd = 'S0'
                    self.curr_icrd = 'S1'
                    self.emit_done = True
        else:
            self.curr_ocrd = ''
            self.curr_icrd = ''

        if self.debug:
            print("DEBUG: SPLIT:", "\t InCrd:", self.in_crd,
                  "\n Curr Ocrd:", self.curr_ocrd, "\t Curr Icrd:", self.curr_icrd,
                  "\n Prev Cntr", self.prev_cntr, "\t Curr Cntr:", self.cntr,
                  "\n Emit Stkn:", self.emit_stkn, "\t Prev Stkn:", self.prev_stkn)

    def set_in_crd(self, crd):
        if crd != '':
            self.in_crd.append(crd)

    def out_outer_crd(self):
        return self.curr_ocrd

    def out_inner_crd(self):
        return self.curr_icrd

    def compute_fifos(self):
        self.in_crd_size = max(self.in_crd_size, len(self.in_crd))

    def print_fifos(self):
        print("FIFOs size in the crd for split blocks: ", self.in_crd_size)
