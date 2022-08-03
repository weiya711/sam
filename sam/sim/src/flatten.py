from .base import *


class Flatten(Primitive):
    def __init__(self, split_factor=4, **kwargs):
        super().__init__(**kwargs)

        self.in_outer_crd = []
        self.in_inner_crd = []
        
        if self.get_stats:
            self.in_inner_crd_size = 0
            self.in_outer_crd_size = 0

        self.curr_crd = None
        self.curr_ocrd = None
        self.get_inner = True
        self.get_outer = True

        self.split_factor = split_factor

    def update(self):
        self.update_done()
        if len(self.in_inner_crd) > 0 or len(self.in_outer_crd) > 0:
            self.block_start = False

        if self.done:
            self.curr_crd = ''
            return

        if self.get_inner and len(self.in_inner_crd) == 0:
            self.curr_crd = ''
            return

        icrd = self.in_inner_crd.pop(0) if len(self.in_inner_crd) > 0 and self.get_inner else None
        self.get_outer |= not isinstance(icrd, int)

        if self.get_outer and len(self.in_outer_crd) == 0:
            self.curr_crd = ''
            return

        self.curr_ocrd = self.in_outer_crd.pop(0) if len(self.in_outer_crd) > 0 and self.get_outer else self.curr_ocrd
        self.get_outer = False

        if isinstance(icrd, int) and isinstance(self.curr_ocrd, int):
            self.curr_crd = self.curr_ocrd * self.split_factor + icrd
        elif is_stkn(icrd) and is_stkn(self.curr_ocrd):
            self.curr_crd = self.curr_ocrd
            self.get_outer = True
        elif icrd == 'D':
            self.curr_crd = icrd
            self.done = True

            self.get_inner = False
            self.get_outer = False
        else:
            self.curr_crd = ''

        if self.debug:
            print("DEBUG: FLATTEN: \n",
                  "\t Curr Icrd:", icrd, "\t Curr Ocrd", self.curr_ocrd,
                  "\t Get Icrd:", self.get_inner, "\t Get Ocrd", self.get_outer,
                  "\n Out Crd:", self.curr_crd
                  )

    def set_in_inner_crd(self, crd):
        if crd != '' and crd is not None:
            self.in_inner_crd.append(crd)

    def set_in_outer_crd(self, crd):
        if crd != '' and crd is not None:
            self.in_outer_crd.append(crd)

    def out_crd(self):
        return self.curr_crd

    def compute_fifos(self):
        if self.get_stats:
            self.in_inner_crd_size = max(self.in_inner_crd_size, len(self.in_inner_crd))
            self.in_outer_crd_size = max(self.in_outer_crd_size, len(self.in_outer_crd))

    def print_fifos(self):
        print("FIFOs size in the inner crd for flatten blocks: ", self.in_inner_crd_size)
        print("FIFOs size in the outer crd for flatten blocks: ", self.in_outer_crd_size)
