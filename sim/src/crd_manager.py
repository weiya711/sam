from .base import *

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

        if len(self.inner_crd) > 0:
            icrd = self.inner_crd.pop()
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
                assert(icrd == 'D')
                self.curr_crd = 'D'
                self.get_next_icrd = False
                self.get_next_ocrd = False
            else:
                # FIXME, need to figure out multiple 'S's in a row here
                self.curr_crd = ''
                self.get_next_icrd = False
                self.get_next_ocrd = True
        else:
            self.curr_crd = ''

        if self.debug:
            print("Curr OCrd:", self.curr_crd, "\t Curr OutputCrd:", self.curr_crd, "\tHasCrd", self.has_crd,
                  "\t GetNext InnerCrd:", self.get_next_icrd, "\t GetNext OuterCrd:", self.get_next_ocrd)

    def set_outer_crd(self, crd):
        if crd != '':
            self.outer_crd.append(crd)

    def set_inner_crd(self, crd):
        if crd != '':
            self.inner_crd.append(crd)

    def out_crd_outer(self):
        return self.curr_crd
