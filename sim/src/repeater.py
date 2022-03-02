from abc import ABC, abstractmethod
from .base import Primitive


class Repeat(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_ref = []
        self.in_repeat = []


    def set_in_ref(self, ref):
        if ref != '':
            self.in_ref.append(ref)

    def set_in_repeat(self, repeat):
        if repeat != '':
            self.in_repeat.append(repeat)

    @abstractmethod
    def out_crd(self):
        pass

    @abstractmethod
    def out_fiber_end(self):
        pass


class RepeatSigGen(Primitive):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            self.istream = []
            self.curr_repeat = ''

        def update(self):
            istream = ''

            if len(self.istream) > 0:
                istream = self.istream.pop(0)
                if istream == 'S' and self.curr_repeat == 'R':
                    self.curr_repeat = 'S'
                elif istream == 'S':
                    self.curr_repeat = ''
                elif istream == 'D':
                    self.curr_repeat = 'D'
                    self.done = True
                else:
                    self.curr_repeat = 'R'
            else:
                self.curr_repeat = ''

            if self.debug:
                print("DEBUG: REP GEN:", "\t In:", istream, "\t Out:", self.curr_repeat)

        # input can either be coordinates or references
        def set_istream(self, istream):
            if istream != '':
                self.istream.append(istream)

        def out_repeat(self):
            return self.curr_repeat
