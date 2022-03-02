from abc import ABC, abstractmethod
from .base import Primitive


class Repeat(Primitive):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.in_ref = []
        self.in_repeat = []

        self.curr_out_ref = ''
        self.curr_in_ref = ''
        self.get_next_ref = True
        self.get_next_rep = True

    def update(self):
        if len(self.in_ref) > 0 and self.get_next_ref:
            self.curr_in_ref = self.in_ref.pop(0)
            if self.curr_in_ref == 'S':
                self.curr_out_ref = 'S'
                self.get_next_rep = False
            elif self.curr_in_ref == 'D':
                self.curr_out_ref = 'D'
                self.get_next_rep = True
                self.done = True
            else:
                self.get_next_rep = True
                self.get_next_ref = False
        elif self.get_next_ref:
            self.curr_out_ref = ''

        repeat = ''
        if len(self.in_repeat) > 0 and self.get_next_rep:
            repeat = self.in_repeat.pop(0)
            if repeat == 'S':
                self.get_next_ref = True
                self.curr_out_ref = 'S'
            elif repeat == 'D':
                if self.curr_out_ref != 'D':
                    raise Exception("Both repeat and ref signal need to end in 'D'")
                self.get_next_ref = True
                self.curr_out_ref = 'D'
            elif repeat == 'R':
                self.get_next_ref = False
                self.curr_out_ref = self.curr_in_ref
            else:
                raise Exception('Repeat signal cannot be: ' + str(repeat))
        elif self.get_next_rep:
            self.curr_out_ref = ''

        if self.debug:
            print("DEBUG: REPEAT:", "\t Get Ref:", self.get_next_ref, "\tIn Ref:", self.curr_in_ref,
                  "\t Get Rep:", self.get_next_rep, "\t Rep:", repeat,
                  "\t Out Ref:", self.curr_out_ref)

    def set_in_ref(self, ref):
        if ref != '':
            self.in_ref.append(ref)

    def set_in_repeat(self, repeat):
        if repeat != '':
            self.in_repeat.append(repeat)

    def out_ref(self):
        return self.curr_out_ref


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
