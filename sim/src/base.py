from abc import ABC, abstractmethod


valid_tkns = ['', 'S', 'D']

#################
# Primitives
#################

class Primitive(ABC):
    def __init__(self, debug=False, **kwargs):
        self.done = False
        self.debug = debug

    def out_done(self):
        return self.done

    def is_debug(self):
        return self.debug

    @abstractmethod
    def update(self):
        pass


def remove_emptystr(l):
    return [x for x in l if x != '']

def remove_stoptkn(l):
    return [x for x in l if x != 'S']

def remove_donetkn(l):
    return [x for x in l if x != 'D']

