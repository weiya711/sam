from abc import ABC, abstractmethod


def gen_stkns(dim=10):
    return ['S' + str(i) for i in range(dim)]


valid_tkns = ['', 'D']
valid_tkns += gen_stkns()


def is_stkn(elem):
    if isinstance(elem, str):
        return elem.startswith('S') and (len(elem) == 2)
    return False


def get_stkn_order(elem):
    assert is_stkn(elem)
    return int(elem[1])


def increment_stkn(elem):
    return 'S' + str(get_stkn_order(elem) + 1)


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

    def reset(self):
        self.done = False


def remove_emptystr(l):
    return [x for x in l if x != '']


def remove_stoptkn(l):
    return [x for x in l if x != 'S']


def remove_donetkn(l):
    return [x for x in l if x != 'D']
