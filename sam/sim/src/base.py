from abc import ABC, abstractmethod


def gen_stkns(dim=10):
    return ['S' + str(i) for i in range(dim)]


valid_tkns = ['', 'D', 'N']
valid_tkns += gen_stkns()


def is_valid_crd(elem, dim=10):
    valid_tkns = ['', 'D'] + gen_stkns(dim)
    return isinstance(elem, int) or elem in valid_tkns


def is_valid_ref(elem, dim=10):
    valid_tkns = ['', 'D', 'N'] + gen_stkns(dim)
    return isinstance(elem, int) or elem in valid_tkns


def is_valid_crdpt(elem):
    valid_tkns = ['', 'D']
    return isinstance(elem, int) or elem in valid_tkns


def is_valid_val(elem, dim=10):
    valid_tkns = ['', 'D'] + gen_stkns(dim)
    return isinstance(elem, int) or isinstance(elem, float) or elem in valid_tkns


def is_0tkn(elem):
    return elem == 'N'


def is_stkn(elem):
    if isinstance(elem, str):
        return elem.startswith('S') and (len(elem) == 2)
    return False


def stkn_order(elem):
    assert is_stkn(elem)
    return int(elem[1])


def increment_stkn(elem):
    if elem == '':
        return 'S0'
    return 'S' + str(stkn_order(elem) + 1)


def decrement_stkn(elem):
    assert (stkn_order(elem) >= 0)
    if stkn_order(elem) > 0:
        return 'S' + str(stkn_order(elem) - 1)
    else:
        return ''


def smaller_stkn(a, b):
    return a if stkn_order(a) < stkn_order(b) else b


def larger_stkn(a, b):
    return a if stkn_order(a) > stkn_order(b) else b


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


def remove_emptystr(stream):
    return [x for x in stream if x != '']


def remove_stoptkn(stream):
    return [x for x in stream if x != 'S']


def remove_donetkn(stream):
    return [x for x in stream if x != 'D']


# ----------- Bitvector ------------------
def right_bit_set(bits):
    pos = 0
    m = 1

    while not (bits & m):
        # left shift
        m = m << 1
        pos += 1

    return 1 << pos


def popcount(bits):
    return bin(bits).count('1')


def get_nth_bit(bits, n):
    assert popcount(bits) > n
    rbit = None
    for i in range(n + 1):
        rbit = right_bit_set(bits)
        bits &= ~rbit
    return rbit
