import copy
import random

import pytest

from sim.src.wr_scanner import WrScan, CompressWrScan
from sim.src.array import Array

TIMEOUT = 5000


def check_arr(arr_obj, gold):
    assert (isinstance(arr_obj, WrScan) or isinstance(arr_obj, Array))
    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (arr_obj.get_arr() == gold + [arr_obj.fill] * (arr_obj.size - len(gold)))
    # Assert the array stores only the values
    if isinstance(arr_obj, WrScan):
        arr_obj.resize_arr(len(gold))
    else:
        arr_obj.resize(len(gold))
    assert (arr_obj.get_arr() == gold)


def check_seg_arr(cwrscan, gold):
    assert (isinstance(cwrscan, CompressWrScan))
    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (cwrscan.get_seg_arr() == (gold + [0] * (cwrscan.seg_size - len(gold))))
    # Assert the array stores only the values
    cwrscan.resize_seg_arr(len(gold))
    assert (cwrscan.get_seg_arr() == gold)


def gen_stream(n=1, max_val=10, max_nnz=10):
    assert(max_val >= max_nnz)

    num_s = 0
    l = []
    end = False
    while not end:
        t = []
        num_v = random.randint(1, max_nnz-1)
        for _ in range(num_v):
            t.append(random.randint(0, max_val))
            t = sorted(set(t))
        for el in t:
            l.append(el)

        num_s = random.randint(1, n)
        for _ in range(num_s):
            l.append('S')
        end = num_s == n

    for _ in range(n - num_s):
        l.append('S')

    l.append('D')
    return l

def dedup_adj(olist):
    l = copy.deepcopy(olist)
    for i in range(len(l)-1, 0, -1):
        if l[i] == l[i-1]:
            del l[i]
    return l