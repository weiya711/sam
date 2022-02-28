from sim.src.wr_scanner import WrScan, CompressWrScan
from sim.src.array import Array

TIMEOUT = 5000


def check_arr(arr_obj, gold):
    assert (isinstance(arr_obj, WrScan) or isinstance(arr_obj, Array))
    # Assert the array stores values with the rest of the memory initialized to initial value
    print("out", arr_obj.get_arr())
    print("gold", gold + [arr_obj.fill] * (arr_obj.size - len(gold)))
    assert (arr_obj.get_arr() == gold + [arr_obj.fill] * (arr_obj.size - len(gold)))

    # Assert the array stores only the values
    if isinstance(arr_obj, WrScan):
        arr_obj.resize_arr(len(gold))
        print("New size", arr_obj.size)
        if isinstance(arr_obj, CompressWrScan):
            print("Seg size", arr_obj.seg_size)
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
