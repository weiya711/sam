import numpy as np
import pytest
import random
from .src.primitive import UncompressRdScan, CompressedRdScan

########################################
# Uncompressed Read Scanner Unit Tests #
########################################
@pytest.mark.parametrize("dim", [1, 2, 4, 16])
def test_rd_scan_uncompress_1d(dim):
    gold_crd = [x for x in range(dim)]
    gold_crd.append('S')
    gold_crd.append('D')
    gold_ref = gold_crd
    assert(len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim)

    in_ref = [0, 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))
        urs.update()
        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())
        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())
        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)

@pytest.mark.parametrize("dim", [1, 2, 4])
def test_rd_scan_uncompress_rd_scan_2d(dim):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S'] )*4 + ['S', 'D']
    gold_ref = cnt + ['S'] + [dim + x for x in cnt] + ['S'] + [2*dim + x for x in cnt] +['S'] + \
               [3*dim + x for x in cnt] + ['S'] + ['S', 'D']
    assert(len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim)

    in_ref = [0, 1, 2, 3, 'S', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))
        urs.update()
        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())
        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())
        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)

def test_rd_scan_uncompress_3d(dim=4):
    cnt = [x for x in range(dim)]
    gold_crd = (cnt + ['S'])*3 + ['S'] + (cnt + ['S'])*2 +  ['S']+ (cnt + ['S'])*2 + ['S'] + ['S', 'D']
    gold_ref = cnt + ['S'] + [dim + x for x in cnt] + ['S'] + [2*dim + x for x in cnt] +['S'] + ['S'] + \
               [3*dim + x for x in cnt] + ['S'] + [4*dim + x for x in cnt] + ['S', 'S'] + \
               [5*dim + x for x in cnt] + ['S'] + \
               [6*dim + x for x in cnt] + ['S'] + ['S'] + ['S', 'D']
    assert(len(gold_crd) == len(gold_ref))

    urs = UncompressRdScan(dim)

    in_ref = [0, 1, 2, 'S', 3, 4, 'S', 5, 6, 'S', 'S', 'D']
    done = False
    time = 0
    out_crd = []
    out_ref = []
    while not done:
        if len(in_ref) > 0:
            urs.set_in_ref(in_ref.pop(0))
        urs.update()
        print("Timestep", time, "\t Crd:", urs.out_crd(), "\t Ref:", urs.out_ref())
        out_crd.append(urs.out_crd())
        out_ref.append(urs.out_ref())
        done = urs.done
        time += 1

    assert (out_crd == gold_crd)
    assert (out_ref == gold_ref)


######################################
# Compressed Read Scanner Unit Tests #
######################################