import pytest
import random
from sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
from sim.src.wr_scanner import ValsWrScan, CompressWrScan
from sim.src.joiner import Intersect2
from sim.src.compute import Multiply2
from sim.src.array import Array

TIMEOUT = 5000

@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_vec_elemmul_u_u_u(dim1, max_val=1000, size=100, fill=0):
    debug = False

    in_vec1 = [random.randint(0, max_val) for _ in range(dim1)]
    in_vec2 = [random.randint(0, max_val) for _ in range(dim1)]

    if debug:
        print("VECTOR 1:", in_vec1)
        print("VECTOR 2:", in_vec2)

    assert(len(in_vec1) == len(in_vec2))

    gold_vec = [in_vec1[i] * in_vec2[i] for i in range(len(in_vec1))]
    in_vec1 += ['S', 'D']
    in_vec2 += ['S', 'D']

    rdscan = UncompressRdScan(dim=dim1, debug=debug)
    val1 = Array(init_arr=in_vec1, debug=debug)
    val2 = Array(init_arr=in_vec2, debug=debug)
    mul = Multiply2(debug=debug)
    wrscan = ValsWrScan(size=size, fill=fill, debug=debug)

    in_ref = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref) > 0:
            rdscan.set_in_ref(in_ref.pop(0))
        rdscan.update()
        val1.set_load(rdscan.out_ref())
        val2.set_load(rdscan.out_ref())
        val1.update()
        val2.update()
        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())
        mul.update()
        wrscan.set_input(mul.out_val())
        wrscan.update()
        print("Timestep", time, "\t Done --", "\tRdScan:", rdscan.out_done(), "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tWrScan:", wrscan.out_done())
        done = wrscan.out_done()
        time += 1

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (wrscan.get_arr() == gold_vec + [fill]*(size-len(gold_vec)))

    # Assert the array stores only the values
    wrscan.resize_arr(len(gold_vec))
    assert (wrscan.get_arr() == gold_vec)

@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_vec_elemmul_u_c_c(nnz, debug_sim, max_val=1000, size=1001, fill=0):
    assert(size > max_val)

    crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr2 = sorted(set(crd_arr2))
    seg_arr2 = [0, len(crd_arr2)]
    vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug_sim:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    out_crd = sorted(set(crd_arr1) & set(crd_arr2))
    if out_crd:
        out_val = [vals_arr1[crd_arr1.index(i)]*vals_arr2[crd_arr2.index(i)] for i in out_crd]
        gold_vec = [out_val[out_crd.index(i)] if i in out_crd else 0 for i in range(max(out_crd)+1)]
    else:
        out_val = ''
        gold_vec = [fill] * size

    if debug_sim:
        print("Compressed RESULT  :\n", out_crd, "\n", out_val, "\n", gold_vec)

    crdscan1 = CompressedRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug_sim)
    crdscan2 = CompressedRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug_sim)
    inter = Intersect2(debug=debug_sim)
    val1 = Array(init_arr=vals_arr1, debug=debug_sim)
    val2 = Array(init_arr=vals_arr2, debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    oval = Array(size=size, fill=fill, debug=debug_sim)
    wrscan = ValsWrScan(size=size, fill=fill, debug=debug_sim)

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0))
        crdscan1.update()
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0))
        crdscan2.update()
        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd())
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd())
        inter.update()
        val1.set_load(inter.out_ref1())
        val2.set_load(inter.out_ref2())
        val1.update()
        val2.update()
        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())
        mul.update()
        wrscan.set_input(mul.out_val())
        wrscan.update()
        oval.set_store(inter.out_crd(), mul.out_val())
        oval.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\tInter:", inter.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(), "\tOutVal:", oval.out_done())
        done = oval.out_done()
        time += 1

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (oval.get_arr() == gold_vec + [fill]*(size-len(gold_vec)))
    # Assert the array stores only the values
    oval.resize(len(gold_vec))
    assert (oval.get_arr() == gold_vec)

    if out_val:
        # Assert the array stores values with the rest of the memory initialized to initial value
        assert (wrscan.get_arr() == out_val + [fill]*(size-len(out_val)))
        # Assert the array stores only the values
        wrscan.resize_arr(len(out_val))
        assert (wrscan.get_arr() == out_val)


@pytest.mark.parametrize("nnz", [1, 10, 100, 500, 1000])
def test_vec_elemmul_c_c_c(nnz, max_val=1000, size=1001, fill=0):
    assert(size > max_val)
    debug = False

    crd_arr1 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr1 = sorted(set(crd_arr1))
    seg_arr1 = [0, len(crd_arr1)]
    vals_arr1 = [random.randint(0, max_val) for _ in range(len(crd_arr1))]

    crd_arr2 = [random.randint(0, max_val) for _ in range(nnz)]
    crd_arr2 = sorted(set(crd_arr2))
    seg_arr2 = [0, len(crd_arr2)]
    vals_arr2 = [random.randint(0, max_val) for _ in range(len(crd_arr2))]

    if debug:
        print("Compressed VECTOR 1:\n", seg_arr1, "\n", crd_arr1, "\n", vals_arr1)
        print("Compressed VECTOR 2:\n", seg_arr2, "\n", crd_arr2, "\n", vals_arr2)

    gold_crd = sorted(set(crd_arr1) & set(crd_arr2))
    gold_seg = [0, len(gold_crd)]
    gold_vals = []
    if gold_crd:
        gold_vals = [vals_arr1[crd_arr1.index(i)]*vals_arr2[crd_arr2.index(i)] for i in gold_crd]

    if debug:
        print("Compressed RESULT  :\n", gold_seg, "\n", gold_crd, "\n", gold_vals)

    crdscan1 = CompressedRdScan(seg_arr=seg_arr1, crd_arr=crd_arr1, debug=debug)
    crdscan2 = CompressedRdScan(seg_arr=seg_arr2, crd_arr=crd_arr2, debug=debug)
    inter = Intersect2(debug=debug)
    val1 = Array(init_arr=vals_arr1, debug=debug)
    val2 = Array(init_arr=vals_arr2, debug=debug)
    mul = Multiply2(debug=debug)
    oval_wrscan = ValsWrScan(size=size, fill=fill)
    ocrd_wrscan = CompressWrScan(size=size, fill=fill)

    in_ref1 = [0, 'D']
    in_ref2 = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref1) > 0:
            crdscan1.set_in_ref(in_ref1.pop(0))
        crdscan1.update()
        if len(in_ref2) > 0:
            crdscan2.set_in_ref(in_ref2.pop(0))
        crdscan2.update()
        inter.set_in1(crdscan1.out_ref(), crdscan1.out_crd())
        inter.set_in2(crdscan2.out_ref(), crdscan2.out_crd())
        inter.update()
        val1.set_load(inter.out_ref1())
        val2.set_load(inter.out_ref2())
        val1.update()
        val2.update()
        mul.set_in1(val1.out_load())
        mul.set_in2(val2.out_load())
        mul.update()
        oval_wrscan.set_input(mul.out_val())
        oval_wrscan.update()
        ocrd_wrscan.set_input(inter.out_crd())
        ocrd_wrscan.update()

        print("Timestep", time, "\t Done --",
              "\tRdScan1:", crdscan1.out_done(), "\tRdScan2:", crdscan2.out_done(),
              "\tInter:", inter.out_done(),
              "\tArr:", val1.out_done(), val2.out_done(),
              "\tMul:", mul.out_done(),
              "\tOutVal:", oval_wrscan.out_done(),
              "\tOutCrd:", ocrd_wrscan.out_done())
        done = ocrd_wrscan.out_done()
        time += 1

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (oval_wrscan.get_arr() == gold_vals + [fill]*(size-len(gold_vals)))
    # Assert the array stores only the values
    oval_wrscan.resize_arr(len(gold_vals))
    assert (oval_wrscan.get_arr() == gold_vals)

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (ocrd_wrscan.get_arr() == gold_crd + [fill]*(size-len(gold_crd)))
    # Assert the array stores only the values
    ocrd_wrscan.resize_arr(len(gold_crd))
    assert (ocrd_wrscan.get_arr() == gold_crd)

    # Assert the array stores values with the rest of the memory initialized to initial value
    assert (ocrd_wrscan.get_seg_arr() == gold_seg + [fill]*(size-len(gold_seg)))
    # Assert the array stores only the values
    ocrd_wrscan.resize_seg_arr(len(gold_seg))
    assert (ocrd_wrscan.get_seg_arr() == gold_seg)


