import pytest
import scipy.sparse

from sam.sim.src.rd_scanner import CompressedCrdRdScan, UncompressCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce

from sam.sim.test.test import *

import os

cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figure out what formats we want to test for the chip
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_mat_mul_ijk_csr_full(ssname, debug_sim, fill=0):
    # filename = os.path.join(formatted_dir, ssname+"_"+"csr.txt")
    # formats = ['d', 's']
    # [B_shape, B0_dim, (B1_seg, B1_crd), B_vals] = read_combined_inputs(filename, formats)
    #
    # filename = os.path.join(formatted_dir, ssname+"_"+"trans_shifted_csc.txt")
    # formats = ['d', 's']
    # [C_shape, C1_dim, (C0_seg, C0_crd), C_vals] = read_combined_inputs(filename, formats)

    # CSR
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_dim = B_shape[0]

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B1_seg = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B1_crd = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    # CSC
    C_dirname = os.path.join(formatted_dir, ssname, "shift-trans", "ds10")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C1_dim = C_shape[1]

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C0_seg = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C0_crd = read_inputs(C0_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    if debug_sim:
        print("Mat B:", B_shape, B0_dim, B1_seg, B1_crd, B_vals)
        print("Mat C:", C_shape, C1_dim, C0_seg, C0_crd, C_vals)

    B_scipy = scipy.sparse.csr_matrix((B_vals, B1_crd, B1_seg), shape=B_shape)
    C_scipy = scipy.sparse.csc_matrix((C_vals, C0_crd, C0_seg), shape=C_shape)

    B_nd = B_scipy.toarray()
    C_nd = C_scipy.toarray()
    gold_nd = B_nd @ C_nd
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Dense Mat1:\n", B_nd)
        print("Dense Mat2:\n", C_nd)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    rdscan_Bi = UncompressCrdRdScan(dim=B0_dim, debug=debug_sim)
    rdscan_Bk = CompressedCrdRdScan(crd_arr=B1_crd, seg_arr=B1_seg, debug=debug_sim)
    val_B = Array(init_arr=B_vals, debug=debug_sim)

    rdscan_Cj = UncompressCrdRdScan(dim=C1_dim, debug=debug_sim)
    rdscan_Ck = CompressedCrdRdScan(crd_arr=C0_crd, seg_arr=C0_seg, debug=debug_sim)
    val_C = Array(init_arr=C_vals, debug=debug_sim)

    repsiggen_Bi = RepeatSigGen(debug=debug_sim)
    repsiggen_Cj = RepeatSigGen(debug=debug_sim)
    repeat_Ci = Repeat(debug=debug_sim)
    repeat_Bj = Repeat(debug=debug_sim)
    inter1 = Intersect2(debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    reduce = Reduce(debug=debug_sim)

    # drop = CrdDrop(debug=debug_sim)
    vals_X = ValsWrScan(size=B0_dim * C1_dim, fill=fill, debug=debug_sim)
    wrscan_Xi = CompressWrScan(seg_size=2, size=B0_dim, fill=fill)
    wrscan_Xj = CompressWrScan(seg_size=B0_dim + 1, size=B0_dim * B0_dim, fill=fill)

    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        # Input iteration for i
        if len(in_ref_B) > 0:
            rdscan_Bi.set_in_ref(in_ref_B.pop(0))
        rdscan_Bi.update()

        repsiggen_Bi.set_istream(rdscan_Bi.out_crd())
        repsiggen_Bi.update()

        repeat_Ci.set_in_repeat(repsiggen_Bi.out_repeat())
        if len(in_ref_C) > 0:
            repeat_Ci.set_in_ref(in_ref_C.pop(0))
        repeat_Ci.update()

        # Input iteration for j
        rdscan_Cj.set_in_ref(repeat_Ci.out_ref())
        rdscan_Cj.update()

        repsiggen_Cj.set_istream(rdscan_Cj.out_crd())
        repsiggen_Cj.update()

        repeat_Bj.set_in_repeat(repsiggen_Cj.out_repeat())
        repeat_Bj.set_in_ref(rdscan_Bi.out_ref())
        repeat_Bj.update()

        # Input iteration for k
        rdscan_Bk.set_in_ref(repeat_Bj.out_ref())
        rdscan_Bk.update()

        rdscan_Ck.set_in_ref(rdscan_Cj.out_ref())
        rdscan_Ck.update()

        inter1.set_in1(rdscan_Bk.out_ref(), rdscan_Bk.out_crd())
        inter1.set_in2(rdscan_Ck.out_ref(), rdscan_Ck.out_crd())
        inter1.update()

        # Computation

        val_B.set_load(inter1.out_ref1())
        val_B.update()
        val_C.set_load(inter1.out_ref2())
        val_C.update()

        mul.set_in1(val_B.out_load())
        mul.set_in2(val_C.out_load())
        mul.update()

        reduce.set_in_val(mul.out_val())
        reduce.update()

        vals_X.set_input(reduce.out_val())
        vals_X.update()

        wrscan_Xi.set_input(rdscan_Bi.out_crd())
        wrscan_Xi.update()

        wrscan_Xj.set_input(rdscan_Cj.out_crd())
        wrscan_Xj.update()

        if time % 100 == 0:
            print("Timestep", time, "\t Done --",
                  "\nRdScan Bi:", rdscan_Bi.out_done(), "\tRepeat Ci:", repeat_Ci.out_done(),
                  "\tRepSigGen Bi:", repsiggen_Bi.out_done(),
                  "\nRepeat Bj:", repeat_Bj.out_done(), "\tRdScan Cj:", rdscan_Cj.out_done(),
                  "\tRepSigGen Cj:", repsiggen_Cj.out_done(),
                  "\nRdScan Bk:", rdscan_Bk.out_done(), "\tRdScan Ck:", rdscan_Ck.out_done(),
                  "\tInter k:", inter1.out_done(),
                  "\nArr:", val_B.out_done(), val_C.out_done(),
                  "\tMul:", mul.out_done(),
                  "\tRed:", reduce.out_done(),
                  "\nVals X:", vals_X.out_done(),
                  "\tWrScan X1:", wrscan_Xi.out_done(), "\tWrScan X2:", wrscan_Xj.out_done(),
                  )

        done = wrscan_Xj.out_done() and wrscan_Xi.out_done() and vals_X.out_done()
        time += 1

    wrscan_Xi.autosize()
    wrscan_Xj.autosize()
    vals_X.autosize()

    out_crds = [wrscan_Xi.get_arr(), wrscan_Xj.get_arr()]
    out_segs = [wrscan_Xi.get_seg_arr(), wrscan_Xj.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print(out_segs)
        print(out_crds)
        print(out_val)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))
