import pytest
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
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


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_matmul_ijk(ssname, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    C_dirname = os.path.join(formatted_dir, ssname, "shift-trans", "ds10")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Bi_17 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberwrite_X0_6 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_15 = RepeatSigGen(debug=debug_sim)
    repeat_Ci_14 = Repeat(debug=debug_sim)
    fiberlookup_Cj_13 = UncompressCrdRdScan(dim=C_shape[1], debug=debug_sim)
    fiberlookup_Ck_9 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    fiberwrite_X1_5 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[1], fill=fill, debug=debug_sim)
    repsiggen_j_11 = RepeatSigGen(debug=debug_sim)
    repeat_Bj_10 = Repeat(debug=debug_sim)
    fiberlookup_Bk_8 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    intersect_7 = Intersect2(debug=debug_sim)
    arrayvals_B_3 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_C_4 = Array(init_arr=C_vals, debug=debug_sim)
    mul_2 = Multiply2(debug=debug_sim)
    reduce_1 = Reduce(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * C_shape[1], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_17.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_17.update()

        fiberwrite_X0_6.set_input(fiberlookup_Bi_17.out_crd())
        fiberwrite_X0_6.update()

        repsiggen_i_15.set_istream(fiberlookup_Bi_17.out_crd())
        repsiggen_i_15.update()

        if len(in_ref_C) > 0:
            repeat_Ci_14.set_in_ref(in_ref_C.pop(0))
        repeat_Ci_14.set_in_repsig(repsiggen_i_15.out_repsig())
        repeat_Ci_14.update()

        fiberlookup_Cj_13.set_in_ref(repeat_Ci_14.out_ref())
        fiberlookup_Cj_13.update()

        fiberlookup_Ck_9.set_in_ref(fiberlookup_Cj_13.out_ref())
        fiberlookup_Ck_9.update()

        fiberwrite_X1_5.set_input(fiberlookup_Cj_13.out_crd())
        fiberwrite_X1_5.update()

        repsiggen_j_11.set_istream(fiberlookup_Cj_13.out_crd())
        repsiggen_j_11.update()

        repeat_Bj_10.set_in_ref(fiberlookup_Bi_17.out_ref())
        repeat_Bj_10.set_in_repsig(repsiggen_j_11.out_repsig())
        repeat_Bj_10.update()

        fiberlookup_Bk_8.set_in_ref(repeat_Bj_10.out_ref())
        fiberlookup_Bk_8.update()

        intersectk_7.set_in1(fiberlookup_Bk_8.out_ref(), fiberlookup_Bk_8.out_crd())
        intersectk_7.set_in2(fiberlookup_Ck_9.out_ref(), fiberlookup_Ck_9.out_crd())
        intersectk_7.update()

        arrayvals_B_3.set_load(intersectk_7.out_ref1())
        arrayvals_B_3.update()

        arrayvals_C_4.set_load(intersectk_7.out_ref2())
        arrayvals_C_4.update()

        mul_2.set_in1(arrayvals_B_3.out_load())
        mul_2.set_in2(arrayvals_C_4.out_load())
        mul_2.update()

        reduce_1.set_in_val(mul_2.out_val())
        reduce_1.update()

        fiberwrite_Xvals_0.set_input(reduce_1.out_val())
        fiberwrite_Xvals_0.update()

        done = fiberwrite_X0_6.out_done() and fiberwrite_X1_5.out_done() and fiberwrite_Xvals_0.out_done()
        time += 1

    fiberwrite_X0_6.autosize()
    fiberwrite_X1_5.autosize()
    fiberwrite_Xvals_0.autosize()

    out_crds = [fiberwrite_X0_6.get_arr(), fiberwrite_X1_5.get_arr()]
    out_segs = [fiberwrite_X0_6.get_seg_arr(), fiberwrite_X1_5.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()
    repsiggen_i_15.print_fifos()
    repeat_Ci_14.print_fifos()
    repsiggen_j_11.print_fifos()
    repeat_Bj_10.print_fifos()
    intersectk_7.print_fifos()
    arrayvals_B_3.print_fifos()
    mul_2.print_fifos()
    reduce_1.print_fifos()
    arrayvals_C_4.print_fifos()
    intersectk_7.print_intersection_rate()
