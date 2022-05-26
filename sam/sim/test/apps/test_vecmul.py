import pytest
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.test.test_gold import test_gold_vecmul
import os
cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_vecmul_i(ssname, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    c_dirname = os.path.join(formatted_dir, ssname, "shift", "s0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "c0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "c0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_Bi_12 = UncompressRdScan(dim=B_shape[0], debug=debug_sim)
    fiberlookup_Bj_7 = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberwrite_x0_5 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_10 = RepeatSigGen(debug=debug_sim)
    repeat_ci_9 = Repeat(debug=debug_sim)
    fiberlookup_cj_8 = CompressedRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    intersectj_6 = Intersect2(debug=debug_sim)
    arrayvals_B_3 = Array(init_arr=B_vals, debug=debug_sim)
    arrayvals_c_4 = Array(init_arr=c_vals, debug=debug_sim)
    mul_2 = Multiply2(debug=debug_sim)
    reduce_1 = Reduce(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * B_shape[0], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_12.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_12.update()

        fiberlookup_Bj_7.set_in_ref(fiberlookup_Bi_12.out_ref())
        fiberlookup_Bj_7.update()

        fiberwrite_x0_5.set_input(fiberlookup_Bi_12.out_crd())
        fiberwrite_x0_5.update()

        repsiggen_i_10.set_istream(fiberlookup_Bi_12.out_crd())
        repsiggen_i_10.update()

        if len(in_ref_c) > 0:
            repeat_ci_9.set_in_ref(in_ref_c.pop(0))
        repeat_ci_9.set_in_repsig(repsiggen_i_10.out_repsig())
        repeat_ci_9.update()

        fiberlookup_cj_8.set_in_ref(repeat_ci_9.out_ref())
        fiberlookup_cj_8.update()

        intersectj_6.set_in1(fiberlookup_cj_8.out_ref(), fiberlookup_cj_8.out_crd())
        intersectj_6.set_in2(fiberlookup_Bj_7.out_ref(), fiberlookup_Bj_7.out_crd())
        intersectj_6.update()

        arrayvals_B_3.set_load(intersectj_6.out_ref2())
        arrayvals_B_3.update()

        arrayvals_c_4.set_load(intersectj_6.out_ref1())
        arrayvals_c_4.update()

        mul_2.set_in1(arrayvals_B_3.out_load())
        mul_2.set_in2(arrayvals_c_4.out_load())
        mul_2.update()

        reduce_1.set_in_val(mul_2.out_val())
        reduce_1.update()

        fiberwrite_xvals_0.set_input(reduce_1.out_val())
        fiberwrite_xvals_0.update()

        done = fiberwrite_x0_5.out_done() and fiberwrite_xvals_0.out_done()
        time += 1

    fiberwrite_x0_5.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_5.get_arr()]
    out_segs = [fiberwrite_x0_5.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()
    repsiggen_i_10.print_fifos()
    repeat_ci_9.print_fifos()
    intersectj_6.print_fifos()
    arrayvals_B_3.print_fifos()
    mul_2.print_fifos()
    reduce_1.print_fifos()
    arrayvals_c_4.print_fifos()
    intersectj_6.print_intersection_rate()
    test_gold_vecmul(ssname , formats = [orig, shift],  out_crds = [fiberwrite_x0_5.get_arr()], out_segs = [fiberwrite_x0_5.get_seg_arr()], out_vals = fiberwrite_xvals_0.get_arr())
