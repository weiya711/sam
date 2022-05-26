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
import os
cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
def test_vecmul_ji_i(ssname, debug_sim, fill=0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss10")
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

    c_dirname = os.path.join(formatted_dir, ssname, "shift", "s0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "c0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "c0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_Bj_11 = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    fiberlookup_cj_12 = CompressedRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    intersectj_10 = Intersect2(debug=debug_sim)
    fiberlookup_Bi_9 = CompressedRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim)
    repsiggen_i_7 = RepeatSigGen(debug=debug_sim)
    repeat_ci_6 = Repeat(debug=debug_sim)
    arrayvals_c_5 = Array(init_arr=c_vals, debug=debug_sim)
    mul_3 = Multiply2(debug=debug_sim)
    spaccumulator1_2 = SparseAccumulator1(debug=debug_sim)
    spaccumulator1_2_drop_crd_in_inner = StknDrop(debug=debug_sim)
    spaccumulator1_2_drop_crd_in_outer = StknDrop(debug=debug_sim)
    spaccumulator1_2_drop_val = StknDrop(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * B_shape[0], fill=fill, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bj_11.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bj_11.update()

        if len(in_ref_c) > 0:
            fiberlookup_cj_12.set_in_ref(in_ref_c.pop(0))
        fiberlookup_cj_12.update()

        intersectj_10.set_in1(fiberlookup_Bj_11.out_ref(), fiberlookup_Bj_11.out_crd())
        intersectj_10.set_in2(fiberlookup_cj_12.out_ref(), fiberlookup_cj_12.out_crd())
        intersectj_10.update()

        fiberlookup_Bi_9.set_in_ref(intersectj_10.out_ref1())
        fiberlookup_Bi_9.update()

        arrayvals_B_4.set_load(fiberlookup_Bi_9.out_ref())
        arrayvals_B_4.update()

        repsiggen_i_7.set_istream(fiberlookup_Bi_9.out_crd())
        repsiggen_i_7.update()

        repeat_ci_6.set_in_ref(intersectj_10.out_ref2())
        repeat_ci_6.set_in_repsig(repsiggen_i_7.out_repsig())
        repeat_ci_6.update()

        arrayvals_c_5.set_load(repeat_ci_6.out_ref())
        arrayvals_c_5.update()

        mul_3.set_in1(arrayvals_c_5.out_load())
        mul_3.set_in2(arrayvals_B_4.out_load())
        mul_3.update()

        spaccumulator1_2_drop_crd_in_outer.set_in_stream(intersectj_10.out_crd())
        spaccumulator1_2_drop_crd_in_inner.set_in_stream(fiberlookup_Bi_9.out_crd())
        spaccumulator1_2_drop_val.set_in_stream(mul_3.out_val())
        spaccumulator1_2.crd_in_outer(spaccumulator1_2_drop_crd_in_outer.out_val())
        spaccumulator1_2.crd_in_inner(spaccumulator1_2_drop_crd_in_inner.out_val())
        spaccumulator1_2.set_val(spaccumulator1_2_drop_val.out_val())
        spaccumulator1_2.update()

        fiberwrite_xvals_0.set_input(spaccumulator1_2.out_val())
        fiberwrite_xvals_0.update()

        fiberwrite_x0_1.set_input(spaccumulator1_2.out_crd_inner())
        fiberwrite_x0_1.update()

        done = fiberwrite_xvals_0.out_done() and fiberwrite_x0_1.out_done()
        time += 1

    fiberwrite_xvals_0.autosize()
    fiberwrite_x0_1.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()
    intersectj_10.print_fifos()
    spaccumulator1_2.print_fifos()
    repsiggen_i_7.print_fifos()
    repeat_ci_6.print_fifos()
    arrayvals_c_5.print_fifos()
    mul_3.print_fifos()
    arrayvals_B_4.print_fifos()
    intersectj_10.print_intersection_rate()
