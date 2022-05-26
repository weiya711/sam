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
from sam.sim.test.test.test_gold import test_gold_vec
import os
cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_vec_scalar_mul_i(ssname, debug_sim, fill=0):
    b_dirname = os.path.join(formatted_dir, ssname, "dummy", "none")
    b_shape_filename = os.path.join(b_dirname, "b_shape.txt")
    b_shape = read_inputs(b_shape_filename)

    b_vals_filename = os.path.join(b_dirname, "b_vals.txt")
    b_vals = read_inputs(b_vals_filename, float)

    c_dirname = os.path.join(formatted_dir, ssname, "dummy", "s0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c0_seg_filename = os.path.join(c_dirname, "c0_seg.txt")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "c0_crd.txt")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_ci_8 = CompressedRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    arrayvals_c_4 = Array(init_arr=c_vals, debug=debug_sim)
    fiberwrite_x0_1 = CompressWrScan(seg_size=2, size=c_shape[0], fill=fill, debug=debug_sim)
    repsiggen_i_6 = RepeatSigGen(debug=debug_sim)
    repeat_bi_5 = Repeat(debug=debug_sim)
    arrayvals_b_3 = Array(init_arr=b_vals, debug=debug_sim)
    mul_2 = Multiply2(debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * c_shape[0], fill=fill, debug=debug_sim)
    in_ref_c = [0, 'D']
    in_ref_b = [0, 'D']
    done = False
    time = 0

    while not done and time < TIMEOUT:
        if len(in_ref_c) > 0:
            fiberlookup_ci_8.set_in_ref(in_ref_c.pop(0))
        fiberlookup_ci_8.update()

        arrayvals_c_4.set_load(fiberlookup_ci_8.out_ref())
        arrayvals_c_4.update()

        fiberwrite_x0_1.set_input(fiberlookup_ci_8.out_crd())
        fiberwrite_x0_1.update()

        repsiggen_i_6.set_istream(fiberlookup_ci_8.out_crd())
        repsiggen_i_6.update()

        if len(in_ref_b) > 0:
            repeat_bi_5.set_in_ref(in_ref_b.pop(0))
        repeat_bi_5.set_in_repsig(repsiggen_i_6.out_repsig())
        repeat_bi_5.update()

        arrayvals_b_3.set_load(repeat_bi_5.out_ref())
        arrayvals_b_3.update()

        mul_2.set_in1(arrayvals_b_3.out_load())
        mul_2.set_in2(arrayvals_c_4.out_load())
        mul_2.update()

        fiberwrite_xvals_0.set_input(mul_2.out_val())
        fiberwrite_xvals_0.update()

        done = fiberwrite_x0_1.out_done() and fiberwrite_xvals_0.out_done()
        time += 1

    fiberwrite_x0_1.autosize()
    fiberwrite_xvals_0.autosize()

    out_crds = [fiberwrite_x0_1.get_arr()]
    out_segs = [fiberwrite_x0_1.get_seg_arr()]
    out_vals = fiberwrite_xvals_0.get_arr()
    repsiggen_i_6.print_fifos()
    repeat_bi_5.print_fifos()
    arrayvals_b_3.print_fifos()
    mul_2.print_fifos()
    arrayvals_c_4.print_fifos()
    test_gold_vec(ssname , formats = [dummy, dummy],  out_crds = [fiberwrite_x0_1.get_arr()], out_segs = [fiberwrite_x0_1.get_seg_arr()], out_vals = fiberwrite_xvals_0.get_arr())
