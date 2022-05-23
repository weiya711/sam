import scipy.sparse
from sam.sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
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
def test_vec_elemadd(ssname, debug_sim, fill=0):
    b_dirname = os.path.join(formatted_dir, ssname, "orig", "s0")
    b_shape_filename = os.path.join(b_dirname, "b_shape.txt")
    b_shape = read_inputs(b_shape_filename)

    b0_seg_filename = os.path.join(b_dirname, "b0_seg.txt")
    b_seg0 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "b0_crd.txt")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "b_vals.txt")
    b_vals = read_inputs(b_vals_filename, float)

    c_dirname = os.path.join(formatted_dir, ssname, "shift", "d0")
    c_shape_filename = os.path.join(c_dirname, "c_shape.txt")
    c_shape = read_inputs(c_shape_filename)

    c_vals_filename = os.path.join(c_dirname, "c_vals.txt")
    c_vals = read_inputs(c_vals_filename, float)

    fiberlookup_ci_7 = UncompressRdScan(dim=c_shape[0], debug=debug_sim)
    fiberlookup_bi_6 = CompressedRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim)
    arrayvals_c_3 = Array(init_arr=c_vals, debug=debug_sim)
    arrayvals_b_2 = Array(init_arr=b_vals, debug=debug_sim)
    fiberwrite_xvals_0 = ValsWrScan(size=1 * b_shape[0], fill=fill, debug=debug_sim)
    fiberwrite_x0_4 = CompressWrScan(seg_size=2, size=b_shape[0], fill=fill, debug=debug_sim)
