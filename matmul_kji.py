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
def test_matmul_kji(ssname, debug_sim, fill=0):
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

    C_dirname = os.path.join(formatted_dir, ssname, "shift-trans", "ds01")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
    C_seg1 = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
    C_crd1 = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    fiberlookup_Ck_17 = UncompressRdScan(dim=C_shape[0], debug=debug_sim)
    fiberlookup_Bk_16 = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    intersectk_15 = Intersect2(debug=debug_sim)
    fiberlookup_Cj_14 = CompressedRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
    repsiggen_j_12 = RepeatSigGen(debug=debug_sim)
    repeat_Bj_11 = Repeat(debug=debug_sim)
    fiberlookup_Bi_10 = CompressedRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    repsiggen_i_8 = RepeatSigGen(debug=debug_sim)
    repeat_Ci_7 = Repeat(debug=debug_sim)
    arrayvals_C_4 = Array(init_arr=C_vals, debug=debug_sim)
    fiberwrite_X0_5 = CompressWrScan(seg_size=C_shape[1] + 1, size=C_shape[1] * B_shape[0], fill=fill, debug=debug_sim)
    arrayvals_B_3 = Array(init_arr=B_vals, debug=debug_sim)
    mul_2 = Multiply2(debug=debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * C_shape[1] * B_shape[0], fill=fill, debug=debug_sim)
    fiberwrite_X1_6 = CompressWrScan(seg_size=2, size=C_shape[1], fill=fill, debug=debug_sim)
