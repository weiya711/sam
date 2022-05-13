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
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd,'mode-formats'))

# FIXME: Figureout formats

@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason = 'CI lacks datasets',
)
def test_mat_identity(ssname, debug_sim, fill = 0):
    B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B0_seg_filename = os.path.join(B_dirname, "B0_seg.txt")
    B_seg0 = read_inputs(B0_seg_filename)
    B0_crd_filename = os.path.join(B_dirname, "B0_crd.txt")
    B_crd0 = read_inputs(B0_crd_filename)

    B1_seg_filename = os.path.join(B_dirname, "B1_seg.txt")
    B_seg1 = read_inputs(B0_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B0_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    fiberlookup_Bi = CompressedRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    fiberlookup_Bj = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    arrayvals_B = Array(init_arr= B_vals, debug = debug_sim)
    fiberwrite_Xvals = ValsWrScan(size= B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    fiberwrite_X1 = CompressWrScan(seg_size = B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill = fill, debug = debug_sim)
    fiberwrite_X0 = CompressWrScan(seg_size = 2, size=B_shape[0], fill = fill, debug = debug_sim)
    in_ref_B = [0, 'D']
    done = False
    time = 0


    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi.update()

        fiberwrite_X0.set_input(fiberlookup_Bi.out_crd())
        fiberwrite_X0.update()

        fiberlookup_Bj.set_in_ref(fiberlookup_Bi.out_ref())
        fiberlookup_Bj.update()

        fiberwrite_X1.set_input(fiberlookup_Bj.out_crd())
        fiberwrite_X1.update()

        arrayvals_B.set_load(fiberlookup_Bj.out_ref())
        arrayvals_B.update()

        fiberwrite_Xvals.set_input(arrayvals_B.out_val())
        fiberwrite_Xvals.update()

        done = fiberwrite_Xvals.out_done() and fiberwrite_X1.out_done() and fiberwrite_X0.out_done()
        time += 1

    fiberwrite_Xvals.autosize()
    fiberwrite_X1.autosize()
    fiberwrite_X0.autosize()

    out_crds = [fiberwrite_X1.get_arr(), fiberwrite_X0.get_arr()]
    out_segs = [fiberwrite_X1.get_seg_arr(), fiberwrite_X0.get_seg_arr()]
    out_vals = fiberwrite_Xvals.get_arr()
