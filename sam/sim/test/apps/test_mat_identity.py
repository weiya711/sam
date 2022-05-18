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
    B_seg1 = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "B1_crd.txt")
    B_crd1 = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    fiberlookup_Bi_5  = CompressedRdScan(crd_arr = B_crd0, seg_arr = B_seg0, debug = debug_sim)
    fiberlookup_Bj_4  = CompressedRdScan(crd_arr = B_crd1, seg_arr = B_seg1, debug = debug_sim)
    arrayvals_B_1 = Array(init_arr= B_vals, debug = debug_sim)
    fiberwrite_Xvals_0 = ValsWrScan(size= 1 * B_shape[0] * B_shape[1], fill=fill, debug=debug_sim)
    fiberwrite_X1_2 = CompressWrScan(seg_size = B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill = fill, debug = debug_sim)
    fiberwrite_X0_3 = CompressWrScan(seg_size = 2, size=B_shape[0], fill = fill, debug = debug_sim)
    in_ref_B = [0, 'D']
    done = False
    time = 0


    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_5.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi_5.update()

        fiberwrite_X0_3.set_input(fiberlookup_Bi_5.out_crd())
        fiberwrite_X0_3.update()

        fiberlookup_Bj_4.set_in_ref(fiberlookup_Bi_5.out_ref())
        fiberlookup_Bj_4.update()

        fiberwrite_X1_2.set_input(fiberlookup_Bj_4.out_crd())
        fiberwrite_X1_2.update()

        arrayvals_B_1.set_load(fiberlookup_Bj_4.out_ref())
        arrayvals_B_1.update()

        fiberwrite_Xvals_0.set_input(arrayvals_B_1.out_val())
        fiberwrite_Xvals_0.update()

    

        done = fiberwrite_Xvals_0.out_done() and fiberwrite_X1_2.out_done() and fiberwrite_X0_3.out_done()
        time += 1

    fiberwrite_Xvals_0.autosize()
    fiberwrite_X1_2.autosize()
    fiberwrite_X0_3.autosize()

    out_crds = [fiberwrite_X1_2.get_arr(), fiberwrite_X0_3.get_arr()]
    out_segs = [fiberwrite_X1_2.get_seg_arr(), fiberwrite_X0_3.get_seg_arr()]
    out_vals = fiberwrite_Xvals_0.get_arr()
    arrayvals_B_1.print_fifos()
