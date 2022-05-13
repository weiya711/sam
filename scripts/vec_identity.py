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
def test_vec_identity(ssname, debug_sim, fill = 0):
    b_dirname = os.path.join(formatted_dir, ssname, "orig", "s0")
    b_shape_filename = os.path.join(b_dirname, "b_shape.txt")
    b_shape = read_inputs(b_shape_filename)

    b0_seg_filename = os.path.join(b_dirname, "b0_seg.txt")
    b_seg0 = read_inputs(b0_seg_filename)
    b0_crd_filename = os.path.join(b_dirname, "b0_crd.txt")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "b_vals.txt")
    b_vals = read_inputs(b_vals_filename, float)

    fiberlookup_bi = CompressedRdScan(crd_arr=b_crd0, seg_arr=b_seg0, debug=debug_sim)
    arrayvals_b = Array(init_arr= b_vals, debug = debug_sim)
    fiberwrite_xvals = ValsWrScan(size= b_shape[0], fill=fill, debug=debug_sim)
    fiberwrite_x0 = CompressWrScan(seg_size = 2, size=b_shape[0], fill = fill, debug = debug_sim)
    in_ref_b = [0, 'D']
    done = False
    time = 0


    while not done and time < TIMEOUT:
        if len(in_ref_b) > 0:
            fiberlookup_bi.set_in_ref(in_ref_b.pop(0))
        fiberlookup_bi.update()

        fiberwrite_x0.set_input(fiberlookup_bi.out_crd())
        fiberwrite_x0.update()

        arrayvals_b.set_load(fiberlookup_bi.out_ref())
        arrayvals_b.update()

        fiberwrite_xvals.set_input(arrayvals_b.out_val())
        fiberwrite_xvals.update()

        done = fiberwrite_xvals.out_done() and fiberwrite_x0.out_done()
        time += 1

    fiberwrite_xvals.autosize()
    fiberwrite_x0.autosize()

    out_crds = [fiberwrite_x0.get_arr()]
    out_segs = [fiberwrite_x0.get_seg_arr()]
    out_vals = fiberwrite_xvals.get_arr()
