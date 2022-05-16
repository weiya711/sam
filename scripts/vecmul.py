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
def test_vecmul(ssname, debug_sim, fill = 0):
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

    fiberlookup_Bi = UncompressRdScan( dim = B_shape[0], debug = debug_sim) 
    repsiggen_i = RepeatSigGen(debug=debug_sim)
    repeat_ci = Repeat(debug=debug_sim)
    fiberlookup_cj = CompressedRdScan(crd_arr=c_crd0, seg_arr=c_seg0, debug=debug_sim)
    fiberwrite_x0 = CompressWrScan(seg_size = 2, size=B_shape[0], fill = fill, debug = debug_sim)
    fiberlookup_Bj = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    intersect = Intersect2(debug = debug_sim)
    arrayvals_c = Array(init_arr= c_vals, debug = debug_sim)
    arrayvals_B = Array(init_arr= B_vals, debug = debug_sim)
    mul = Multiply2(debug=debug_sim)
    reduce = Reduce(debug=debug_sim)
    fiberwrite_xvals = ValsWrScan(size= 1 * B_shape[0], fill=fill, debug=debug_sim)
    in_ref_B = [0, 'D']
    in_ref_c = [0, 'D']
    done = False
    time = 0


    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi.update()

        fiberlookup_Bj.set_in_ref(fiberlookup_Bi.out_ref())
        fiberlookup_Bj.update()

        fiberwrite_x0.set_input(fiberlookup_Bi.out_crd())
        fiberwrite_x0.update()

        repsiggen_i.set_istream(fiberlookup_Bi.out_crd())
        repsiggen_i.update()

        if len(in_ref_c) > 0:
            repeat_ci.set_in_ref(in_ref_c.pop(0))
        repeat_ci.set_in_repsig(repsiggen_i.out_repsig())
        repeat_ci.update()

        fiberlookup_cj.set_in_ref(repeat_ci.out_ref())
        fiberlookup_cj.update()

        intersect.set_in1(fiberlookup_cj.out_ref(), fiberlookup_cj.out_crd())
        intersect.set_in2(fiberlookup_Bj.out_ref(), fiberlookup_Bj.out_crd())
        intersect.update()

        arrayvals_B.set_load(intersect.out_ref2())
        arrayvals_B.update()

        arrayvals_c.set_load(intersect.out_ref1())
        arrayvals_c.update()

        mul.set_in1(arrayvals_B.out_load())
        mul.set_in2(arrayvals_c.out_load())
        mul.update()

        reduce.set_in_val(mul.out_val())
        reduce.update()

        fiberwrite_xvals.set_input(reduce.out_val())
        fiberwrite_xvals.update()

        done = fiberwrite_x0.out_done() and fiberwrite_xvals.out_done()
        time += 1

    fiberwrite_x0.autosize()
    fiberwrite_xvals.autosize()

    out_crds = [fiberwrite_x0.get_arr()]
    out_segs = [fiberwrite_x0.get_seg_arr()]
    out_vals = fiberwrite_xvals.get_arr()
