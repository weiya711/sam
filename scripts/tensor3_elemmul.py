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
def test_tensor3_elemmul(ssname, debug_sim, fill = 0):
    C_dirname = os.path.join(formatted_dir, ssname, "orig", "sds012")
    C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
    C_shape = read_inputs(C_shape_filename)

    C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
    C_seg0 = read_inputs(C0_seg_filename)
    C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
    C_crd0 = read_inputs(C0_crd_filename)

    C2_seg_filename = os.path.join(C_dirname, "C2_seg.txt")
    C_seg2 = read_inputs(C2_seg_filename)
    C2_crd_filename = os.path.join(C_dirname, "C2_crd.txt")
    C_crd2 = read_inputs(C2_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
    C_vals = read_inputs(C_vals_filename, float)

    B_dirname = os.path.join(formatted_dir, ssname, "shift", "sss012")
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

    B2_seg_filename = os.path.join(B_dirname, "B2_seg.txt")
    B_seg2 = read_inputs(B2_seg_filename)
    B2_crd_filename = os.path.join(B_dirname, "B2_crd.txt")
    B_crd2 = read_inputs(B2_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    fiberlookup_Ci = CompressedRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim)
    fiberlookup_Bi = CompressedRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
    intersect = Intersect2(debug = debug_sim)
    fiberlookup_Cj = UncompressRdScan( dim = C_shape[1], debug = debug_sim) 
    fiberlookup_Bj = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
    intersect = Intersect2(debug = debug_sim)
    fiberlookup_Ck = CompressedRdScan(crd_arr=C_crd2, seg_arr=C_seg2, debug=debug_sim)
    fiberlookup_Bk = CompressedRdScan(crd_arr=B_crd2, seg_arr=B_seg2, debug=debug_sim)
    intersect = Intersect2(debug = debug_sim)
    arrayvals_C = Array(init_arr= C_vals, debug = debug_sim)
    arrayvals_B = Array(init_arr= B_vals, debug = debug_sim)
    mul = Multiply2(debug=debug_sim)
    fiberwrite_Xvals = ValsWrScan(size= 1 * B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim)
    fiberwrite_X2 = CompressWrScan(seg_size = B_shape[0] * B_shape[1] + 1, size=B_shape[0] * B_shape[1] * B_shape[2], fill = fill, debug = debug_sim)
    fiberwrite_X1 = CompressWrScan(seg_size = B_shape[0] + 1, size=B_shape[0] * B_shape[1], fill = fill, debug = debug_sim)
    in_ref_C = [0, 'D']
    in_ref_B = [0, 'D']
    done = False
    time = 0


    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))
        fiberlookup_Bi.update()

        if len(in_ref_C) > 0:
            fiberlookup_Ci.set_in_ref(in_ref_C.pop(0))
        fiberlookup_Ci.update()

        fiberlookup_Bj.set_in_ref(intersect.out_ref_out-B())
        fiberlookup_Bj.update()

        fiberlookup_Cj.set_in_ref(intersect.out_ref_out-C())
        fiberlookup_Cj.update()

        fiberlookup_Bk.set_in_ref(intersect.out_ref_out-B())
        fiberlookup_Bk.update()

        fiberlookup_Ck.set_in_ref(intersect.out_ref_out-C())
        fiberlookup_Ck.update()

        intersect.set_in1(fiberlookup_Bk.out_ref(), fiberlookup_Bk.out_crd())
        intersect.set_in2(fiberlookup_Ck.out_ref(), fiberlookup_Ck.out_crd())
        intersect.update()

        intersect.set_in1(fiberlookup_Bj.out_ref(), fiberlookup_Bj.out_crd())
        intersect.set_in2(fiberlookup_Cj.out_ref(), fiberlookup_Cj.out_crd())
        intersect.update()

        intersect.set_in1(fiberlookup_Bi.out_ref(), fiberlookup_Bi.out_crd())
        intersect.set_in2(fiberlookup_Ci.out_ref(), fiberlookup_Ci.out_crd())
        intersect.update()

        arrayvals_B.set_load(intersect.out_ref1())
        arrayvals_B.update()

        arrayvals_C.set_load(intersect.out_ref2())
        arrayvals_C.update()

        mul.set_in1(arrayvals_B.out_load())
        mul.set_in2(arrayvals_C.out_load())
        mul.update()

        fiberwrite_Xvals.set_input(mul.out_val())
        fiberwrite_Xvals.update()

        done = fiberwrite_Xvals.out_done() and fiberwrite_X2.out_done() and fiberwrite_X1.out_done()
        time += 1

    fiberwrite_Xvals.autosize()
    fiberwrite_X2.autosize()
    fiberwrite_X1.autosize()

    out_crds = [fiberwrite_X2.get_arr(), fiberwrite_X1.get_arr()]
    out_segs = [fiberwrite_X2.get_seg_arr(), fiberwrite_X1.get_seg_arr()]
    out_vals = fiberwrite_Xvals.get_arr()
