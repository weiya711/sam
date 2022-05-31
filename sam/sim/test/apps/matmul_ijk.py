import scipy.sparse
from sam.sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sim.test.test import *
import os 
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default = './mode-formats')

# FIXME: Figureout formats

@pytest.mark.skipif(
	os.getenv('CI', 'false') == 'true',
	reason = 'CI lacks datasets',
)
def test_matmul_ijk(ssname, debug_sim, fill = 0):
	filenmame = os.path.join(formatted_dir, ssname + "_" + "dcsr.txt")
	formats  = ['d', 's']
	[B_shape, (B_seg0, Bcrd0), (B_seg1, Bcrd1), B_vals] = read_inputs(filename, formats)
	filenmame = os.path.join(formatted_dir, ssname + "_" + "_trans_shifted_csr.txt")
	formats  = ['d', 's']
	[C_shape, C_dim0, (C_seg1, Ccrd1), C_vals] = read_inputs(filename, formats)
	fiberlookup_Bi = CompressedRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim)
	repsiggen_i = RepeatSigGen(debug=debug_sim)
	repeat_Ci = Repeat(debug=debug_sim)
	fiberlookup_Cj = UncompressRdScan( dim = C_dim0, debug = debug_sim) 
	repsiggen_j = RepeatSigGen(debug=debug_sim)
	repeat_Bj = Repeat(debug=debug_sim)
	fiberlookup_Bk = CompressedRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim)
	fiberwrite_X1 = CompressWrScan(seg_size =  B_dim0 + 1 , size= B_dim0 *B_dim0, fill = fill)
	fiberlookup_Ck = CompressedRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim)
	intersect = Intersect2(debug = debug_sim)
	arrayvals_C = Array(init_arr= C_vals, debug = debug_sim)
	arrayvals_B = Array(init_arr= B_vals, debug = debug_sim)
	mul = Multiply2(debug=debug_sim)
	reduce = Reduce(debug=debug_sum)
	fiberwrite_Xvals = ValsWrScan(size= B_dim0 * C_dim1, fill=fill, debug=debug_sim)
	fiberwrite_X0 = CompressWrScan(seg_size =  1 + 1 , size=B_dim0, fill = fill)
	in_ref_B = [0, 'D']
	in_ref_C = [0, 'D']
	done = False
        time = 0
        while not done and time < TIMEOUT:
                if len(in_ref_B) > 0:
			fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))
		fiberlookup_Bi.update()

		fiberwrite_X0.set_input(fiberlookup_Bi.out_crd())
		fiberwrite_X0.update()

		repsiggen_i.set_istream(fiberlookup_Bi.out_crd())
		repsiggen_i.update()

		if len(in_ref_C) > 0:
			repeat_Ci.set_in_ref(in_ref_C.pop(0))
		repeat_Ci.set_in_repsig(repsiggen_i.out_repeat())
		repeat_Ci.update()

		fiberlookup_Cj.set_in_ref(repeat_Ci.out_ref())
		fiberlookup_Cj.update()

		fiberlookup_Ck.set_in_ref(fiberlookup_Cj.out_ref())
		fiberlookup_Ck.update()

		fiberwrite_X1.set_input(fiberlookup_Cj.out_crd())
		fiberwrite_X1.update()

		repsiggen_j.set_istream(fiberlookup_Cj.out_crd())
		repsiggen_j.update()

		repeat_Bj.set_in_ref(fiberlookup_Bi.out_repeat())
		repeat_Bj.set_in_repsig(repsiggen_j.out_repeat())
		repeat_Bj.update()

		fiberlookup_Bk.set_in_ref(repeat_Bj.out_ref())
		fiberlookup_Bk.update()

		intersect.set_in1(fiberlookup_Bk.out_ref(), fiberlookup_Bk.out_crd())
		intersect.set_in2(fiberlookup_Ck.out_ref(), fiberlookup_Ck.out_crd())
		intersect.update()

		arrayvals_B.set_load(intersect.out_ref())
		arrayvals_B.update()

		arrayvals_C.set_load(intersect.out_ref())
		arrayvals_C.update()

		mul.set_in1(arrayvals_B.out_load())
		mul.set_in2(arrayvals_C.out_load())
		mul.update()

		reduce.set_in_val(mul.out_val())
		reduce.update()

		fiberwrite_Xvals.set_input(reduce.out_val())
		fiberwrite_Xvals.update()

		done = fiberwrite_X1.out_done() and fiberwrite_Xvals.out_done() and fiberwrite_X0.out_done()
		time += 1



        
        B_scipy = scipy.sparse.csr_matrix((B_vals, B_crd1, B_seg1), shape=B_shape)
        C_scipy = scipy.sparse.csc_matrix((C_vals, C_crd0, C_seg0), shape=C_shape)
        B_nd = B_scipy.toarray()
        C_nd = C_scipy.toarray()
        gold_nd = B_nd @ C_nd
        gold_tup = convert_ndarr_point_tuple(gold_nd)



	fiberwrite_X1.autosize()
	fiberwrite_Xvals.autosize()
	fiberwrite_X0.autosize()


        out_crds = [fiberwrite_X0.get_arr(), fiberwrite_X1.get_arr()]
        out_segs = [fiberwrite_X1.get_seg_arr(), fiberwrite_X1.get_seg_arr()]
        out_val = fiberwrite_Xvals.get_arr()



        if not out_val:
            assert out_val == gold_tup
        elif not gold_tup:
            assert all([v == 0 for v in out_val])
        else:
            out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
            out_tup = remove_zeros(out_tup)
            assert (check_point_tuple(out_tup, gold_tup))
