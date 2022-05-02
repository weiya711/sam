from sim.src.rd_scanner import UncompressRdScan, COmpressedRdScan
from sim.src.wr_scanner import ValsWrScan
from sim.src.joiner import Intersect2
from sim.src.compute import Multiply2
from sim.src.crd_manager import CrdDrop
from sim.src.base import remove_emptystr
from sim.test.test import *
def test_matmul_ikj():
	fiberlookup_Bi = CompressedRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
	repsiggen_i = RepeatSigGen(debug=debug_sim)
	repeat_Ci = Repeat(debu=debug_sim)
	fiberlookup_Cj = UncompressRdScan( dim = 0, debug = debug_sim) 
	repsiggen_j = RepeatSigGen(debug=debug_sim)
	repeat_Bj = Repeat(debu=debug_sim)
	fiberlookup_Bk = CompressedRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)
	fiberwrite_X1 = CompressWrScan(seg_size = dim *dim, size=dim, fill = fill)
	fiberlookup_Ck = CompressedRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)
	intersect = Intersect2(debug = debug_sim)
	arrayvals_C = Array(init_arr=in_mat_vals2, debug = debug_sim)
	arrayvals_B = Array(init_arr=in_mat_vals2, debug = debug_sim)
	mul = Multiply2(debug=debug_sim)
	reduce = Reduce(debug=debug_sum)
	fiberwrite_Xvals = ValsWrScan(size=dim*dim, fill=fill, debug=debug_sim)
	fiberwrite_X0 = CompressWrScan(seg_size = dim, size=dim, fill = fill)


	while not done and time < TIMEOUT:
		if len(in_ref_B) > 0:
			fiberlookup_Bi.set_in_ref(in_ref_B.pop(0))
		intersect.update()

		fiberwrite_X0.set_input(fiberlookup_Biout_crd())
		fiberwrite_X0.update()

		repsiggen_i.set_istream(fiberlookup_Bi.out_crd)
		repsiggen_i.update()

		repeat_Ci.set_in_repsig(repsiggen_i.out_repeat())

		repeat_Ci.update()

		fiberlookup_Cj.set_in_ref(repeat_Ci.out_ref())
		fiberlookup_Cj.update()

		fiberlookup_Ck.set_in_ref(fiberlookup_Cj.out_ref())
		fiberlookup_Ck.update()

		fiberwrite_X1.set_input(fiberlookup_Cjout_crd())
		fiberwrite_X1.update()

		repsiggen_j.set_istream(fiberlookup_Cj.out_crd)
		repsiggen_j.update()

		repeat_Bj.set_in_ref(fiberlookup_Bi.out_repeat())

		repeat_Bj.set_in_repsig(repsiggen_j.out_repeat())

		repeat_Bj.update()

		fiberlookup_Bk.set_in_ref(repeat_Bj.out_ref())
		fiberlookup_Bk.update()

		intersect.set_in0(fiberlookup_Bk.out_ref(), fiberlookup_Bk.out_crd()))
		intersect.set_in1(fiberlookup_Ck.out_ref(), fiberlookup_Ck.out_crd()))
		intersect.update()

		arrayvals_B.set_load(intersect.out_ref())
		arrayvals_B.update()

		arrayvals_C.set_load(intersect.out_ref())
		arrayvals_C.update()

		mul.set_in1(arrayvals_B.out_load())
		mul.set_in1(arrayvals_C.out_load())
		mul.update()

		reduce.set_in_val(mul.out_val())
		reduce.update()

write  ------ 
write  ------ 
