from sim.src.rd_scanner import UncompressRdScan, COmpressedRdScan
from sim.src.wr_scanner import ValsWrScan
from sim.src.joiner import Intersect2
from sim.src.compute import Multiply2
from sim.src.crd_manager import CrdDrop
from sim.src.base import remove_emptystr
from sim.test.test import *
def test_matmul_ikj():
	fiberlookup_Bi_17 = CompressedRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
	repsiggen_i_15 = RepeatSigGen(debug=debug_sim)
	repeat_Ci_14 = Repeat(debu=debug_sim)
	fiberlookup_Cj_13 = UncompressRdScan( dim = 0, debug = debug_sim) 
	repsiggen_j_11 = RepeatSigGen(debug=debug_sim)
	repeat_Bj_10 = Repeat(debu=debug_sim)
	fiberlookup_Bk_8 = CompressedRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)
	fiberlookup_Ck_9 = CompressedRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)
	intersect_7 = Intersect2(debug = debug_sim)
	arrayvals_C_6 = Array(init_arr=in_mat_vals2, debug = debug_sim)
	arrayvals_B_5 = Array(init_arr=in_mat_vals2, debug = debug_sim)
	mul_4 = Multiply2(debug=debug_sim)
	reduce_3 = Reduce(debug=debug_sum)


	while not done and time < TIMEOUT:
		repeat_Bj_10.set_in_"ref"(fiberlookup_Bi_17.out_repeat())

		repsiggen_i_15.set_istream(fiberlookup_Bi_17.out_"crd")
		repsiggen_i_15.update()

		repeat_Ci_14.set_in_"repsig"(repsiggen_i_15.out_repeat())

		fiberlookup_Cj_13.set_in_ref(repeat_Ci_14.in_ref_C.pop())
		fiberlookup_Cj_13.update()

		fiberlookup_Ck_9.set_in_ref(fiberlookup_Cj_13.in_ref_C.pop())
		fiberlookup_Ck_9.update()

		repsiggen_j_11.set_istream(fiberlookup_Cj_13.out_"crd")
		repsiggen_j_11.update()

		repeat_Bj_10.set_in_"repsig"(repsiggen_j_11.out_repeat())

		fiberlookup_Bk_8.set_in_ref(repeat_Bj_10.in_ref_B.pop())
		fiberlookup_Bk_8.update()

