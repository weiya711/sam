from sam.sim.src.rd_scanner import CompressedRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2
from sam.sim.src.compute import Multiply2
from sam.sim.src.crd_manager import CrdDrop
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce

from sam.sim.test.test import *


@pytest.mark.parametrize("dim", [4, 16, 32, 64])
def test_mat_mul_ijk_cc_cc_cc(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)
    in_mat_crds2, in_mat_segs2 = gen_n_comp_arrs(2, dim)
    in_mat_vals2 = gen_val_arr(len(in_mat_crds2[-1]), max_val, -max_val)

    if debug_sim:
        print("Mat 1:", in_mat_segs1, in_mat_crds1, in_mat_vals1)
        print("Mat 2:", in_mat_segs2, in_mat_crds2, in_mat_vals2)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    in2_tup = convert_point_tuple(get_point_list(in_mat_crds2, in_mat_segs2, in_mat_vals2))

    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    nd2 = convert_point_tuple_ndarr(in2_tup, dim)
    gold_nd = nd1 @ nd2.T
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Pts Mat1:", in1_tup)
        print("Pts Mat2:", in2_tup)
        print("Dense Mat1:", nd1)
        print("Dense Mat2:", nd2)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    rdscan_Bi = CompressedRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_Bk = CompressedRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)
    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)

    rdscan_Cj = CompressedRdScan(crd_arr=in_mat_crds2[0], seg_arr=in_mat_segs2[0], debug=debug_sim)
    rdscan_Ck = CompressedRdScan(crd_arr=in_mat_crds2[1], seg_arr=in_mat_segs2[1], debug=debug_sim)
    val_C = Array(init_arr=in_mat_vals2, debug=debug_sim)

    repsiggen_Bi = RepeatSigGen(debug=debug_sim)
    repsiggen_Cj = RepeatSigGen(debug=debug_sim)
    repeat_Ci = Repeat(debug=debug_sim)
    repeat_Bj = Repeat(debug=debug_sim)
    inter1 = Intersect2(debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    reduce = Reduce(debug=debug_sim)

    #drop = CrdDrop(debug=debug_sim)
    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_Xi = CompressWrScan(seg_size=2, size=dim, fill=fill)
    wrscan_Xj = CompressWrScan(seg_size=dim + 1, size=dim * dim, fill=fill)

    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        # Input iteration for i
        if len(in_ref_B) > 0:
            rdscan_Bi.set_in_ref(in_ref_B.pop(0))
        rdscan_Bi.update()

        repsiggen_Bi.set_istream(rdscan_Bi.out_crd())
        repsiggen_Bi.update()

        repeat_Ci.set_in_repeat(repsiggen_Bi.out_repeat())
        if len(in_ref_C) > 0:
            repeat_Ci.set_in_ref(in_ref_C.pop(0))
        repeat_Ci.update()

        # Input iteration for j
        rdscan_Cj.set_in_ref(repeat_Ci.out_ref())
        rdscan_Cj.update()

        repsiggen_Cj.set_istream(rdscan_Cj.out_crd())
        repsiggen_Cj.update()

        repeat_Bj.set_in_repeat(repsiggen_Cj.out_repeat())
        repeat_Bj.set_in_ref(rdscan_Bi.out_ref())
        repeat_Bj.update()

        # Input iteration for k
        rdscan_Bk.set_in_ref(repeat_Bj.out_ref())
        rdscan_Bk.update()

        rdscan_Ck.set_in_ref(rdscan_Cj.out_ref())
        rdscan_Ck.update()

        inter1.set_in1(rdscan_Bk.out_ref(), rdscan_Bk.out_crd())
        inter1.set_in2(rdscan_Ck.out_ref(), rdscan_Ck.out_crd())
        inter1.update()

        # Computation

        val_B.set_load(inter1.out_ref1())
        val_B.update()
        val_C.set_load(inter1.out_ref2())
        val_C.update()

        mul.set_in1(val_B.out_load())
        mul.set_in2(val_C.out_load())
        mul.update()

        reduce.set_in_val(mul.out_val())
        reduce.update()

        vals_X.set_input(reduce.out_val())
        vals_X.update()

        wrscan_Xi.set_input(rdscan_Bi.out_crd())
        wrscan_Xi.update()

        wrscan_Xj.set_input(rdscan_Cj.out_crd())
        wrscan_Xj.update()

        if time % 100 == 0:
            print("Timestep", time, "\t Done --",
                  "\nRdScan Bi:", rdscan_Bi.out_done(), "\tRepeat Ci:", repeat_Ci.out_done(),
                  "\tRepSigGen Bi:", repsiggen_Bi.out_done(),
                  "\nRepeat Bj:", repeat_Bj.out_done(), "\tRdScan Cj:", rdscan_Cj.out_done(),
                  "\tRepSigGen Cj:", repsiggen_Cj.out_done(),
                  "\nRdScan Bk:", rdscan_Bk.out_done(), "\tRdScan Ck:", rdscan_Ck.out_done(),
                  "\tInter k:", inter1.out_done(),
                  "\nArr:", val_B.out_done(), val_C.out_done(),
                  "\tMul:", mul.out_done(),
                  "\tRed:", reduce.out_done(),
                  "\nVals X:", vals_X.out_done(),
                  "\tWrScan X1:", wrscan_Xi.out_done(), "\tWrScan X2:", wrscan_Xj.out_done(),
                  )

        done = wrscan_Xj.out_done() and wrscan_Xi.out_done() and vals_X.out_done()
        time += 1

    wrscan_Xi.autosize()
    wrscan_Xj.autosize()
    vals_X.autosize()

    out_crds = [wrscan_Xi.get_arr(), wrscan_Xj.get_arr()]
    out_segs = [wrscan_Xi.get_seg_arr(), wrscan_Xj.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print(out_segs)
        print(out_crds)
        print(out_val)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))
