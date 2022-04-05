from sim.src.rd_scanner import UncompressRdScan, CompressedRdScan
from sim.src.wr_scanner import ValsWrScan
from sim.src.joiner import Intersect2
from sim.src.compute import Multiply2
from sim.src.crd_manager import CrdDrop
from sim.src.base import remove_emptystr

from sim.test.test import *


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
    gold_nd = nd1 @ nd2
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Pts Mat1:", in1_tup)
        print("Pts Mat2:", in2_tup)
        print("Dense Mat1:", nd1)
        print("Dense Mat2:", nd2)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    rdscan_B1 = CompressedRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    rdscan_C1 = CompressedRdScan(crd_arr=in_mat_crds2[0], seg_arr=in_mat_segs2[0], debug=debug_sim)
    rdscan_C2 = CompressedRdScan(crd_arr=in_mat_crds2[1], seg_arr=in_mat_segs2[1], debug=debug_sim)

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)
    val_C = Array(init_arr=in_mat_vals2, debug=debug_sim)
    inter1 = Intersect2(debug=debug_sim)
    inter2 = Intersect2(debug=debug_sim)
    mul = Multiply2(debug=debug_sim)
    drop = CrdDrop(debug=debug_sim)
    vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=dim, fill=fill)
    wrscan_X2 = CompressWrScan(seg_size=dim + 1, size=dim * dim, fill=fill)

    in_ref_B = [0, 'D']
    in_ref_C = [0, 'D']
    done = False
    time = 0
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))
        rdscan_B1.update()

        if len(in_ref_C) > 0:
            rdscan_C1.set_in_ref(in_ref_C.pop(0))
        rdscan_C1.update()

        inter1.set_in1(rdscan_B1.out_ref(), rdscan_B1.out_crd())
        inter1.set_in2(rdscan_C1.out_ref(), rdscan_C1.out_crd())
        inter1.update()

        rdscan_B2.set_in_ref(inter1.out_ref1())
        rdscan_B2.update()

        rdscan_C2.set_in_ref(inter1.out_ref2())
        rdscan_C2.update()

        inter2.set_in1(rdscan_B2.out_ref(), rdscan_B2.out_crd())
        inter2.set_in2(rdscan_C2.out_ref(), rdscan_C2.out_crd())
        inter2.update()

        val_B.set_load(inter2.out_ref1())
        val_B.update()
        val_C.set_load(inter2.out_ref2())
        val_C.update()

        mul.set_in1(val_B.out_load())
        mul.set_in2(val_C.out_load())
        mul.update()

        drop.set_outer_crd(inter1.out_crd())
        drop.set_inner_crd(inter2.out_crd())
        drop.update()

        vals_X.set_input(mul.out_val())
        vals_X.update()

        wrscan_X1.set_input(drop.out_crd_outer())
        wrscan_X1.update()

        wrscan_X2.set_input(inter2.out_crd())
        wrscan_X2.update()
        print("Timestep", time, "\t Done --",
              "\tRdScan B1:", rdscan_B1.out_done(), "\tRdScan B2:", rdscan_B2.out_done(),
              "\tInter1:", inter1.out_done(),
              "\tRdScan C1:", rdscan_C1.out_done(), "\tRdScan C2:", rdscan_C2.out_done(),
              "\tInter2:", inter2.out_done(),
              "\tArr:", val_B.out_done(), val_C.out_done(),
              "\tMul:", mul.out_done(),
              "\tDrop:", drop.out_done(),
              "\tWrScan:", vals_X.out_done(),
              "\tWrScan X1:", wrscan_X1.out_done(), "\tWrScan X2:", wrscan_X2.out_done(),
              )
        done = wrscan_X2.out_done()
        time += 1

    wrscan_X2.autosize()
    wrscan_X1.autosize()
    vals_X.autosize()

    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print(out_segs)
        print(out_crds)
        print(out_val)

    if out_val == []:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))