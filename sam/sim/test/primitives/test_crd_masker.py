import pytest
from sam.sim.src.base import remove_emptystr
from sam.sim.src.compute import Add2, Multiply2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
from sam.sim.src.crd_masker import RandomDropout, LowerTriangular, UpperTriangular, Diagonal
from sam.sim.src.crd_manager import CrdHold, CrdDrop, CrdPtConverter
from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.src.wr_scanner import CompressWrScan, ValsWrScan
from sam.sim.test.primitives.test_intersect import TIMEOUT
from sam.sim.test.test import *
import numpy as np
import torch

@pytest.mark.parametrize("dim", [8, 16, 32, 64])
@pytest.mark.parametrize("drop_prob", [0.25, 0.5, 0.75, 0.9])
def test_dropout_2d(dim, drop_prob, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    B_shape = nd1.shape
    np.random.seed(2)
    prob = np.random.rand(*nd1.shape)
    d = prob < (1 - drop_prob)
    gold_nd = np.multiply(nd1, d)
    gold_tup = convert_ndarr_point_tuple(gold_nd)
    # TODO: Add scalar multiply with unary alu
    # in_ = in_ / keep_prob

    flat_prob = prob.flatten()

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    crd = CrdHold(debug=debug_sim)
    crd_conv = CrdPtConverter(debug=debug_sim)
    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)

    # vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=2 * dim, fill=fill, debug=debug_sim)
    wrscan_X2 = CompressWrScan(seg_size=2 *dim, size=2 * dim, fill=fill, debug=debug_sim)
    vals_X = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim)

    dropout = RandomDropout(dimension=2, drop_probability=drop_prob, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    in_ref_B = [0, 'D']

    inner_ref_in = []
    inner_ref_out = []
    inner_crd = []
    # last_i = ""
    # count = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())

        # val_B.set_load(rdscan_B2.out_ref())

        # Might be unnecessary 
        # TODO: Maybe add crd drops in filter block
        # Filter block was already doing a crd hold without this block but not correct
        crd.set_outer_crd(rdscan_B1.out_crd())
        crd.set_inner_crd(rdscan_B2.out_crd())

        dropout.set_prob(prob, drop_prob)

        # i = rdscan_B1.out_crd()
        # if i != "" and isinstance(i, int):
        #     last_i = int(i)
        # j = rdscan_B2.out_crd()

        # if last_i != "" and j != "" and isinstance(j, int):
            # last_j = int(j)
            # dropout.set_predicate(prob[loc[0][count], loc[1][count]], drop_prob)
            # print("Dropping: ", last_i, last_j)
            # dropout.set_predicate(prob[last_j, last_i], drop_prob)
            # last_i = ""
            # count += 1

        dropout.set_inner_crd(crd.out_crd_inner())
        dropout.set_outer_crd(crd.out_crd_outer())
        dropout.set_inner_ref(rdscan_B2.out_ref())

        if debug_sim:
            inner_ref_in.append(rdscan_B2.out_ref())
            print("ref inner in:", remove_emptystr(inner_ref_in))

            inner_ref_out.append(dropout.out_ref())
            print("ref inner out:", remove_emptystr(inner_ref_out))

            inner_crd.append(dropout.out_crd(0))
            print("ref crd:", remove_emptystr(inner_crd))

        val_B.set_load(dropout.out_ref())

        vals_X.set_input(val_B.out_val())

        wrscan_X1.set_input(dropout.out_crd(1))
        wrscan_X2.set_input(dropout.out_crd(0))

        # print("Timestep", time, "\t Out:", dropout.out_crd(0))

        rdscan_B1.update()
        rdscan_B2.update()
        crd.update()
        dropout.update()
        val_B.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        done = wrscan_X1.out_done() and wrscan_X2.out_done() and vals_X.out_done()
        time += 1

    wrscan_X1.autosize()
    wrscan_X2.autosize()
    vals_X.autosize()

    # print("out_arr: ", wrscan_X1.get_arr())
    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print("out_segs", out_segs)
        print("out_crds", out_crds)
        print("out_val", out_val)

    if out_val == []:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))


@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128])
def test_tril_2d(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    B_shape = nd1.shape
    gold_nd = np.tril(nd1)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    crd = CrdHold(debug=debug_sim)
    crd_conv = CrdPtConverter(debug=debug_sim)
    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)

    # vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=2 * dim, fill=fill, debug=debug_sim)
    wrscan_X2 = CompressWrScan(seg_size=2 *dim, size=2 * dim, fill=fill, debug=debug_sim)
    vals_X = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim)

    dropout = LowerTriangular(dimension=2, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    in_ref_B = [0, 'D']

    drop_in = []
    drop_in1 = []

    crd_hold1 = []
    crd_hold2 = []

    inner_ref_in = []
    inner_ref_out = []
    inner_crd = []
    # last_i = ""
    # count = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())

        # val_B.set_load(rdscan_B2.out_ref())

        # Might be unnecessary 
        # TODO: Maybe add crd drops in filter block
        # Filter block was already doing a crd hold without this block but not correct
        crd.set_outer_crd(rdscan_B1.out_crd())
        crd.set_inner_crd(rdscan_B2.out_crd())

        dropout.set_inner_crd(crd.out_crd_inner())
        dropout.set_outer_crd(crd.out_crd_outer())
        dropout.set_inner_ref(rdscan_B2.out_ref())

        if debug_sim:
            inner_ref_in.append(rdscan_B2.out_ref())
            print("ref inner in:", remove_emptystr(inner_ref_in))

            inner_ref_out.append(dropout.out_ref())
            print("ref inner out:", remove_emptystr(inner_ref_out))

            inner_crd.append(dropout.out_crd(0))
            print("ref crd:", remove_emptystr(inner_crd))

        val_B.set_load(dropout.out_ref())

        vals_X.set_input(val_B.out_val())

        wrscan_X1.set_input(dropout.out_crd(1))
        wrscan_X2.set_input(dropout.out_crd(0))

        # print("Timestep", time, "\t Out:", dropout.out_crd(0))

        rdscan_B1.update()
        rdscan_B2.update()
        crd.update()
        dropout.update()
        val_B.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        done = wrscan_X1.out_done() and wrscan_X2.out_done() and vals_X.out_done()
        time += 1

    wrscan_X1.autosize()
    wrscan_X2.autosize()
    vals_X.autosize()

    # print("out_arr: ", wrscan_X1.get_arr())
    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print("out_segs", out_segs)
        print("out_crds", out_crds)
        print("out_val", out_val)

    if out_val == []:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))

@pytest.mark.parametrize("dim", [8, 16, 32, 64, 128])
def test_triu_2d(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    B_shape = nd1.shape
    gold_nd = np.triu(nd1)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    crd = CrdHold(debug=debug_sim)
    crd_conv = CrdPtConverter(debug=debug_sim)
    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)

    # vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=2 * dim, fill=fill, debug=debug_sim)
    wrscan_X2 = CompressWrScan(seg_size=2 *dim, size=2 * dim, fill=fill, debug=debug_sim)
    vals_X = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim)

    dropout = UpperTriangular(dimension=2, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    in_ref_B = [0, 'D']

    drop_in = []
    drop_in1 = []

    crd_hold1 = []
    crd_hold2 = []

    inner_ref_in = []
    inner_ref_out = []
    inner_crd = []
    # last_i = ""
    # count = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())

        # val_B.set_load(rdscan_B2.out_ref())

        # Might be unnecessary 
        # TODO: Maybe add crd drops in filter block
        # Filter block was already doing a crd hold without this block but not correct
        crd.set_outer_crd(rdscan_B1.out_crd())
        crd.set_inner_crd(rdscan_B2.out_crd())

        dropout.set_inner_crd(crd.out_crd_inner())
        dropout.set_outer_crd(crd.out_crd_outer())
        dropout.set_inner_ref(rdscan_B2.out_ref())

        if debug_sim:
            inner_ref_in.append(rdscan_B2.out_ref())
            print("ref inner in:", remove_emptystr(inner_ref_in))

            inner_ref_out.append(dropout.out_ref())
            print("ref inner out:", remove_emptystr(inner_ref_out))

            inner_crd.append(dropout.out_crd(0))
            print("ref crd:", remove_emptystr(inner_crd))

        val_B.set_load(dropout.out_ref())

        vals_X.set_input(val_B.out_val())

        wrscan_X1.set_input(dropout.out_crd(1))
        wrscan_X2.set_input(dropout.out_crd(0))

        # print("Timestep", time, "\t Out:", dropout.out_crd(0))

        rdscan_B1.update()
        rdscan_B2.update()
        crd.update()
        dropout.update()
        val_B.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        done = wrscan_X1.out_done() and wrscan_X2.out_done() and vals_X.out_done()
        time += 1

    wrscan_X1.autosize()
    wrscan_X2.autosize()
    vals_X.autosize()

    # print("out_arr: ", wrscan_X1.get_arr())
    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print("out_segs", out_segs)
        print("out_crds", out_crds)
        print("out_val", out_val)

    if out_val == []:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))


@pytest.mark.parametrize("dim", [8])
def test_diagonal_2d(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    nd1 = convert_point_tuple_ndarr(in1_tup, dim)
    # gold_nd = np.diagonal(nd1)
    gold_nd = nd1 * np.eye(*nd1.shape)
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    crd = CrdHold(debug=debug_sim)
    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)

    # vals_X = ValsWrScan(size=dim * dim, fill=fill, debug=debug_sim)
    wrscan_X1 = CompressWrScan(seg_size=2, size=2 * dim, fill=fill, debug=debug_sim)
    wrscan_X2 = CompressWrScan(seg_size=2 *dim, size=2 * dim, fill=fill, debug=debug_sim)
    vals_X = ValsWrScan(size=5804660 * 2, fill=fill, debug=debug_sim)

    dropout = Diagonal(dimension=2, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    in_ref_B = [0, 'D']

    drop_in = []
    drop_in1 = []

    crd_hold1 = []
    crd_hold2 = []

    inner_ref_in = []
    inner_ref_out = []
    inner_crd = []
    # last_i = ""
    # count = 0

    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())

        # val_B.set_load(rdscan_B2.out_ref())

        # Might be unnecessary 
        # TODO: Maybe add crd drops in filter block
        # Filter block was already doing a crd hold without this block but not correct
        crd.set_outer_crd(rdscan_B1.out_crd())
        crd.set_inner_crd(rdscan_B2.out_crd())

        dropout.set_inner_crd(crd.out_crd_inner())
        dropout.set_outer_crd(crd.out_crd_outer())
        dropout.set_inner_ref(rdscan_B2.out_ref())

        if debug_sim:
            inner_ref_in.append(rdscan_B2.out_ref())
            print("ref inner in:", remove_emptystr(inner_ref_in))

            inner_ref_out.append(dropout.out_ref())
            print("ref inner out:", remove_emptystr(inner_ref_out))

            inner_crd.append(dropout.out_crd(0))
            print("ref crd:", remove_emptystr(inner_crd))

        val_B.set_load(dropout.out_ref())

        vals_X.set_input(val_B.out_val())

        wrscan_X1.set_input(dropout.out_crd(1))
        wrscan_X2.set_input(dropout.out_crd(0))

        # print("Timestep", time, "\t Out:", dropout.out_crd(0))

        rdscan_B1.update()
        rdscan_B2.update()
        crd.update()
        dropout.update()
        val_B.update()
        vals_X.update()
        wrscan_X1.update()
        wrscan_X2.update()

        done = wrscan_X1.out_done() and wrscan_X2.out_done() and vals_X.out_done()
        time += 1

    wrscan_X1.autosize()
    wrscan_X2.autosize()
    vals_X.autosize()

    # print("out_arr: ", wrscan_X1.get_arr())
    out_crds = [wrscan_X1.get_arr(), wrscan_X2.get_arr()]
    out_segs = [wrscan_X1.get_seg_arr(), wrscan_X2.get_seg_arr()]
    out_val = vals_X.get_arr()

    if debug_sim:
        print("out_segs", out_segs)
        print("out_crds", out_crds)
        print("out_val", out_val)
        print("gold:", gold_tup)

    if out_val == []:
        assert out_val == gold_tup
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        assert (check_point_tuple(out_tup, gold_tup))


# FIXME: Might not be necessary 
# @pytest.mark.parametrize("dim1", [4, 16, 32, 64])
# def test_dropout_1d(dim1, debug_sim):
#     in_ = [x + 1 for x in range(dim1)]
#     in1 = in_ + ['S0', 'D']
#     print("Input:", in1)
#     in2 = dim1 / 2
#     # crd_nums = np.arange(dim1)
#     # crd = crd_nums.tolist() + ['S0', 'D']
#     # assert (len(in1) == len(in1))

#     drop_prob = 0.4
#     # keep_prob = 1 - drop_prob
#     # print(in1)
#     np.random.seed(0)
#     prob = np.random.rand(len(in_))
#     d = prob < drop_prob
#     in_ = np.multiply(in_, d)
#     # in_ = in_ / keep_prob

#     # print(in_)
#     gold_val = in_[in_ != 0].tolist() + ['S0', 'D']
#     # gold_val = torch.nn.functional.dropout(torch.from_numpy(np.arange(dim1)), p=0.5).numpy().tolist() + ['S0', 'D']

#     dropout = RandomDropout(dimension=1, drop_probability=drop_prob, debug=debug_sim)

#     done = False
#     time = 0
#     out_val = []
#     # max1.set_in2(in2)
#     while not done and time < TIMEOUT:
#         if len(in1) > 0:
#             dropout.set_crd(0, in1.pop(0))

#         if time < dim1:
#             dropout.set_predicate(prob[time], drop_prob)

#         dropout.update()

#         out_val.append(dropout.out_crd(0))

#         # print("Timestep", time, "\t Out:", dropout.out_crd(0))

#         done = dropout.out_done()
#         time += 1

#     out_val = remove_emptystr(out_val)
#     print("Test:", out_val)
#     print("Ref:", gold_val)

#     assert (out_val == gold_val)
