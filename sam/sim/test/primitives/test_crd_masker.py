import pytest
from sam.sim.src.base import remove_emptystr
from sam.sim.src.compute import Add2, Multiply2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
from sam.sim.src.crd_masker import RandomDropout
from sam.sim.src.rd_scanner import CompressedCrdRdScan
from sam.sim.test.primitives.test_intersect import TIMEOUT
from sam.sim.test.test import *
import numpy as np
import torch


@pytest.mark.parametrize("dim1", [4, 16, 32, 64])
def test_dropout_1d(dim1, debug_sim):
    in_ = [x + 1 for x in range(dim1)]
    in1 = in_ + ['S0', 'D']
    print("Input:", in1)
    in2 = dim1 / 2
    # crd_nums = np.arange(dim1)
    # crd = crd_nums.tolist() + ['S0', 'D']
    # assert (len(in1) == len(in1))

    drop_prob = 0.4
    # keep_prob = 1 - drop_prob
    # print(in1)
    np.random.seed(0)
    prob = np.random.rand(len(in_))
    d = prob < drop_prob
    in_ = np.multiply(in_, d)
    # in_ = in_ / keep_prob

    # print(in_)
    gold_val = in_[in_ != 0].tolist() + ['S0', 'D']
    # gold_val = torch.nn.functional.dropout(torch.from_numpy(np.arange(dim1)), p=0.5).numpy().tolist() + ['S0', 'D']

    drop = RandomDropout(dimension=1, drop_probability=drop_prob, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    # max1.set_in2(in2)
    while not done and time < TIMEOUT:
        if len(in1) > 0:
            drop.set_crd(0, in1.pop(0))

        if time < dim1:
            drop.set_predicate(prob[time], drop_prob)

        drop.update()

        out_val.append(drop.out_crd(0))

        # print("Timestep", time, "\t Out:", drop.out_crd(0))

        done = drop.out_done()
        time += 1

    out_val = remove_emptystr(out_val)
    print("Ref:", gold_val)

    assert (out_val == gold_val)


@pytest.mark.parametrize("dim", [4])
def test_dropout_2d(dim, debug_sim, max_val=1000, fill=0):
    in_mat_crds1, in_mat_segs1 = gen_n_comp_arrs(2, dim)
    in_mat_vals1 = gen_val_arr(len(in_mat_crds1[-1]), max_val, -max_val)

    in1_tup = convert_point_tuple(get_point_list(in_mat_crds1, in_mat_segs1, in_mat_vals1))
    nd1 = convert_point_tuple_ndarr(in1_tup, dim)

    drop_prob = 0.8
    # keep_prob = 1 - drop_prob
    np.random.seed(0)

    print(nd1.shape)
    # pytest.set_trace()
    prob = np.random.rand(*nd1.shape)
    d = prob < drop_prob
    gold_nd = np.multiply(nd1, d)
    gold_tup = convert_ndarr_point_tuple(gold_nd)
    # in_ = in_ / keep_prob

    flat_prob = prob.flatten()

    # print(in_)
    # gold_val = torch.nn.functional.dropout(torch.from_numpy(np.arange(dim1)), p=0.5).numpy().tolist() + ['S0', 'D']

    rdscan_B1 = CompressedCrdRdScan(crd_arr=in_mat_crds1[0], seg_arr=in_mat_segs1[0], debug=debug_sim)
    rdscan_B2 = CompressedCrdRdScan(crd_arr=in_mat_crds1[1], seg_arr=in_mat_segs1[1], debug=debug_sim)

    val_B = Array(init_arr=in_mat_vals1, debug=debug_sim)

    drop = RandomDropout(dimension=1, drop_probability=drop_prob, debug=debug_sim)

    done = False
    time = 0
    out_val = []
    in_ref_B = [0, 'D']
    # max1.set_in2(in2)
    while not done and time < TIMEOUT:
        if len(in_ref_B) > 0:
            rdscan_B1.set_in_ref(in_ref_B.pop(0))

        rdscan_B2.set_in_ref(rdscan_B1.out_ref())

        val_B.set_load(rdscan_B2.out_ref())

        drop.set_crd(0, rdscan_B1.out_crd())
        drop.set_crd(1, rdscan_B2.out_crd())

        if time < len(flat_prob):
            drop.set_predicate(flat_prob[time], drop_prob)

        drop.update()

        out_val.append(drop.out_crd(0))

        # print("Timestep", time, "\t Out:", drop.out_crd(0))

        done = drop.out_done()
        time += 1

    out_val = remove_emptystr(out_val)
    # print("Ref:", gold_val)

    assert (out_val == gold_val)


# @pytest.mark.parametrize("dim1", [4, 16, 32, 64])
# def test_exp_1d(dim1, debug_sim):
#     in1 = [x for x in range(dim1)] + ['S0', 'D']
#     in2 = None
#     # assert (len(in1) == len(in1))

#     gold_val = np.exp(np.arange(dim1)).tolist() + ['S0', 'D']

#     exp1 = Exp(debug=debug_sim)

#     done = False
#     time = 0
#     out_val = []
#     exp1.set_in2(in2)
#     while not done and time < TIMEOUT:
#         if len(in1) > 0:
#             exp1.set_in1(in1.pop(0))

#         exp1.update()

#         out_val.append(exp1.out_val())

#         print("Timestep", time, "\t Out:", exp1.out_val())

#         done = exp1.out_done()
#         time += 1

#     out_val = remove_emptystr(out_val)
#     print("Ref:", gold_val)
#     print("Out:", out_val)

#     assert (out_val == gold_val)


# @pytest.mark.parametrize("dim1", [4, 16, 32, 64])
# def test_scalar_mult_1d(dim1, debug_sim):
#     in1 = [x for x in range(dim1)] + ['S0', 'D']
#     in2 = 4

#     gold_val = np.arange(dim1) * in2
#     gold_val = gold_val.tolist() + ['S0', 'D']

#     scal1 = ScalarMult(debug=debug_sim)

#     done = False
#     time = 0
#     out_val = []
#     scal1.set_in2(in2)
#     while not done and time < TIMEOUT:
#         if len(in1) > 0:
#             scal1.set_in1(in1.pop(0))

#         scal1.update()

#         out_val.append(scal1.out_val())

#         print("Timestep", time, "\t Out:", scal1.out_val())

#         done = scal1.out_done()
#         time += 1

#     out_val = remove_emptystr(out_val)
#     print("Ref:", gold_val)
#     print("Out:", out_val)

#     assert (out_val == gold_val)
