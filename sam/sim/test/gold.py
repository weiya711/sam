import scipy.sparse
import scipy.io
import os
import math
import numpy as np

from sam.sim.test.test import check_point_tuple, remove_zeros, convert_point_tuple, convert_ndarr_point_tuple, \
    get_point_list, read_inputs
from sam.util import TnsFileLoader, round_sparse, ScipyTensorShifter, \
    SUITESPARSE_FORMATTED_PATH, SUITESPARSE_PATH, FROSTT_PATH, VALIDATION_OUTPUT_PATH

KDIM = 256

ss_dir = SUITESPARSE_PATH
ss_formatted_dir = SUITESPARSE_FORMATTED_PATH
frostt_dir = FROSTT_PATH
validate_dir = VALIDATION_OUTPUT_PATH


def check_gold_matmul(ssname, debug_sim, cast, out_crds, out_segs, out_val, out_format="ss01"):
    # CSR
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()

    B_scipy = B_tensor
    C_scipy = B_tensor.transopose()

    gold_nd = (B_scipy * C_scipy).toarray()
    transpose = out_format[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        if debug_sim:
            print("Out:", out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_elemmul(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    # CSR
    B_dirname = os.path.join(ss_formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B1_seg = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B1_crd = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    # CSR
    C_dirname = os.path.join(ss_formatted_dir, ssname, "shift", "ds01")
    C_shape_filename = os.path.join(C_dirname, "tensor_C_mode_shape")
    C_shape = read_inputs(C_shape_filename)

    C1_seg_filename = os.path.join(C_dirname, "tensor_C_mode_1_seg")
    C1_seg = read_inputs(C1_seg_filename)
    C1_crd_filename = os.path.join(C_dirname, "tensor_C_mode_1_crd")
    C1_crd = read_inputs(C1_crd_filename)

    C_vals_filename = os.path.join(C_dirname, "tensor_C_mode_vals")
    C_vals = read_inputs(C_vals_filename, float)

    B_scipy = scipy.sparse.csr_matrix((B_vals, B1_crd, B1_seg), shape=B_shape)
    C_scipy = scipy.sparse.csr_matrix((C_vals, C1_crd, C1_seg), shape=C_shape)

    gold_nd = (B_scipy.multiply(C_scipy)).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_identity(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    B_dirname = os.path.join(ss_formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B1_seg = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B1_crd = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    B_scipy = scipy.sparse.csr_matrix((B_vals, B1_crd, B1_seg), shape=B_shape)

    gold_nd = B_scipy.toarray()
    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_elemadd(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    shifter = ScipyTensorShifter()
    B_scipy = B_tensor
    C_scipy = shifter.shiftLastMode(B_scipy)

    gold_nd = (B_scipy + C_scipy).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_vecmul_ji(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    return check_gold_mat_vecmul(ssname, debug_sim, out_crds, out_segs, out_val, format_str)


def check_gold_mat_vecmul_ij(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    return check_gold_mat_vecmul(ssname, debug_sim, out_crds, out_segs, out_val, format_str)


def check_gold_mat_vecmul(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    # CSR
    B_dirname = os.path.join(ss_formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B1_seg = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B1_crd = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    c_dirname = os.path.join(ss_formatted_dir, ssname, "other")
    c_fname = [f for f in os.listdir(c_dirname) if ssname + "-vec_mode1" in f]
    assert len(c_fname) == 1, "Should only have one 'other' folder that matches"
    c_fname = c_fname[0]
    c_dirname = os.path.join(c_dirname, c_fname)

    c_shape = B_shape[1]

    c0_seg_filename = os.path.join(c_dirname, "tensor_C_mode_0_seg")
    c_seg0 = read_inputs(c0_seg_filename)
    c0_crd_filename = os.path.join(c_dirname, "tensor_C_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "tensor_C_mode_vals")
    c_vals = read_inputs(c_vals_filename, float)

    B_scipy = scipy.sparse.csr_matrix((B_vals, B1_crd, B1_seg), shape=B_shape)
    c_nd = np.zeros(c_shape)

    for i in range(len(c_crd0)):
        val = c_vals[i]
        crd = c_crd0[i]
        c_nd[crd] = val

    gold_nd = (B_scipy @ c_nd)
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Vec2:\n", c_nd)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_sddmm(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    B_dirname = os.path.join(ss_formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B1_seg = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B1_crd = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    C_shape = (B_shape[0], KDIM)
    C_vals = np.arange(math.prod(C_shape)).reshape(C_shape)

    D_shape = (KDIM, B_shape[1])
    D_vals = np.arange(math.prod(D_shape)).reshape(D_shape[::-1]).transpose()

    B_scipy = scipy.sparse.csr_matrix((B_vals, B1_crd, B1_seg), shape=B_shape)

    gold_nd = (B_scipy.multiply(C_vals @ D_vals)).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_vals)
        print("Dense Mat3:\n", D_vals)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        if debug_sim:
            print("Out:", out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_residual(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    C_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in C_tensor.data]
        C_tensor = scipy.sparse.csr_matrix((data, C_tensor.indices, C_tensor.indptr), dtype=int)

    b_dirname = os.path.join(ss_formatted_dir, ssname, "mat_residual")
    b_shape = C_tensor.shape[0]
    b0_crd_filename = os.path.join(b_dirname, "tensor_b_mode_0_crd")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "tensor_b_mode_vals")
    b_vals = read_inputs(b_vals_filename, float)

    d_dirname = os.path.join(ss_formatted_dir, ssname, "mat_residual")
    d_shape = C_tensor.shape[1]
    d0_crd_filename = os.path.join(d_dirname, "tensor_d_mode_0_crd")
    d_crd0 = read_inputs(d0_crd_filename)

    d_vals_filename = os.path.join(d_dirname, "tensor_d_mode_vals")
    d_vals = read_inputs(d_vals_filename, float)


    C_scipy = C_tensor
    b_nd = np.zeros(b_shape)
    d_nd = np.zeros(d_shape)

    for i in range(len(b_crd0)):
        val = b_vals[i]
        crd = b_crd0[i]
        b_nd[crd] = val

    for i in range(len(d_crd0)):
        val = d_vals[i]
        crd = d_crd0[i]
        d_nd[crd] = val

    gold_nd = b_nd - (C_scipy @ d_nd)
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Vec1:\n", b_nd)
        print("Dense Mat1:\n", C_scipy.toarray())
        print("Dense Vec2:\n", d_nd)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_mattransmul(ssname, debug_sim, out_crds, out_segs, out_val, format_str):
    # CSR
    B_dirname = os.path.join(ss_formatted_dir, ssname, "orig", "ds01")
    B_shape_filename = os.path.join(B_dirname, "tensor_B_mode_shape")
    B_shape = read_inputs(B_shape_filename)

    B1_seg_filename = os.path.join(B_dirname, "tensor_B_mode_1_seg")
    B1_seg = read_inputs(B1_seg_filename)
    B1_crd_filename = os.path.join(B_dirname, "tensor_B_mode_1_crd")
    B1_crd = read_inputs(B1_crd_filename)

    B_vals_filename = os.path.join(B_dirname, "tensor_B_mode_vals")
    B_vals = read_inputs(B_vals_filename, float)

    b_dirname = os.path.join(ss_formatted_dir, ssname, "other")
    b_fname = [f for f in os.listdir(b_dirname) if ssname + "-vec_mode1" in f]
    assert len(b_fname) == 1, "Should only have one 'other' folder that matches"
    b_fname = b_fname[0]
    b_dirname = os.path.join(b_dirname, b_fname)

    b_shape = B_shape[1]

    b0_crd_filename = os.path.join(b_dirname, "tensor_C_mode_0_crd")
    b_crd0 = read_inputs(b0_crd_filename)

    b_vals_filename = os.path.join(b_dirname, "tensor_C_mode_vals")
    b_vals = read_inputs(b_vals_filename, float)

    c_dirname = os.path.join(ss_formatted_dir, ssname, "other")
    c_fname = [f for f in os.listdir(c_dirname) if ssname + "-vec_mode0" in f]
    assert len(c_fname) == 1, "Should only have one 'other' folder that matches"
    c_fname = c_fname[0]
    c_dirname = os.path.join(c_dirname, c_fname)

    c_shape = B_shape[0]

    c0_crd_filename = os.path.join(c_dirname, "tensor_C_mode_0_crd")
    c_crd0 = read_inputs(c0_crd_filename)

    c_vals_filename = os.path.join(c_dirname, "tensor_C_mode_vals")
    c_vals = read_inputs(c_vals_filename, float)

    s1 = 2
    s2 = 2

    B_scipy = scipy.sparse.csr_matrix((B_vals, B1_crd, B1_seg), shape=B_shape)
    c_nd = np.zeros(c_shape)
    b_nd = np.zeros(b_shape)

    for i in range(len(c_crd0)):
        val = c_vals[i]
        crd = c_crd0[i]
        c_nd[crd] = val

    for i in range(len(b_crd0)):
        val = b_vals[i]
        crd = b_crd0[i]
        b_nd[crd] = val

    gold_nd = s1 * B_scipy.T @ c_nd + s2 * b_nd
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Vec1:\n", b_nd)
        print("Dense Mat1:\n", B_scipy.transpose().toarray())
        print("Dense Vec2:\n", c_nd)
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_mat_elemadd3(ssname, debug_sim, cast, out_crds, out_segs, out_val, format_str):
    # MTX
    B_tensor = scipy.io.mmread(os.path.join(ss_dir, ssname + ".mtx")).tocsr()
    if cast:
        data = [round_sparse(x) for x in B_tensor.data]
        B_tensor = scipy.sparse.csr_matrix((data, B_tensor.indices, B_tensor.indptr), dtype=int)

    shifter = ScipyTensorShifter()
    B_scipy = B_tensor
    C_scipy = shifter.shiftLastMode(B_scipy)
    D_scipy = shifter.shiftLastMode(C_scipy)

    gold_nd = (B_scipy + C_scipy + D_scipy).toarray()
    transpose = format_str[-2:] == "10"
    if transpose:
        gold_nd = gold_nd.transpose()

    gold_tup = convert_ndarr_point_tuple(gold_nd)

    if debug_sim:
        print("Out segs:", out_segs)
        print("Out crds:", out_crds)
        print("Out vals:", out_val)
        print("Dense Mat1:\n", B_scipy.toarray())
        print("Dense Mat2:\n", C_scipy.toarray())
        print("Dense Gold:", gold_nd)
        print("Gold:", gold_tup)

    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_elemadd(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-plus2-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_ttv(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-ttv-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
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


def check_gold_tensor3_ttm(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-ttm-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    print("GOLD:", gold_tup)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


def check_gold_tensor3_innerprod(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    if frosttname == "fb1k":
        assert out_val == [1066.0]
    else:
        assert False, "Gold not entered yet"


def check_gold_tensor3_mttkrp(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    validation_path = os.path.join(validate_dir, "frostt-taco", frosttname + "-mttkrp-taco.tns")
    tnsLoader = TnsFileLoader(False)
    dims, coordinates, vals = tnsLoader.load(validation_path)
    coordinates.append(vals)
    gold_tup = convert_point_tuple(coordinates)
    print("GOLD:", gold_tup)
    if not out_val:
        assert out_val == gold_tup
    elif not gold_tup:
        assert all([v == 0 for v in out_val])
    else:
        out_tup = convert_point_tuple(get_point_list(out_crds, out_segs, out_val))
        out_tup = remove_zeros(out_tup)
        assert (check_point_tuple(out_tup, gold_tup))


# ---------------- OTHER CHECKS (TODO later) ---------------- #
def check_gold_tensor3_identity(frosttname, debug_sim, out_crds, out_segs, out_val, format_str):
    pass
