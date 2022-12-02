import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2, Divide2
from sam.sim.src.unary_alu import Max, Exp, ScalarMult
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv
cwd = os.getcwd()
formatted_dir = os.getenv('FROSTT_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.frostt
def test_tensor3_identity_dense(samBench, frosttname, check_gold, debug_sim, report_stats, fill=0):
    B_dirname = os.path.join(formatted_dir, frosttname, "orig", "ddd012")
    B_shape_filename = os.path.join(B_dirname, "B_shape.txt")
    B_shape = read_inputs(B_shape_filename)

    B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
    B_vals = read_inputs(B_vals_filename, float)

    fiberlookup_Bi_7 = UncompressCrdRdScan(dim=B_shape[0], debug=debug_sim, statistics=report_stats)
    fiberlookup_Bj_6 = UncompressCrdRdScan(dim=B_shape[1], debug=debug_sim, statistics=report_stats)
    fiberlookup_Bk_5 = UncompressCrdRdScan(dim=B_shape[2], debug=debug_sim, statistics=report_stats)
    arrayvals_B_4 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
    fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * B_shape[1] * B_shape[2], fill=fill, debug=debug_sim, statistics=report_stats)
    in_ref_B = [0, 'D']
    done = False
    time_cnt = 0

    while not done and time_cnt < TIMEOUT:
        if len(in_ref_B) > 0:
            fiberlookup_Bi_7.set_in_ref(in_ref_B.pop(0))
        fiberwrite_X0_3.set_input(fiberlookup_Bi_7.out_crd())
        fiberlookup_Bj_6.set_in_ref(fiberlookup_Bi_7.out_ref())
        fiberwrite_X1_2.set_input(fiberlookup_Bj_6.out_crd())
        fiberlookup_Bk_5.set_in_ref(fiberlookup_Bj_6.out_ref())
        fiberwrite_X2_1.set_input(fiberlookup_Bk_5.out_crd())
        arrayvals_B_4.set_load(fiberlookup_Bk_5.out_ref())
        fiberwrite_Xvals_0.set_input(arrayvals_B_4.out_val())
        fiberlookup_Bi_7.update()

        fiberwrite_X0_3.update()
        fiberlookup_Bj_6.update()
        fiberwrite_X1_2.update()
        fiberlookup_Bk_5.update()
        fiberwrite_X2_1.update()
        arrayvals_B_4.update()
        fiberwrite_Xvals_0.update()

        done = fiberwrite_Xvals_0.out_done()
        time_cnt += 1

    fiberwrite_Xvals_0.autosize()

    out_crds = []
    out_segs = []
    out_vals = fiberwrite_Xvals_0.get_arr()

    def bench():
        time.sleep(0.01)

    extra_info = dict()
    extra_info["dataset"] = frosttname
    extra_info["cycles"] = time_cnt
    extra_info["tensor_B_shape"] = B_shape
    sample_dict = fiberlookup_Bi_7.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bi_7" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X0_3.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X0_3" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bj_6.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bj_6" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X1_2.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X1_2" + "_" + k] = sample_dict[k]

    sample_dict = fiberlookup_Bk_5.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberlookup_Bk_5" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_X2_1.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_X2_1" + "_" + k] = sample_dict[k]

    sample_dict = arrayvals_B_4.return_statistics()
    for k in sample_dict.keys():
        extra_info["arrayvals_B_4" + "_" + k] = sample_dict[k]

    sample_dict = fiberwrite_Xvals_0.return_statistics()
    for k in sample_dict.keys():
        extra_info["fiberwrite_Xvals_0" + "_" + k] = sample_dict[k]

    if check_gold:
        print("Checking gold...")
        check_gold_tensor3_identity_dense(frosttname, debug_sim, out_crds, out_segs, out_vals, "ddd012")
    samBench(bench, extra_info)
