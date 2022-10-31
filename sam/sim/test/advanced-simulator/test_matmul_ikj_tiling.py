import pytest
import time
import scipy.sparse
from sam.sim.src.rd_scanner import UncompressCrdRdScan, CompressedCrdRdScan
from sam.sim.src.wr_scanner import ValsWrScan
from sam.sim.src.joiner import Intersect2, Union2
from sam.sim.src.compute import Multiply2, Add2
from sam.sim.src.crd_manager import CrdDrop, CrdHold
from sam.sim.src.repeater import Repeat, RepeatSigGen
from sam.sim.src.accumulator import Reduce
from sam.sim.src.channel import memory_block
from sam.sim.src.accumulator import SparseAccumulator1, SparseAccumulator2
from sam.sim.src.token import *
from sam.sim.test.test import *
from sam.sim.test.gold import *
import os
import csv
cwd = os.getcwd()
formatted_dir = os.getenv('SUITESPARSE_FORMATTED_PATH', default=os.path.join(cwd, 'mode-formats'))


# FIXME: Figureout formats
@pytest.mark.skipif(
    os.getenv('CI', 'false') == 'true',
    reason='CI lacks datasets',
)
@pytest.mark.suitesparse
def test_matmul_ikj_tiling(samBench, ssname, check_gold, debug_sim, report_stats, fill=0):

    struct = {"Bi00": 2, "Bk00": 3, "Ck00": 3, "Cj00": 2, "Bi0": 2, "Bk0": 2, "Ck0": 2, "Cj0": 2}
    
    fiberlookup_Bi00 = UncompressCrdRdScan(dim=struct["Bi00"], debug=debug_sim)
    fiberlookup_Bk00 = UncompressCrdRdScan(dim=struct["Bk00"], debug=debug_sim)
    repsiggen_i00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ci00 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck00 = UncompressCrdRdScan(dim=struct["Ck00"], debug=debug_sim)
    fiberlookup_Cj00 = UncompressCrdRdScan(dim=struct["Cj00"], debug=debug_sim)
    repsiggen_j00 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bj00 = Repeat(debug=debug_sim, statistics=report_stats)

    in_ref_B00 = [0, 'D']
    in_ref_C00 = [0, 'D']
    in_ref_B0 = []
    in_ref_C0 = []
    done = False
    time_cnt = 0

    glb_model_b_ = []
    glb_model_c_ = []
    glb_model_b = memory_block(name="GLB_B")
    glb_model_c = memory_block(name="GLB_C")

    mem_model_b_ = []
    mem_model_c_ = []
    mem_model_b = memory_block(name="B")
    mem_model_c = memory_block(name="C")
 
    glb_model_x = memory_block(name="X", level = "glb2global")
    mem_model_x = memory_block(name="X", level = "mem2glb")
    
    flag_glb = False
    flag = False
    tiled_done = False
    tiled_done_processed = False
    while not done and time_cnt < TIMEOUT:
        print(time_cnt)
        print("Check")

        if len(in_ref_B00) > 0:
           fiberlookup_Bi00.set_in_ref(in_ref_B00.pop(0))
        fiberlookup_Bk00.set_in_ref(fiberlookup_Bi00.out_ref())
        repsiggen_i00.set_istream(fiberlookup_Bi00.out_crd()) 
        if len(in_ref_C00) > 0:
            repeat_Ci00.set_in_ref(in_ref_C00.pop(0))
        repeat_Ci00.set_in_repsig(repsiggen_i00.out_repsig())
        fiberlookup_Ck00.set_in_ref(repeat_Ci00.out_ref())
        fiberlookup_Cj00.set_in_ref(fiberlookup_Ck00.out_ref())
        repsiggen_j00.set_istream(fiberlookup_Cj00.out_crd())
        repeat_Bj00.set_in_ref(fiberlookup_Bk00.out_ref())
        repeat_Bj00.set_in_repsig(repsiggen_j00.out_repsig())
            
        if glb_model_b.valid_tile() and glb_model_c.valid_tile():
            flag_glb = True
            in_ref_B0 = [0, 'D']
            in_ref_C0 = [0, 'D']
            fiberlookup_Bi0 = UncompressCrdRdScan(dim=struct["Bi0"], debug=debug_sim)
            fiberlookup_Bk0 = UncompressCrdRdScan(dim=struct["Bk0"], debug=debug_sim)
            repsiggen_i0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Ci0 = Repeat(debug=debug_sim, statistics=report_stats)
            fiberlookup_Ck0 = UncompressCrdRdScan(dim=struct["Ck0"], debug=debug_sim)
            fiberlookup_Cj0 = UncompressCrdRdScan(dim=struct["Cj0"], debug=debug_sim)
            repsiggen_j0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Bj0 = Repeat(debug=debug_sim, statistics=report_stats)
            glb_model_b.valid_tile_recieved()
            glb_model_c.valid_tile_recieved()

        glb_model_b.add_tile(repeat_Bj00.out_ref(), 4)
        glb_model_c.add_tile(fiberlookup_Cj00.out_ref(), 4)
        #glb_model_b.set_downstream_token(mem_model_b.return_token())
        #glb_model_c.set_downstream_token(mem_model_c.return_token())
        glb_model_b.check_if_done(mem_model_b.out_done())
        glb_model_c.check_if_done(mem_model_c.out_done())


        if flag_glb:
            if len(in_ref_B0) > 0:
                fiberlookup_Bi0.set_in_ref(in_ref_B0.pop(0))
            fiberlookup_Bk0.set_in_ref(fiberlookup_Bi0.out_ref())
            repsiggen_i0.set_istream(fiberlookup_Bi0.out_crd()) 
            if len(in_ref_C0) > 0:
                repeat_Ci0.set_in_ref(in_ref_C0.pop(0))
            repeat_Ci0.set_in_repsig(repsiggen_i0.out_repsig())
            fiberlookup_Ck0.set_in_ref(repeat_Ci0.out_ref())
            fiberlookup_Cj0.set_in_ref(fiberlookup_Ck0.out_ref())
            repsiggen_j0.set_istream(fiberlookup_Cj0.out_crd())
            repeat_Bj0.set_in_ref(fiberlookup_Bk0.out_ref())
            repeat_Bj0.set_in_repsig(repsiggen_j0.out_repsig())
        
            mem_model_b_.append(repeat_Bj0.out_ref())
            mem_model_c_.append(fiberlookup_Cj0.out_ref())
            mem_model_b.add_tile(repeat_Bj0.out_ref(), 4)
            mem_model_c.add_tile(fiberlookup_Cj0.out_ref(), 4)


        if mem_model_b.valid_tile() and mem_model_c.valid_tile():
            flag = True
            tiled_done_processed = False
            in_ref_B = [0, 'D']
            in_ref_C = [0, 'D']
            print("Updating Token") 
            B_k00 = glb_model_b.token()%struct["Bk00"]
            B_i00 = glb_model_b.token()//struct["Bk00"]
            B_k0 = mem_model_b.token()%struct["Bk0"]
            B_i0 = mem_model_b.token()//struct["Bk0"]

            C_j00 = glb_model_c.token()%struct["Cj00"]
            C_k00 = glb_model_c.token()//struct["Cj0"]
            C_j0 = mem_model_c.token()%struct["Cj0"]
            C_k0 = mem_model_c.token()//struct["Cj0"]

            B_dirname = os.path.join(formatted_dir, ssname, "orig", "ss01")
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
            B_vals_filename = os.path.join(B_dirname, "B_vals.txt")
            B_vals = read_inputs(B_vals_filename, float)
            C_dirname = os.path.join(formatted_dir, ssname, "shift-trans", "ss01")
            C_shape_filename = os.path.join(C_dirname, "C_shape.txt")
            C_shape = read_inputs(C_shape_filename)
            C0_seg_filename = os.path.join(C_dirname, "C0_seg.txt")
            C_seg0 = read_inputs(C0_seg_filename)
            C0_crd_filename = os.path.join(C_dirname, "C0_crd.txt")
            C_crd0 = read_inputs(C0_crd_filename)
            C1_seg_filename = os.path.join(C_dirname, "C1_seg.txt")
            C_seg1 = read_inputs(C1_seg_filename)
            C1_crd_filename = os.path.join(C_dirname, "C1_crd.txt")
            C_crd1 = read_inputs(C1_crd_filename)
            C_vals_filename = os.path.join(C_dirname, "C_vals.txt")
            C_vals = read_inputs(C_vals_filename, float)

            fiberlookup_Bi_19 = CompressedCrdRdScan(crd_arr=B_crd0, seg_arr=B_seg0, debug=debug_sim, statistics=report_stats)
            fiberlookup_Bk_14 = CompressedCrdRdScan(crd_arr=B_crd1, seg_arr=B_seg1, debug=debug_sim, statistics=report_stats)
            repsiggen_i_17 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Ci_16 = Repeat(debug=debug_sim, statistics=report_stats)
            fiberlookup_Ck_15 = CompressedCrdRdScan(crd_arr=C_crd0, seg_arr=C_seg0, debug=debug_sim, statistics=report_stats)
            intersectk_13 = Intersect2(debug=debug_sim, statistics=report_stats)
            crdhold_5 = CrdHold(debug=debug_sim, statistics=report_stats)
            fiberlookup_Cj_12 = CompressedCrdRdScan(crd_arr=C_crd1, seg_arr=C_seg1, debug=debug_sim, statistics=report_stats)
            arrayvals_C_8 = Array(init_arr=C_vals, debug=debug_sim, statistics=report_stats)
            crdhold_4 = CrdHold(debug=debug_sim, statistics=report_stats)
            repsiggen_j_10 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
            repeat_Bj_9 = Repeat(debug=debug_sim, statistics=report_stats)
            arrayvals_B_7 = Array(init_arr=B_vals, debug=debug_sim, statistics=report_stats)
            mul_6 = Multiply2(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3 = SparseAccumulator1(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3_drop_crd_inner = StknDrop(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3_drop_crd_outer = StknDrop(debug=debug_sim, statistics=report_stats)
            spaccumulator1_3_drop_val = StknDrop(debug=debug_sim, statistics=report_stats)
            fiberwrite_Xvals_0 = ValsWrScan(size=1 * B_shape[0] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
            fiberwrite_X1_1 = CompressWrScan(seg_size=B_shape[0] + 1, size=B_shape[0] * C_shape[1], fill=fill, debug=debug_sim, statistics=report_stats)
            fiberwrite_X0_2 = CompressWrScan(seg_size=2, size=B_shape[0], fill=fill, debug=debug_sim, statistics=report_stats) 
            mem_model_b.valid_tile_recieved()
            mem_model_c.valid_tile_recieved()
        
        if flag:
            if len(in_ref_B) > 0:
                fiberlookup_Bi_19.set_in_ref(in_ref_B.pop(0))
            fiberlookup_Bk_14.set_in_ref(fiberlookup_Bi_19.out_ref())
            repsiggen_i_17.set_istream(fiberlookup_Bi_19.out_crd())
            if len(in_ref_C) > 0:
                repeat_Ci_16.set_in_ref(in_ref_C.pop(0))
            repeat_Ci_16.set_in_repsig(repsiggen_i_17.out_repsig())
            fiberlookup_Ck_15.set_in_ref(repeat_Ci_16.out_ref())
            
            intersectk_13.set_in1(fiberlookup_Ck_15.out_ref(), fiberlookup_Ck_15.out_crd())
            intersectk_13.set_in2(fiberlookup_Bk_14.out_ref(), fiberlookup_Bk_14.out_crd())
            crdhold_5.set_outer_crd(fiberlookup_Bi_19.out_crd())
            crdhold_5.set_inner_crd(intersectk_13.out_crd())
            fiberlookup_Cj_12.set_in_ref(intersectk_13.out_ref1())
            arrayvals_C_8.set_load(fiberlookup_Cj_12.out_ref())
            crdhold_4.set_outer_crd(crdhold_5.out_crd_outer())
            crdhold_4.set_inner_crd(fiberlookup_Cj_12.out_crd())
            repsiggen_j_10.set_istream(fiberlookup_Cj_12.out_crd())
            repeat_Bj_9.set_in_ref(intersectk_13.out_ref2())
            repeat_Bj_9.set_in_repsig(repsiggen_j_10.out_repsig())
            arrayvals_B_7.set_load(repeat_Bj_9.out_ref())
            mul_6.set_in1(arrayvals_B_7.out_val())
            mul_6.set_in2(arrayvals_C_8.out_val())
            spaccumulator1_3_drop_crd_outer.set_in_stream(crdhold_4.out_crd_outer())
            spaccumulator1_3_drop_crd_inner.set_in_stream(crdhold_4.out_crd_inner())
            spaccumulator1_3_drop_val.set_in_stream(mul_6.out_val())
            spaccumulator1_3.set_crd_outer(spaccumulator1_3_drop_crd_outer.out_val())
            spaccumulator1_3.set_crd_inner(spaccumulator1_3_drop_crd_inner.out_val())
            spaccumulator1_3.set_val(spaccumulator1_3_drop_val.out_val())
            fiberwrite_Xvals_0.set_input(spaccumulator1_3.out_val())
            fiberwrite_X1_1.set_input(spaccumulator1_3.out_crd_inner())
            fiberwrite_X0_2.set_input(spaccumulator1_3.out_crd_outer())
            mem_model_b.check_if_done([fiberwrite_Xvals_0.out_done(), fiberwrite_X1_1.out_done(), fiberwrite_X0_2.out_done()])
            mem_model_c.check_if_done([fiberwrite_Xvals_0.out_done(), fiberwrite_X1_1.out_done(), fiberwrite_X0_2.out_done()])

        fiberlookup_Bi00.update()
        fiberlookup_Bk00.update()
        repsiggen_i00.update()
        repeat_Ci00.update() 
        fiberlookup_Ck00.update()
        fiberlookup_Cj00.update()
        repsiggen_j00.update()
        repeat_Bj00.update()
        glb_model_b.update(time_cnt)
        glb_model_c.update(time_cnt)
        print("--------")
    
        if flag_glb:
            fiberlookup_Bi0.update()
            fiberlookup_Bk0.update()
            repsiggen_i0.update()
            repeat_Ci0.update() 
            fiberlookup_Ck0.update()
            fiberlookup_Cj0.update()
            repsiggen_j0.update()
            repeat_Bj0.update()
        
        mem_model_b.update(time_cnt)
        mem_model_c.update(time_cnt)

        if flag:
            fiberlookup_Bi_19.update()
            fiberlookup_Bk_14.update()
            repsiggen_i_17.update()
            repeat_Ci_16.update()
            fiberlookup_Ck_15.update()
            intersectk_13.update()
            crdhold_5.update()
            fiberlookup_Cj_12.update()
            arrayvals_C_8.update()
            crdhold_4.update()
            repsiggen_j_10.update()
            repeat_Bj_9.update()
            arrayvals_B_7.update()
            mul_6.update()
            spaccumulator1_3_drop_crd_outer.update()
            spaccumulator1_3_drop_crd_inner.update()
            spaccumulator1_3_drop_val.update()
            spaccumulator1_3.update()

            fiberwrite_Xvals_0.update()
            fiberwrite_X1_1.update()
            fiberwrite_X0_2.update()

            tiled_done = fiberwrite_X0_2.out_done() and fiberwrite_X1_1.out_done() and fiberwrite_Xvals_0.out_done()

            mem_model_x.add_upstream(tilecoord=[B_i00, B_k00, C_k00, C_j00, B_i0, B_k0, C_k0, C_j0],
                    data=[fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr(), fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr(),
                        fiberwrite_Xvals_0.get_arr()], valid = tiled_done)
            mem_model_x.input_token_(glb_model_x.return_token())
            glb_model_x.add_upstream(mem_model_x.token(), mem_model_x.get_size(), mem_model_x.out_done())
            glb_model_x.input_token_("D")

            mem_model_x.out_update(time_cnt)
            glb_model_x.out_update(time_cnt)

            if tiled_done:
                fiberwrite_X0_2.autosize()
                fiberwrite_X1_1.autosize()
                fiberwrite_Xvals_0.autosize()

                out_crds = [fiberwrite_X0_2.get_arr(), fiberwrite_X1_1.get_arr()]
                out_segs = [fiberwrite_X0_2.get_seg_arr(), fiberwrite_X1_1.get_seg_arr()]
                out_vals = fiberwrite_Xvals_0.get_arr()
                #print("TILE IDs: Bi00: ", glb_model_b.token(), " ", glb_model_c.token(), " ", mem_model_b.token(), " ",mem_model_c.token()) 
               
                print("TILE IDs: ", B_i00, B_k00, C_k00, C_j00, " , " , B_i0, B_k0, C_k0, C_j0) 
                
                print("Values: ", out_vals)
                if check_gold:
                    print("Checking gold...")
                    check_gold_matmul(ssname, debug_sim, out_crds, out_segs, out_vals, "ss01")


            if mem_model_c.token() == "D" and glb_model_c.token() == "D" and mem_model_b.token() == "D" and glb_model_b.token() == "D" and tiled_done:
                done = True
        
        print("###################")
        time_cnt += 1

        if time_cnt > 100000:
            return
