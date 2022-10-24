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

    fiberlookup_Bi0 = UncompressCrdRdScan(dim=struct["Bi0"], debug=debug_sim)
    fiberlookup_Bk0 = UncompressCrdRdScan(dim=struct["Bk0"], debug=debug_sim)
    repsiggen_i0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Ci0 = Repeat(debug=debug_sim, statistics=report_stats)
    fiberlookup_Ck0 = UncompressCrdRdScan(dim=struct["Ck0"], debug=debug_sim)
    fiberlookup_Cj0 = UncompressCrdRdScan(dim=struct["Cj0"], debug=debug_sim)
    repsiggen_j0 = RepeatSigGen(debug=debug_sim, statistics=report_stats)
    repeat_Bj0 = Repeat(debug=debug_sim, statistics=report_stats)

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

    glb_ref_b = None
    glb_ref_c = None

    while not done and time_cnt < TIMEOUT:
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
            
        fiberlookup_Bi00.update()
        fiberlookup_Bk00.update()
        repsiggen_i00.update()
        repeat_Ci00.update() 
        fiberlookup_Ck00.update()
        fiberlookup_Cj00.update()
        repsiggen_j00.update()
        repeat_Bj00.update()

        #glb_model_b_.append(repeat_Bj00.out_ref())
        #glb_model_c_.append(fiberlookup_Cj00.out_ref())

        #print(time_cnt, " ", glb_model_b_, " ", glb_model_c_)

        # glb_model_b.add_tile(repeat_Bj00.out_ref())
        #glb_model_c.add_tile(fiberlookup_Cj00.out_ref())

        #glb_model_b.update(time_cnt)
        #glb_model_c.update(time_cnt)
        #print("--------")
        
        if glb_model_b.valid_tile() and glb_model_c.valid_tile():
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
        
        if True:
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
            
            fiberlookup_Bi0.update()
            fiberlookup_Bk0.update()
            repsiggen_i0.update()
            repeat_Ci0.update() 
            fiberlookup_Ck0.update()
            fiberlookup_Cj0.update()
            repsiggen_j0.update()
            repeat_Bj0.update()

        
        glb_model_b_.append(repeat_Bj00.out_ref())
        glb_model_c_.append(fiberlookup_Cj00.out_ref())
        print(time_cnt)#, " ", glb_model_b_, " ", glb_model_c_)

        glb_model_b.add_tile(repeat_Bj00.out_ref())
        glb_model_c.add_tile(fiberlookup_Cj00.out_ref())
        glb_model_b.set_downstream_token(mem_model_b.return_token())
        glb_model_c.set_downstream_token(mem_model_c.return_token())

        glb_model_b.update(time_cnt)
        glb_model_c.update(time_cnt)
        print("--------")
 
        if True:
            mem_model_b_.append(repeat_Bj0.out_ref())
            mem_model_c_.append(fiberlookup_Cj0.out_ref())
            #print(time_cnt, " ", mem_model_b_, " ", mem_model_c_)
            mem_model_b.add_tile(repeat_Bj0.out_ref())
            mem_model_c.add_tile(fiberlookup_Cj0.out_ref())
            mem_model_b.pop_tile_after(time_cnt, 10)
            mem_model_c.pop_tile_after(time_cnt, 10)
            
        mem_model_b.update(time_cnt)
        mem_model_c.update(time_cnt)

        if mem_model_b.valid_tile() and mem_model_c.valid_tile():        
            mem_model_b.valid_tile_recieved()
            mem_model_c.valid_tile_recieved()

        if mem_model_c.token() == "D" and glb_model_c.token() == "D" and mem_model_b.token() == "D" and glb_model_b.token() == "D":
            done = True

        print("###################")
        time_cnt += 1

        if time_cnt > 100000:
            return
