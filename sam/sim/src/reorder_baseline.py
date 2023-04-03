import math
from .base import *
from .joiner import *
from .parellelize import *
from .rd_scanner import *


class Reorder_baseline(Primitive):
    def __init__(self, crd_arr=[], seg_arr=[],  sf=8, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.in_crd = []
        self.in_ref = []
        self.sf = sf
        self.ocrd_i = ""

        self.parellelize_block_ref = Parellelize(parellelize_factor=sf, debug=self.debug)
        self.parellelize_block_crd = Parellelize(parellelize_factor=sf, debug=self.debug)
        
        self.rd_scanners = []
        for i in range(sf):
            self.rd_scanners.append(CompressedCrdRdScan(crd_arr=crd_arr, seg_arr=seg_arr, debug=self.debug))
        #print(len(self.rd_scanners), self.rd_scanners)
        self.layers_num = int(math.log2(sf))
        self.union_tree = []
        temp_sf = sf
        for i in range(self.layers_num):
            temp_arr = []
            temp_sf = self.sf // (2 ** (i+1))
            for j in range(temp_sf):
                temp_arr.append(Merge_block(debug=self.debug, name="Union_" + str(i) + "_" + str(j)))
            #nprint(temp_arr)
            self.union_tree.append(temp_arr)

        self.final_ref = None
        self.final_crd = None

        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail = True


    def input_crd(self, crd):
        if crd != None and crd != "":
            self.in_crd.append(crd)

    def input_ref(self, ref):
        if ref != None and ref != "":
            self.in_ref.append(ref)

    def return_final_crd(self):
        return self.final_crd

    def return_final_ref(self):
        return self.final_ref

    def out_ocrd_i(self):
        return self.ocrd_i


    def check_backpressure(self):
        if self.backpressure_en:
            copy_backpressure = self.ready_backpressure
            self.ready_backpressure = True
            return copy_backpressure
        return True

    def set_backpressure(self, backpressure):
        if not backpressure:
            self.ready_backpressure = False

    def update_ready(self):
        if self.backpressure_en:
            if len(self.in_crd) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True

    def update(self):
        self.update_done()
        self.update_ready()
        if len(self.in_crd) > 0:
            self.block_start = False
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            
            if self.done:
                return
            crd = ""
            ref = ""
            if len(self.in_ref) > 0 and len(self.in_crd) > 0:
                crd = self.in_crd.pop(0)
                ref = self.in_ref.pop(0)

            self.parellelize_block_crd.add_tokens(crd)
            self.parellelize_block_ref.add_tokens(ref)


            self.parellelize_block_crd.update()
            self.parellelize_block_ref.update()
            
            for j in range(self.sf):
                #print("ADDING FOR ", j, self.parellelize_block_ref.return_tokens()[j], self.parellelize_block_crd.return_tokens()[j])
                self.rd_scanners[j].set_in_ref(self.parellelize_block_ref.return_tokens()[j])
                self.rd_scanners[j].set_in_crd(self.parellelize_block_crd.return_tokens()[j])
                #print(j, "th node")
                self.rd_scanners[j].update() #set_in_ref(self.parellelize_block_ref.return_tokens()[j])
            
            for i in range(self.layers_num):
                for j in range(len(self.union_tree[i])):
                    if i == 0:
                        #print("---", 2*j, 2*j + 1)
                        #print(self.rd_scanners[2*j].out_ref(), self.rd_scanners[2*j + 1].out_ref())
                        self.union_tree[i][j].set_in1(self.rd_scanners[2*j].out_ref(), self.rd_scanners[2*j].out_crd(), self.rd_scanners[2*j].return_upper_crd())
                        self.union_tree[i][j].set_in2(self.rd_scanners[2*j + 1].out_ref(), self.rd_scanners[2*j + 1].out_crd(), self.rd_scanners[2*j + 1].return_upper_crd())
                        self.union_tree[i][j].update()
                    else:
                        self.union_tree[i][j].set_in1(self.union_tree[i-1][2*j].out_ref_min(), self.union_tree[i-1][2*j].out_crd(), self.union_tree[i-1][2*j].out_ocrd_i())
                        self.union_tree[i][j].set_in2(self.union_tree[i-1][2*j + 1].out_ref_min(), self.union_tree[i-1][2*j+1].out_crd(), self.union_tree[i-1][2*j+1].out_ocrd_i())
                        self.union_tree[i][j].update()

            self.final_crd = self.union_tree[self.layers_num - 1][0].out_crd()
            self.final_ref = self.union_tree[self.layers_num - 1][0].out_ref_min()
            if not is_stkn(self.final_crd) and not is_stkn(self.final_ref):
                self.ocrd_i = self.union_tree[self.layers_num - 1][0].out_ocrd_i()
            else:
                self.ocrd_i = ""
            if is_stkn(self.final_crd):
                self.final_crd = decrement_stkn(self.final_crd)
            if is_stkn(self.final_ref):
                self.final_ref = decrement_stkn(self.final_ref)

            if self.final_crd == "D" and self.final_ref == "D":
                self.done = True
        if self.debug:
            print("DEBUG REORDER BASELINE: ", "\n Out final crd: ", self.final_crd, "\n Out final ref", self.final_ref, "\n ", self.ocrd_i)
