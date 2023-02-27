from .base import *


#################
# Read Scanners
#################


class CrdRdScan(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = ''
        self.curr_crd = ''

        self.in_ref = []

    def set_in_ref(self, in_ref):
        if in_ref != '' and in_ref is not None:
            self.in_ref.append(in_ref)

    def out_ref(self):
        return self.curr_ref

    def out_crd(self):
        return self.curr_crd


class repeated_token_dopper(Primitive):
    def __init__(self, name="val"):
        self.last_token = ""
        self.new_token = ""
        self.output_token = ""
        self.name = name
        self.done = False
        self.first_stop = True
        self.in_token = []

    def add_input(self, token):
        if token != "" and token is not None:
            self.in_token.append(token)


    def add_token(self, token):
        # if token != "":
        #     self.last_token = self.new_token
        if token != "":
            self.new_token = token

    def get_token(self):
        return self.output_token

    def update(self):
        
        #if self.debug:
        #    print("REPEATED_BLK name:", self.name, self.first_stop, self.new_token)
        if len(self.in_token) > 0:
            self.new_token = self.in_token.pop(0)

        if self.done:
            return
        self.output_token = ""
        if self.first_stop and self.new_token != "" and not is_stkn(self.new_token):
            self.first_stop = False
        elif self.first_stop and is_stkn(self.new_token):
            return
                
        if (self.last_token == "D" or self.new_token != "") and (self.last_token != self.new_token or self.last_token == "D") and not(is_stkn(self.new_token) and is_stkn(self.last_token)):
            # print(self.name, "update1")
            self.output_token = self.last_token
            self.last_token = self.new_token
        elif self.new_token != "" and is_stkn(self.last_token) and is_stkn(self.new_token) and self.new_token != self.last_token:
            #print(self.name, "update2")
            self.last_token = self.new_token
            self.output_token = self.last_token
            self.last_token = ""
            self.new_token = ""
        if self.output_token == "D":
            self.done = True
        #if self.debug:
        #    print("REPEATED_BLK name:", self.name, self.last_token, self.new_token, " returns----", self.output_token)


class Reorder_and_split(CrdRdScan):
    def __init__(self, crd_arr=[], seg_arr=[], skip=True, counter_mode_dense=False, limit=10, sf=1, **kwargs):
        super().__init__(**kwargs)

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr
        self.start_addr = 0
        self.stop_addr = 0
        self.curr_addr = 0
        self.in_ref = []
        self.in_crd = []
        self.end_fiber = False
        self.curr_ref = ''
        self.curr_crd = ''

        self.curr_ref_i = ''
        self.curr_crd_i = ''
        self.out_ref_i_ = ''
        self.out_crd_i_ = ''

        self.curr_ref_k = ''
        self.curr_ref_k = ''

        self.emit_fiber_stkn = False
        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)
        self.begin = True
        self.counter = 0
        self.counter_end = limit
        self.split_factor = sf
        self.buffer_crd = []
        self.buffer_ref = []
        self.state = "resting"
        self.last_state = -1

        self.nxt_state_table = {}
        self.seg_tuple = {}
        self.out_ref_k_ = 0
        self.temp_ref_k_ = 0
        self.out_crd_k_ = ""
        self.old_k_outer_copy = ""
        self.ref_offset = 0

        self.last_state_ = "none"
        self.counter_dense = counter_mode_dense
        self.outer_lvl = False

        if self.get_stats:
            self.num_min_calls = 0
            self.num_min_values_returned = 0

    def set_ref(self, ref):
        if ref != "" and ref is not None:
            self.in_ref.append(ref)

    def set_crd(self, crd):
        if crd != "" and crd is not None:
            self.in_crd.append(crd)


    def set_input(self, ref, crd):
        # print("INPUT coming in: ", ref, crd)
        if crd != "" and ref != "" and crd is not None and ref is not None:
            self.in_ref.append(ref)
            self.in_crd.append(crd)
        elif crd != "" and (ref == "" or ref is None) and crd is not None:
            self.in_crd.append(crd)
        elif ref != "" and (crd == "" or crd is not None) and ref is not None:
            self.in_ref.append(ref)
        # print(self.in_crd, self.in_ref)

    def out_crd_i(self):
        #if not self.output:
        #    self.output = True
        return self.out_crd_i_
        #else:
        #    return ""

    def out_crd_k(self):
        if isinstance(self.curr_crd, int):
            return self.curr_crd % self.split_factor
        else:
            return self.curr_crd

    def out_ref_i(self):
        #if not self.output:
        #    self.output = True
        if isinstance(self.out_ref_i_, int):
            return self.out_ref_i_
        return self.out_ref_i_
        #else:
        #    return ""

    def out_ref_k(self):
        # if isinstance(self.curr_ref, int):
        #     return self.curr_ref % self.split_factor
        # else:
        return self.curr_ref

    def reset(self):
        # print("RESET HARD")
        self.ref_offset = len(self.nxt_state_table.keys())
        self.counter = 0
        self.temp_ref_k_ = 0
        self.nxt_state_table = {}
        self.last_state = -1
        self.out_ref_k_ = 0
        self.out_crd_k_ = ""

    def out_ref_k_outer(self):
        if self.outer_lvl: 
            return self.out_ref_k_
        return ""

    def out_crd_k_outer(self):
        if self.outer_lvl:    
            return self.out_crd_k_
        return ""

    def min_table(self):
        min_val = 10000000000000000000000000
        for a in self.nxt_state_table.keys():
            if self.nxt_state_table[a] != -1:
                if min_val > self.nxt_state_table[a]:
                    min_val = self.nxt_state_table[a]
        # if min_val == 10000000000000000000000000:
        #     return
        if self.debug:
            print("MIN VALUE Calc ", min_val)
        if self.get_stats:
            self.num_min_calls += 1
        if self.counter_dense:
            return self.counter + 1
        return min_val // self.split_factor

    def check_len(self, arr):
        i = 0
        for a in self.nxt_state_table.keys():
            if self.nxt_state_table[a] != -1:
                i += 1
        return i

    def return_statistics(self):
        if not self.get_stats:
            return {}
        dic = {}
        dic["next_item_table_size"] = len(self.nxt_state_table.keys())
        dic["next_min_called"] = self.num_min_calls
        dic["num_min_vals_returned"] = self.num_min_values_returned / self.num_min_calls
        return dic


    def get_valid_streams(self):
        return_crd = []
        return_ref = []
        k = 0
        flag = False
        if self.check_len([""]) == 0:
            flag = True
        for i in self.nxt_state_table.keys():
            if self.nxt_state_table[i] != -1:
                if self.nxt_state_table[i] >= self.counter * self.split_factor and self.nxt_state_table[i] < (self.counter + 1) * self.split_factor:
                    return_crd.append(i)
                    return_ref.append(k + self.ref_offset)
            k += 1
        if self.debug:
            print("FLAG printing the min of the table and ge the vals ", return_crd, return_ref)
        if self.get_stats:
            self.num_min_values_returned += len(return_crd)
        return return_crd, return_ref, flag

    def update(self):
        self.outer_lvl = True
        if self.state == "resting" and len(self.in_ref) > 0 and len(self.in_crd) > 0:
            if self.last_state == "S2_out":
                self.reset()
            self.curr_ref_i = self.in_ref.pop(0)
            self.curr_crd_i = self.in_crd.pop(0)
            if not isinstance(self.curr_ref_i, int) and is_stkn(self.curr_ref_i):
                self.counter = self.min_table()
                self.out_crd_k_ = self.counter
                self.out_ref_k_ = self.temp_ref_k_
                # self.temp_ref_k_ += 1
                # if isinstance(self.out_ref_k_, int):
                #     self.out_ref_k_ += 1
                if self.last_state_ == "none":
                    self.state = "repeat_i_rows"
                    self.next_state = "none"
                    self.stop_lvl = self.curr_ref_i
                else:
                    # stkn = self.curr_ref_i #ncrement_stkn(self.curr_ref_i)
                    #self.curr_crd = increment_stkn(self.curr_ref_i)
                    #self.curr_ref = increment_stkn(self.curr_ref_i)
                    #self.out_ref_i_ = stkn
                    #self.out_crd_i_ = stkn
                    #self.state = self.next_state 
                    self.state = "S1_out"
                    self.next_state = "repeat_i_rows"
                    self.stop_lvl = self.curr_ref_i
                    if self.debug:
                        print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd_i,
                              "curr_ref", self.curr_ref_i, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                              self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor, "ref val", self.ref_val, self.output)
                    #return

            elif not isinstance(self.curr_ref_i, int) and self.curr_ref_i == "D":
                self.state = "done_state"
            else:
                self.nxt_state_table[self.curr_crd_i] = -1
                self.crd_sub_arr = self.crd_arr[self.seg_arr[self.curr_ref_i] : self.seg_arr[self.curr_ref_i + 1]]
                self.seg_tuple[self.curr_crd_i] = [self.seg_arr[self.curr_ref_i], self.seg_arr[self.curr_ref_i + 1]]
                self.state = "reading_row"
                self.ref_val = self.seg_arr[self.curr_ref_i]
                self.curr_ref = self.ref_val
                self.last_state = "resting"
                self.output = False
            
            if self.debug:
                print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd_i,
                      "curr_ref", self.curr_ref_i, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                      self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor, "ref val", self.ref_val, self.output)
        if self.state == "reading_row":
            # self.temp_ref_k_ = self.out_ref_k_
            if len(self.crd_sub_arr) > 0:
                self.state = "process_row"
                self.last_state = "reading_row"
            else:
                self.state = "resting"
            if self.debug:
                print("In case of reading row", self.state, self.temp_ref_k_)
        if self.state == "process_row":
            if self.debug:
                print("PROCESSING_ROW", self.curr_crd_i, self.crd_sub_arr, self.counter)
            if len(self.crd_sub_arr) > 0:
                self.curr_crd = self.crd_sub_arr.pop(0)
                self.curr_ref = self.ref_val

                if self.curr_crd >= self.counter * self.split_factor and self.curr_crd < (self.counter + 1) * self.split_factor:
                    if self.curr_ref_i != self.out_ref_i_:
                        self.out_ref_i_ = self.curr_ref_i
                        self.out_crd_i_ = self.curr_crd_i
                        self.out_crd_k_ = self.counter
                        self.out_ref_k_ = self.temp_ref_k_
                        self.last_state = "process_row"
                        self.last_state_ = "process_row"
                        # copy_tup =  self.seg_tuple[self.out_crd_i_]
                        # self.seg_tuple[self.out_crd_i_] = [self.seg_arr[self.out_ref_i_]:  ]
                    else:
                        self.out_ref_i_ = ""
                        self.out_crd_i_ = ""

                else: #if self.split_factor != 1:
                    if self.debug:
                        print("else_case", self.curr_crd, (self.counter + 1) * self.split_factor, len(self.in_crd), self.last_state)
                    if self.curr_crd >= (self.counter + 1) * self.split_factor:
                        self.state = "resting"
                        self.nxt_state_table[self.curr_crd_i] = self.curr_crd
                        copy_tup =  self.seg_tuple[self.curr_crd_i]
                        self.seg_tuple[self.out_crd_i_] = [self.ref_val, copy_tup[1]]


                        if self.last_state == "process_row":
                            if self.split_factor == 1:
                                self.state = "resting"
                            #    self.next_state = "resting"
                            else:
                                self.state = "S0_out"
                                self.next_state = "resting"
                            self.nxt_state_table[self.curr_crd_i] = self.curr_crd
                        self.curr_crd = ""
                        self.curr_ref = ""
                        self.out_ref_i_ = ""
                        self.out_crd_i_ = ""
                    self.ref_val = self.ref_val + 1
                if self.debug:
                    print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd,
                          "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                          self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor, "ref val",
                          self.ref_val, self.output, self.curr_ref_i, self.curr_crd_i, self.out_ref_i_, self.out_crd_i_)
                    print(self.crd_arr, self.seg_arr, self.crd_sub_arr)
                    print(self.nxt_state_table)
                self.ref_val = self.ref_val + 1
                return
            else:
                if self.last_state == "process_row":
                    if self.split_factor == 1:
                        self.state = "resting"
                    else:
                        # print("EMIT A STOP TOKEN AND POP NEXT")
                        self.state = "S0_out"
                        self.next_state = "resting"
                else:
                    self.state = "resting"
            if self.debug:
                print(self.state)
                # print("in process_row", self.next_state, self.state)

        if self.state == "repeat_i_rows":
            if self.debug:
                print("State 2, ", self.nxt_state_table)
            self.curr_crd_i_row, self.curr_ref_i_row, flag = self.get_valid_streams()
            if self.debug:
                print(self.curr_crd_i_row, self.curr_ref_i_row, flag, self.nxt_state_table)
            if not flag: #len(self.curr_crd_i_row) > 0:
                self.state = "reading_row2"
            elif flag:
                if self.stop_lvl == "S0":
                    self.state = "S2_out"
                    self.next_state = "resting"
                if self.stop_lvl == "S1":
                    self.state = "S3_out"
                    self.next_state = "resting"
            if self.debug:
                print("repeat i rows: ", self.state, self.next_state)

        if self.state == "reading_row2":
            # self.temp_ref_k_ = self.out_ref_k_
            if len(self.curr_crd_i_row) == 0:
                self.state = "S1_out"
                self.next_state = "repeat_i_rows"
                self.counter = self.min_table()
                self.out_crd_k_ = self.counter
                # self.temp_ref_k_ += 1
                self.out_ref_k_ = self.temp_ref_k_
                # if isinstance(self.out_ref_k_, int):
                #     self.out_ref_k_ += 1
            else:
                self.curr_crd_i = self.curr_crd_i_row.pop(0)
                self.curr_ref_i = self.curr_ref_i_row.pop(0)
                self.crd_sub_arr = self.crd_arr[self.seg_tuple[self.curr_crd_i][0] : self.seg_tuple[self.curr_crd_i][1]]
                self.ref_val = self.seg_tuple[self.curr_crd_i][0]
                self.state = "process_row2"
                self.output = True
                self.out_crd_k_ = self.counter
                self.out_ref_k_ = self.temp_ref_k_
                if self.debug:
                    print("Reorder_Blk from 3: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd,
                          "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                          self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor)
                    print("nxt table ------", self.nxt_state_table, self.curr_crd_i_row, self.curr_crd_i)
        if self.state == "process_row2":
            if self.debug:
                print("CHECK THE ARR: ", self.crd_sub_arr, self.counter)
            if len(self.crd_sub_arr) > 0:
                self.curr_crd = self.crd_sub_arr.pop(0)
                self.curr_ref = self.ref_val
                if self.curr_crd >= self.counter * self.split_factor and self.curr_crd < (self.counter + 1) * self.split_factor:
                    if self.debug:
                        print("IF CASE")
                    if self.curr_ref_i != self.out_ref_i_:
                        self.out_ref_i_ = self.curr_ref_i
                        self.out_crd_i_ = self.curr_crd_i
                        self.out_crd_k_ = self.counter
                        self.out_ref_k_ = self.temp_ref_k_
                        self.last_state = "process_row2"
                    else:
                        self.out_ref_i_ = ""
                        self.out_crd_i_ = ""

                    # self.output = True
                else:
                    if self.curr_crd >= (self.counter + 1) * self.split_factor: # and self.check_len(self.nxt_state_table.keys()) == 0: # len(self.nxt_state_table) > 0:
                        copy_tup =  self.seg_tuple[self.curr_crd_i]
                        self.seg_tuple[self.out_crd_i_] = [self.ref_val, copy_tup[1]]


                        if self.debug:
                            print("ELSE CASE")
                        if self.split_factor == 1:
                            # self.state = "S0_out"
                            self.state = "reading_row2"
                        else:
                            self.state = "S0_out"
                            self.next_state = "reading_row2"
                        self.nxt_state_table[self.curr_crd_i] = self.curr_crd
                    elif self.check_len(self.nxt_state_table.keys()) == 0: # len(self.nxt_state_table) == 0:
                        self.state = "repeat_i_rows"
                    self.curr_crd = ""
                    self.curr_ref = ""
                    self.out_ref_i_ = ""
                    self.out_crd_i_ = ""
                if self.debug:
                    print("--- Reorder_Blk from 4: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd,
                          "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                          self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor) 
                    print(self.crd_arr, self.seg_arr)
                    print("nxt table ------", self.nxt_state_table, self.curr_crd_i_row, self.crd_sub_arr)
                self.ref_val = self.ref_val + 1
                return
            elif self.last_state == "process_row2":
                if self.split_factor != 1:  #self.state = "S0_out"
                    self.state = "S0_out"
                
                if self.debug:
                    print(self.nxt_state_table)
                    print(self.check_len(self.nxt_state_table.keys()))
                #self.nxt_state_table.pop(self.curr_crd_i)
                self.nxt_state_table[self.curr_crd_i] = -1
                if self.check_len(self.nxt_state_table.keys()) == 0:
                    self.state = "repeat_i_rows"
                else:
                    if self.split_factor != 1:
                        self.next_state = "reading_row2"
                    else:
                        self.state = "reading_row2"
                self.curr_crd = ""
                self.curr_ref = ""
                self.out_ref_i_ = ""
                self.out_crd_i_ = ""
                if self.debug:
                    print("+++ Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd,
                          "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                          self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor) 
                    print(self.crd_arr, self.seg_arr)
                    print("nxt table ------", self.nxt_state_table, self.curr_crd_i_row)
 

        if self.state == "S1_out":
            self.curr_crd = "S1"
            self.curr_ref = "S1"
            self.out_ref_i_ = "S0"
            self.out_crd_i_ = "S0"
            self.outer_lvl = False
            self.state = self.next_state
            # print("###############################")
            # print(self.last_state)
            # print(self.next_state)
            self.temp_ref_k_ += 1

            if self.debug:
                print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd", self.curr_crd,
                      "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_, "crd_arr_check",
                      self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor)
            return
        if self.state == "S0_out":
            self.curr_crd = "S0"
            self.curr_ref = "S0"
            self.out_ref_i_ = ""
            self.out_crd_i_ = ""
            self.out_crd_k_ = ""
            self.out_ref_k_ = ""
            self.state = self.next_state
            if self.debug:
                print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd",
                      self.curr_crd, "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_,
                      "crd_arr_check", self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor)
            return
        if self.state == "S2_out":
            self.curr_crd = "S2"
            self.curr_ref = "S2"
            self.out_ref_i_ = "S1"
            self.out_crd_i_ = "S1"
            self.out_ref_k_ = "S0"
            self.out_crd_k_ = "S0"
            self.state = self.next_state
            self.last_state = "S2_out"
            if self.debug:
                print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd",
                      self.curr_crd, "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_,
                       "crd_arr_check", self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor)
            return
        if self.state == "S3_out":
            self.curr_crd = "S3"
            self.curr_ref = "S3"
            self.out_ref_i_ = "S2"
            self.out_crd_i_ = "S2"
            self.out_ref_k_ = "S1"
            self.out_crd_k_ = "S1"
            self.state = self.next_state
            self.last_state = "S2_out"
            if self.debug:
                print("Reorder_Blk: state", self.state, "in_crd", self.in_crd, "in_ref", self.in_ref, "in_crd",
                      self.curr_crd, "curr_ref", self.curr_ref, "out_ref_i", self.out_ref_i_, "out_crd_i", self.out_crd_i_,
                       "crd_arr_check", self.crd_sub_arr, "limits:", self.counter*self.split_factor, (self.counter+1)*self.split_factor)
            return 
        if self.state == "done_state":
            self.curr_crd = "D"
            self.curr_ref = "D"
            self.out_ref_i_ = "D"
            self.out_crd_i_ = "D"
            self.out_ref_k_ = "D"
            self.out_crd_k_ = "D"
            self.done = True

    def _emit_stkn_code(self):
        self.end_fiber = True

        if len(self.in_ref) > 0:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
        else:
            self.emit_fiber_stkn = True
            stkn = ''
        self.curr_crd = stkn
        self.curr_ref = stkn
        self.curr_addr = 0
        self.stop_addr = 0
        self.start_addr = 0

    def _set_curr(self):
        self.curr_ref = self.curr_addr
        self.curr_crd = self.crd_arr[self.curr_addr]
        if self.get_stats:
            if self.curr_ref not in self.unique_refs:
                self.unique_refs.append(self.curr_ref)
            if self.curr_crd not in self.unique_crds:
                self.unique_crds.append(self.curr_crd)
            self.total_outputs += 1


    def update_not_used(self):
        self.update_done()
        if len(self.in_ref) > 0 or (self.skip and len(self.in_crd_skip) > 0):
            self.block_start = False

        # Process skip token first and save
        if len(self.in_crd_skip) > 0 and self.skip_processed:
            self.curr_skip = self.in_crd_skip.pop(0)
            if self.skip_stkn_cnt == self.out_stkn_cnt and isinstance(self.curr_skip, int) \
                    and self.curr_skip < self.prev_crd:
                # ignore the skip if it's too small
                self.skip_processed = True
                if self.get_stats:
                    self.intersection_behind_cnt += 1
            elif self.skip_stkn_cnt < self.out_stkn_cnt:
                # ignore the skip if it's a fiber behind
                self.skip_processed = True
                if self.get_stats:
                    self.fiber_behind_cnt += 1
            else:
                self.skip_processed = False

            if is_stkn(self.curr_skip):
                self.skip_stkn_cnt += 1

        curr_in_ref = None
        # After Done token has been seen and outputted, do nothing
        if self.curr_crd == 'D' or self.curr_ref == 'D' or self.done:
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.curr_crd = ''
            self.curr_ref = ''

        # Scanner needs to emit stop token and the next element has finally arrived.
        # Previously set emit_fiber_stkn to True but wait on next in_ref
        elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_crd = stkn
            self.curr_ref = stkn

            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.emit_fiber_stkn = False

        # There exists another input reference at the segment and
        # either at the start of computation or end of fiber
        elif len(self.in_ref) > 0 and (self.end_fiber or self.begin):
            self.begin = False
            self.end_fiber = False

            curr_in_ref = self.in_ref.pop(0)

            # Input reference is out of bounds
            if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_slen:
                raise Exception('Not enough elements in seg array')

            # Input reference is a stop or done token, so forward that token (and set done if done token)
            elif is_stkn(curr_in_ref) or curr_in_ref == 'D':
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0
                self.curr_crd = curr_in_ref
                self.curr_ref = curr_in_ref
                self.end_fiber = True
                if curr_in_ref == 'D':
                    self.done = True

            # See 'N' 0-token which immediately emits a stop token and ends the fiber
            elif is_0tkn(curr_in_ref):
                self.curr_crd = 'N'
                self.curr_ref = 'N'
                self.end_fiber = True
                self.emit_fiber_stkn = True
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0

            # Default case where input reference is an integer value
            # which means to get the segment
            else:
                self.start_addr = self.seg_arr[curr_in_ref]
                self.stop_addr = self.seg_arr[curr_in_ref + 1]
                self.curr_addr = self.start_addr

                # This case is if the segment has no coordinates (i.e. 5, 5)
                if self.curr_addr >= self.stop_addr:
                    # End of fiber, get next input reference
                    self._emit_stkn_code()

                # Default behave normally and emit the coordinates in the segment
                else:
                    if self.skip and not self.skip_processed:
                        # assert self.out_stkn_cnt == self.skip_stkn_cnt
                        curr_range = self.crd_arr[self.start_addr: self.stop_addr]
                        # Skip to next coordinate
                        if isinstance(self.curr_skip, int) \
                                and self.curr_skip > self.prev_crd:
                            print("RD SCAN: SKIP HERE")
                            # If coordinate skipped to exists, emit that
                            if self.curr_skip in curr_range:
                                self.curr_addr = curr_range.index(self.curr_skip) + self.start_addr
                                self._set_curr()
                                if self.get_stats:
                                    self.elements_skipped += curr_range.index(self.curr_skip) + 1
                                    self.skip_cnt += 1

                            # Else emit smallest coordinate larger than the one provided by skip
                            else:
                                larger = [i for i in curr_range if i > self.curr_skip]
                                if not larger:
                                    self._emit_stkn_code()
                                    if self.get_stats:
                                        self.elements_skipped += len(curr_range)
                                        self.skip_cnt += 1
                                else:
                                    val_larger = min(larger)
                                    self.curr_addr = curr_range.index(val_larger) + self.start_addr
                                    self._set_curr()
                                    if self.get_stats:
                                        self.elements_skipped += curr_range.index(val_larger) + 1
                                        self.skip_cnt += 1

                        # Early exit from skip
                        elif is_stkn(self.curr_skip):
                            self._emit_stkn_code()
                        self.skip_processed = True
                    # Else behave normally
                    else:
                        self._set_curr()

        # Finished emitting coordinates and have reached the end of the fiber for this level
        elif (self.curr_addr == self.stop_addr - 1 or self.curr_addr == self.meta_clen - 1) and \
                not self.begin:
            # End of fiber, get next input reference
            self._emit_stkn_code()

        # Base case: increment address and reference by 1 and get next coordinate
        elif len(self.in_ref) > 0 and not self.begin:
            default_behavior = True
            if self.skip and not self.skip_processed:
                # assert self.out_stkn_cnt == self.skip_stkn_cnt
                curr_range = self.crd_arr[self.start_addr: self.stop_addr]
                if isinstance(self.curr_skip, int) \
                        and self.curr_skip > self.prev_crd:
                    print("RD SCAN: SKIP HERE")
                    # If coordinate skipped to exists, emit that
                    if self.curr_skip in curr_range:
                        self.curr_addr = curr_range.index(self.curr_skip) + self.start_addr
                        self._set_curr()
                        if self.get_stats:
                            self.elements_skipped += curr_range.index(self.curr_skip) + 1
                            self.skip_cnt += 1

                    # Else emit smallest coordinate larger than the one provided by skip
                    else:
                        larger = [i for i in curr_range if i > self.curr_skip]
                        if not larger:
                            self._emit_stkn_code()
                            if self.get_stats:
                                self.elements_skipped += len(curr_range)
                                self.skip_cnt += 1
                        else:
                            val_larger = min(larger)
                            self.curr_addr = curr_range.index(val_larger) + self.start_addr
                            self._set_curr()
                            if self.get_stats:
                                self.elements_skipped += curr_range.index(val_larger) + 1
                                self.skip_cnt += 1

                    default_behavior = False
                elif is_stkn(self.curr_skip):
                    self._emit_stkn_code()
                    default_behavior = False
                self.skip_processed = True

            if default_behavior:
                self.curr_addr += 1
                self._set_curr()

        # Default stall (when done)
        elif not self.begin:
            self.curr_ref = ''
            self.curr_crd = ''

        # Needed for skip lists
        if is_stkn(self.curr_crd):
            self.out_stkn_cnt += 1
        # Needed for skip lists
        if self.skip_stkn_cnt < self.out_stkn_cnt:
            # ignore the skip if it's a fiber behind
            self.skip_processed = True
        # Needed for skip lists
        if isinstance(self.curr_crd, int):
            self.prev_crd = self.curr_crd

        if self.get_stats and is_stkn(self.curr_crd):
            self.stop_count += 1

        # Debugging print statements
        if self.debug:
            print("DEBUG: C RD SCAN:"
                  "\n \tCurr crd:", self.curr_crd, "\t curr ref:", self.curr_ref,
                  "\n curr addr:", self.curr_addr, "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\n end fiber:", self.end_fiber, "\t curr input:", curr_in_ref,
                  "\n skip in:", self.curr_skip, "\t skip processed", self.skip_processed, "\t prev crd:",
                  self.prev_crd,
                  "\n Out stkn cnt:", self.out_stkn_cnt, "\t Skip stkn cnt:", self.skip_stkn_cnt)

    def update_noskip(self):
        curr_in_ref = None
        # After Done token has been seen and outputted, do nothing
        if self.curr_crd == 'D' or self.curr_ref == 'D' or self.done:
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.curr_crd = ''
            self.curr_ref = ''

        # Scanner needs to emit stop token and the next element has finally arrived.
        # Previously set emit_fiber_stkn to True but wait on next in_ref
        elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_crd = stkn
            self.curr_ref = stkn

            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0
            self.emit_fiber_stkn = False

        # There exists another input reference at the segment and
        # either at the start of computation or end of fiber
        elif len(self.in_ref) > 0 and (self.end_fiber or (self.curr_crd is None or self.curr_ref is None)):
            if self.curr_crd is None or self.curr_ref is None:
                assert (self.curr_crd == self.curr_ref)
            self.end_fiber = False

            curr_in_ref = self.in_ref.pop(0)

            # Input reference is out of bounds
            if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_slen:
                raise Exception('Not enough elements in seg array')

            # Input reference is a stop or done token, so forward that token (and set done if done token)
            elif is_stkn(curr_in_ref) or curr_in_ref == 'D':
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0
                self.curr_crd = curr_in_ref
                self.curr_ref = curr_in_ref
                self.end_fiber = True
                if curr_in_ref == 'D':
                    self.done = True

            # See 'N' 0-token which immediately emits a stop token and ends the fiber
            elif is_0tkn(curr_in_ref):
                self.end_fiber = True

                if len(self.in_ref) > 0:
                    next_in = self.in_ref[0]
                    if is_stkn(next_in):
                        self.in_ref.pop(0)
                        stkn = increment_stkn(next_in)
                    else:
                        stkn = 'S0'
                else:
                    self.emit_fiber_stkn = True
                    stkn = ''
                self.curr_crd = stkn
                self.curr_ref = stkn
                self.curr_addr = 0
                self.stop_addr = 0
                self.start_addr = 0

            # Default case where input reference is an integer value
            # which means to get the segment
            else:
                self.start_addr = self.seg_arr[curr_in_ref]
                self.stop_addr = self.seg_arr[curr_in_ref + 1]
                self.curr_addr = self.start_addr

                # This case is if the segment has no coordinates (i.e. 5, 5)
                if self.curr_addr >= self.stop_addr:
                    # End of fiber, get next input reference
                    self.end_fiber = True

                    if len(self.in_ref) > 0:
                        next_in = self.in_ref[0]
                        if is_stkn(next_in):
                            self.in_ref.pop(0)
                            stkn = increment_stkn(next_in)
                        else:
                            stkn = 'S0'
                    else:
                        self.emit_fiber_stkn = True
                        stkn = ''
                    self.curr_crd = stkn
                    self.curr_ref = stkn

                # Default behave normally and emit the coordinates in the segment
                else:
                    self.curr_crd = self.crd_arr[self.curr_addr]
                    self.curr_ref = self.curr_addr

        # Finished emitting coordinates and have reached the end of the fiber for this level
        elif (self.curr_addr == self.stop_addr - 1 or self.curr_addr == self.meta_clen - 1) and \
                self.curr_crd is not None and self.curr_ref is not None:
            # End of fiber, get next input reference
            self.end_fiber = True

            if len(self.in_ref) > 0:
                next_in = self.in_ref[0]
                if is_stkn(next_in):
                    self.in_ref.pop(0)
                    stkn = increment_stkn(next_in)
                else:
                    stkn = 'S0'
            else:
                self.emit_fiber_stkn = True
                stkn = ''
            self.curr_crd = stkn
            self.curr_ref = stkn
            self.curr_addr = 0
            self.stop_addr = 0
            self.start_addr = 0

        # Base case: increment address and reference by 1 and get next coordinate
        elif len(self.in_ref) > 0 and self.curr_crd is not None and self.curr_ref is not None:
            self.curr_addr += 1
            self.curr_ref = self.curr_addr
            self.curr_crd = self.crd_arr[self.curr_addr]

        # Default stall (when done)
        elif self.curr_crd is not None and self.curr_ref is not None:
            self.curr_ref = ''
            self.curr_crd = ''

        if self.debug:
            print("DEBUG: C RD SCAN: \t "
                  "Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                  "\t start addr:", self.start_addr, "\t stop addr:", self.stop_addr,
                  "\t end fiber:", self.end_fiber, "\t curr input:", curr_in_ref)

    def set_crd_skip(self, in_crd):
        assert in_crd is None or is_valid_crd(in_crd)
        if in_crd != '' and in_crd is not None:
            if is_stkn(in_crd):
                idx = last_stkn(self.in_crd_skip)
                if idx is not None:
                    # Flush coordinates
                    self.in_crd_skip = self.in_crd_skip[:idx + 1]
            self.in_crd_skip.append(in_crd)
