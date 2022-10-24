import numpy as np

def hash_tile(tile_id):
    hash_val = 0
    num = len(tile_id)//2
    cnt = 0
    encoder = 1
    encoder_diff = 100
    for a in reversed(tile_id):
        hash_val += encoder*a
        encoder *= encoder_diff
        cnt += 1
    return hash_val, hash_val//(encoder_diff**num)

class output_memory_block():
    def __init__(self, name = "B", level = None, size = 1000*2, debug=False, bandwidth = 2, length = 1, mode = "all_upacked"):
        self.name = name
        self.level = level
        self.size = size
        self.bandwidth = bandwidth
        self.length = length

        self.tile_ptrs_glb = []
        self.tile_ptrs_fifo = [] # virtual
        self.tile_ptrs_size = []

        self.mode = mode
        self.timestamp = None
        self.ready = True
        self.loading = False
        self.valid = False
        self.curr_tile = None
        self.curr_size = 0
        self.old_tile = None
        self.debug = debug
        self.outputed = False
        self.done = False
        self.done_received = False
        self.valid_processed = False
        self.child_ready = False

    def out_done(self):
        return self.done

    def out_ready(self):
        return self.ready

    def return_if_loading(self):
        return self.loading

    def check_if_done(self, arr):
        if isinstance(arr, bool):
            self.done_received = arr
            return
        token = True
        if len(arr) > 0:
            for i in arr:
                if not i:
                    token = False
                    self.done_received = False
                    return
        self.done_received = True
        return

    def add_upstream(self, tilecoord, data, valid):
        if not valid:
            self.valid_processed = False
        if valid and not self.valid_processed and self.level == "mem2glb": 
            self.valid_processed = True
            limits = len(data)-1
            size_levels = [0]*(limits//2)
            size_vals = len(data[len(data)-1])
            for lengths in range(len(size_levels)):
                size_levels[lengths] = len(data[lengths]) + len(data[lengths + limits//2])
            
            max_levels_size = max(max(size_levels), size_vals)
            self.tile_ptrs_size.append(max_levels_size)
            tile_hash, tile_glb = hash_tile(tilecoord)
            print("Tile pointer vales = ", tile_hash, " ", tile_glb)
            #tilecoord
            self.tile_ptrs_fifo.append(tile_glb)

        if valid and not self.valid_processed and self.level == "glb2global": 
            self.valid_processed = True
            self.tile_ptrs_fifo.append(tilecoord)
            #size = data
            self.tile_ptrs_size.append(data)

    
    def update(self, cyclenum):
        if self.level == "mem2glb":
            if self.ready and len(self.tile_ptrs_fifo) > 0:
                if self.curr_tile != None:
                    self.old_tile = self.curr_tile
                
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size = self.tile_ptrs_size.pop(0)
                self.loading = True
                self.ready = False
                self.done = False
                self.timestamp = cyclenum
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                self.loading = False
                self.done = True
            elif self.done and self.child_ready:
                self.ready = True

        if self.level == "glb2global":
            self.old_tile = None
            if self.ready and len(self.tile_ptrs_fifo) > 0 and (self.curr_tile == None or self.curr_tile == "D" or self.curr_tile == self.tile_ptrs_fifo[0]):
                #print("-0-0-0-0-0-0-0-0-0-")
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size += self.tile_ptrs_size.pop(0)
                self.done = False
                #self.loading = True
            elif self.ready and len(self.tile_ptrs_fifo) > 0 and self.curr_tile != None and self.curr_tile != self.tile_ptrs_fifo[0]:
                #if len(self.tile_ptrs_fifo) > 0 and self.curr_tile != self.tile_ptrs_fifo[0]:
                #print("***************************************************************************************")
                self.loading = True
                self.ready = False 
                self.done = False
                self.timestamp = cyclenum
                #self.curr_size = 0
            #elif self.ready and len(self.tile_ptrs_fifo) == 0 and self.curr_tile != None:
            #    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^") 
            #    self.loading = True
            #    self.ready = False
            #    self.done = False
            #    #self.curr_size = 0
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                #print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                self.ready = True
                self.curr_size = 0
                self.done = True
                if len(self.tile_ptrs_fifo) > 0:
                    self.curr_tile = self.tile_ptrs_fifo[0]
        #if self.level == "glb2global" and self.ready == False:
            #print("@@@@@@@@@@@@@@@@@@@@")
        if self.debug:
            print(self.level, " valid: ", self.valid, " ready: ", self.ready, " loading: ", self.loading, " done: ", self.done, " curr tile: ", self.curr_tile, " Done ", self.done, " valid processed: ", self.valid_processed, " Child ready ", self.child_ready, " , ", self.curr_size, "       " , self.tile_ptrs_fifo)

    def set_child_ready(self, token):
        self.child_ready = token

    def token(self):
        return self.curr_tile

    def get_size(self):
        return self.curr_size

    def check_tile(self, tile_id):
        if tile_id in self.tile_ptrs_ids:
            return True
        return False

    def compute_latency(self, tile):
        return 16

    def input_token(self):
        if self.downstream_token == "D":
            return True

    def input_token_(self, token):
        self.downstream_token = token



class memory_block():
    def __init__(self, name = "B", level = None, size = 1000*2, debug=False, bandwidth = 2, length = 1, mode = "all_upacked"):
        self.name = name
        self.level = level
        self.size = size
        self.bandwidth = bandwidth
        self.length = length
        self.tile_ptrs = []

        self.tile_ptrs_fifo = [] # virtual
        self.tile_ptrs_size = []
        
        self.mode = mode
        self.timestamp = None
        self.ready = True
        self.loading = False
        self.valid = False
        self.curr_tile = None
        self.curr_size = None
        self.old_tile = None
        self.debug = debug
        self.outputed = False
        self.done = False
        self.signalled = False
        self.downstream_token = None
        self.done_received = False
        self.valid_processed = False
        self.done_processed = False

    def out_done(self):
        return self.done

    def add_tile(self, tile_ptr, size):
        if tile_ptr == "D":
            self.tile_ptrs_fifo.append("D")
            self.tile_ptrs_size.append(0)
        if tile_ptr != "" and isinstance(tile_ptr, int):
            self.done = False
            self.tile_ptrs_fifo.append(tile_ptr)
            self.tile_ptrs_size.append(size)

    def set_downstream_token(self, token):
        self.downstream_token = token

    def return_token(self):
        if self.done:
            return "D"
        if self.loading:
            return "L"
        #if self.signalled:
        #    return self.curr_tile

    def return_if_loading(self):
        return self.loading

    def check_if_done(self, arr):
        if isinstance(arr, bool):
            self.done_received = arr
            return
        token = True
        if len(arr) > 0:
            for i in arr:
                if not i:
                    token = False
                    self.done_received = False
                    return
        self.done_received = True
        return

    def add_upstream(self, tilecoord, data, valid):
        if not valid:
            self.done_processed = False
        if valid and not self.done_processed and self.level == "mem2glb": 
            self.done_processed = True
            limits = len(data)-1
            size_levels = [0]*(limits//2)
            size_vals = len(data[len(data)-1])
            for lengths in range(len(size_levels)):
                size_levels[lengths] = len(data[lengths]) + len(data[lengths + limits//2])
            
            max_levels_size = max(max(size_levels), size_vals)
            self.tile_ptrs_size.append(max_levels_size)
            tile_hash, tile_glb = hash_tile(tilecoord)
            self.tile_ptrs_fifo.append(tile_hash)

        if valid and not self.done_processed and self.level == "glb2global":
            
            self.done_processed = True
            self.tile_ptrs_fifo.append(tilecoord)
            size = data
            self.tile_ptrs_size.append(data)

    
    def out_update(self, cyclenum):
        if self.level == "mem2glb":
            self.old_tile = None
            if self.downstream_token == "D":
                self.ready = True
            elif self.downstream_token == "L":
                return
            elif self.ready and len(self.tile_ptrs_fifo) > 0:
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size = self.tile_ptrs_size.pop(0)
                self.loading = True
                self.ready = False
                self.timestamp = cyclenum
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                self.loading = False
                self.done = True
            #elif self.done_received and not self.done_processed:
            #    #Reset cycle
            #    self.ready = True
            #    self.done_processed = True
            #    self.done = False
            #if self.done_processed == True and self.done_received == False:
            #    #Disable reset
            #    self.done_processed=False

        if self.level == "glb2global":
            self.old_tile = None
            if self.ready and len(self.tile_ptrs_fifo) > 0:
                self.curr_tile = self.tile_ptrs_fifo.pop(0)
                self.curr_size = self.tile_ptrs_size.pop(0)
                self.loading = True
                self.ready = False
                self.timestamp = cyclenum
                self.done = False
            elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
                self.ready = True
                self.done = True
            

    def update(self, cyclenum):
        self.signalled = False
        if self.ready and len(self.tile_ptrs_fifo) > 0:
            if self.curr_tile != "D":
                self.old_tile = self.curr_tile
            self.curr_tile = self.tile_ptrs_fifo.pop(0)
            self.curr_size = self.tile_ptrs_size.pop(0)
            if self.curr_tile == "D":
                self.timestamp = None
                self.ready = True
                self.loading = False
                self.valid = False
                #self.curr_tile = None
                self.outputed = False
                self.done = True
                self.signalled = True
                self.done_received = False
                self.done_processed = False
                
                if self.debug:
                    print(self.name, " valid: ", self.valid, " ready: ", self.ready, " loading: ", self.loading, " done: ", self.done, " downstream token: ", self.downstream_token, " Done received and processed ", self.done_received, " ", self.done_processed ," : current tile: ", self.curr_tile)
                return
            self.timestamp = cyclenum
            self.ready = False
            self.loading = True
            self.valid = False
            #self.done_processed = False
        elif self.loading and cyclenum < self.timestamp + self.compute_latency(self.curr_tile):
            pass
        elif self.loading and cyclenum > self.timestamp + self.compute_latency(self.curr_tile):
            self.loading = False
            self.valid = True
            self.outputed = False
        elif self.done_received and not self.done_processed:
            self.ready = True
            self.done_processed = True
            self.valid = False
        elif not self.ready and not self.loading and self.input_token():
            self.ready = True
            self.valid = False
        if self.done_processed == True and self.done_received == False:
            self.done_processed=False
        if self.debug:
            print(self.name, " valid: ", self.valid, " ready: ", self.ready, " loading: ", self.loading, " done: ", self.done, " downstream token: ", self.downstream_token, " Done received and processed ", self.done_received, " ", self.done_processed ," : current tile: ", self.curr_tile)

    def remove_tile(self, tile_ptr = None, tile_id = -1):
        if tile_ptr != None:
            index = self.tile_ptrs_ids.index(tile_ptr.getid())
            self.tile_ptrs_ids.pop(index)
            self.tile_ptrs_tile.pop(index)

    def pop_tile(self, lower_level_done):
        if lower_level_done:
            self.ready = True

    def pop_tile_after(self, cycle_num, time):
        if isinstance(self.curr_tile, int) and cycle_num > self.timestamp + self.compute_latency(self.curr_tile) + time:
            self.ready = True

    def token(self):
        return self.curr_tile

    def get_size(self):
        return self.curr_size

    def check_tile(self, tile_id):
        if tile_id in self.tile_ptrs_ids:
            return True
        return False

    def compute_latency(self, tile):
        if self.curr_tile == self.old_tile:
            return 0
        return 4

    def valid_tile(self):
        if self.valid and not self.outputed:
            return True

    def valid_tile_recieved(self):
        print(self.name, " returns ", self.curr_tile)
        self.outputed = True

    def if_stop(self):
        if self.done:
            return True
    
    def input_token(self):
        if self.downstream_token == "D":
            return True

    def input_token_(self, token):
        self.downstream_token = token


class channel():
    def __init__(size = 1000**3, bandwidth = 2, tile_ptrs=None, mode = "all_upacked"):
        self.size = size
        self.tile_ptrs_tile = [] #tiile_ptrs_glbtile
        
        self.tile_ptrs_ids = []

        self.tile_ptrs_fifo = [] # virtual
        self.mode = mode

        seld.timestamps = {}

    def add_tile(tile_ptr):
        for ptrs in tile_ptr:
            self.tile_ptrs_fifo.append(ptrs)

    def update(cyclenum):
        head = self.tile_ptrs_glbfifo[0]
        
        if head.getid() not in self.tile_ptrs_ids:
            if check_limit(head):
                if self.timestamps[heal.getid()] != None:
                    self.timestamps[head.getid()] = cyclenum
                if cyclenum > self.timestamps[head.getid()] + compute_latency(head):
                    self.tile_ptrs_tile.append(head_glb)
                    self.tile_ptrs_ids.append(head_glb.getid)
                    self.tile_ptrs_fifo.pop()

    def remove_tile(tile_ptr = None, tile_id = -1):
        if tile_ptr != None:
            index = self.tile_ptrs_ids.index(tile_ptr.getid())
            self.tile_ptrs_ids.pop(index)
            self.tile_ptrs_tile.pop(index)

    def check_tile(tile_id):
        if tile_id in self.tile_ptrs_ids:
            return True
        return False

    def compute_latency(tile):
        return 0


    def check_limit(tile):
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_tile:
                if val_levels == 0:
                    val_levels = key.return_tensor_indexes()
                elif val_levels != key.return_tensor_indexes():
                    assert False
                cumu_sum += key.return_vals_size()

            if val_levels != 0 and val_levels != tile.return_tensor_indexes():
                assert False
            cumu_sum += tile.return_vals_size()
            if cumu_sum > self.size[num_levels-2]:
                return False
            else:
                return True


    def check_tile_limits():
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_tile:
                if val_levels == 0:
                    val_levels = self.tile_ptrs_tile[key].return_tensor_indexes()
                elif val_levels != self.tile_ptrs_tile[key].return_tensor_indexes():
                    assert False
                cumu_sum += self.tile_ptrs_tile[key].return_vals_size()
            assert cumu_sum < self.size[num_levels-2]
