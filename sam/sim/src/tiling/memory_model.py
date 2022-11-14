import numpy as np

class tile_ptr():
    def __init__(tileid = 0, levels = 0, levels_size = None, vals_size=None):
        self.tileid = tileid
        self.levels = levels
        if self.levels > 0:
            for key in levels_size:
                if levels_size[key] != None:
                    self.levels_size = levels_size[key]
            self.vals_size = max(vals_size, max(levels_size))

    def return_tensor_indexes():
        return self.levels

    def return_vals_size():
        return self.vals_size


class memory_model():
    def __init__(num_levels = 3, size = [float('inf'), 1000**3, 1000**2], lengths=[float('inf'), 10, 2], tile_ptrs=None, mode = "all_upacked"):
        self.num_levels = num_levels
        self.size = size
        self.lengths = lengths
        #self.tile_ptrs_arr = tile_ptrs_arr
        self.tile_ptrs_glbtile = [] #tiile_ptrs_glbtile
        self.tile_ptrs_memtile = [] #tile_ptrs_memtile
        
        self.tile_ptrs_glbids = []
        self.tile_ptrs_memids = []

        self.tile_ptrs_glbfifo = [] # virtual
        self.tile_ptrs_memfifo = [] # virtual
        self.mode = mode

        seld.glbtimestamps = {}
        seld.memtimestamps = {}
    #def check_backpressure_glb():
    #def check_backpressure_memtile():

    def add_tile_to_glb(tile_ptr):
        for ptrs in tile_ptr:
            self.tile_ptrs_glbfifo.append(ptrs)

    def add_tile_to_lowest_level(tile_ptr):
        for ptrs in tile_ptr:
            self.tile_ptrs_memfifo.append(ptrs)

    def update(cyclenum):
        head_glb = self.tile_ptrs_glbfifo[0]
        head_memtile = self.tile_ptrs_memfifo[0]
        
        if head_glb.getid() not in self.tile_ptrs_glbids:
            if check_glb_limit(head_glb):
                if self.glbtimestamps[heal_glb.getid()] != None:
                    self.glbtimestamps[head_glb.getid()] = cyclenum
                if cyclenum > self.glbtimestamps[head_glb.getid()] + compute_latency_glb(head_glb):
                    self.tile_ptrs_glbtile.append(head_glb)
                    self.tile_ptrs_glbids.append(head_glb.getid)
                    self.tile_ptrs_glbfifo.pop()

        if head_memtile.getid() not in self.tile_ptrs_memids:
            if check_mem_limit(head_memtile):
                if self.memtimestamps[head_memtile.getid()] != None:
                    self.memtimestamps[head_glb.getid()] = cyclenum
                if cyclenum > self.memtimestamps[head_glb.getid()] + compute_latency_mem(head_glb):
                    self.tile_ptrs_memtile.append(head_memtile)
                    self.tile_ptrs_memids.append(head_memtile.getid)
                    self.tile_ptrs_memfifo.pop()

    def remove_GLB_tile(tile_ptr = None, tile_id = -1):
        if tile_ptr != None:
            index = self.tile_ptrs_glbids.index(tile_ptr.getid())
            self.tile_ptrs_glbids.pop(index)
            self.tile_ptrs_glbtile.pop(index)

    def remove_mem_tile(tile_ptr = None, tile_id = -1):
        if tile_ptr != None:
            index = self.tile_ptrs_memtile.index(tile_ptr.getid())
            self.tile_ptrs_memids.pop(index)
            self.tile_ptrs_memtile.pop(index)

    
    def check_mem_tile(tile_id):
        if tile_id in self.tile_ptrs_memids:
            return True
        return False

    def check_glb_tile(tile_id):
        if tile_id in self.tile_ptrs_memids:
            return True
        return False


    def check_glb_limit(tile):
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_glbtile:
                if val_levels = 0:
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


    def check_mem_limit(tile):
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_memtile:
                if val_levels = 0:
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



    def check_glb_tile_limits():
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_memtile:
                if val_levels = 0:
                    val_levels = self.tile_ptrs_memtile[key].return_tensor_indexes()
                elif val_levels != self.tile_ptrs_memtile[key].return_tensor_indexes():
                    assert False
                cumu_sum += self.tile_ptrs_memtile[key].return_vals_size()
            assert cumu_sum < self.size[num_levels-2]

    def check_mem_tile_limits():
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_memtile:
                if val_levels = 0:
                    val_levels = self.tile_ptrs_memtile[key].return_tensor_indexes()
                elif val_levels != self.tile_ptrs_memtile[key].return_tensor_indexes():
                    assert False
            for key in self.tile_ptrs_memtile:
                cumu_sum += self.tile_ptrs_memtile[key].return_vals_size()
            assert cumu_sum < self.size[num_levels-1]
