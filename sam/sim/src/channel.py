import numpy as np

class tile_ptr():
    def __init__(tileid = 0; levels = 0, levels_size = None, vals_size=None):
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


    def check_tile_limits():
        if self.mode == "all_unpacked":
            cumu_sum = 0
            val_levels = 0
            for key in self.tile_ptrs_tile:
                if val_levels = 0:
                    val_levels = self.tile_ptrs_tile[key].return_tensor_indexes()
                elif val_levels != self.tile_ptrs_tile[key].return_tensor_indexes():
                    assert False
                cumu_sum += self.tile_ptrs_tile[key].return_vals_size()
            assert cumu_sum < self.size[num_levels-2]
