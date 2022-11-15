from .base import *


class Array(Primitive):
    def __init__(self, init_arr=None, size=1024, fill=0, depth=1, **kwargs):
        super().__init__(**kwargs)

        self.fill = fill
        if init_arr is None:
            self.size = size
            self.arr = [self.fill] * self.size
        else:
            assert (isinstance(init_arr, list))
            self.arr = init_arr
            self.size = len(init_arr)
        self.load_addrs = []
        self.store_vals = []
        self.load_en = False
        self.store_en = False

        if self.backpressure_en:
            self.backpressure = []
            self.data_ready = True
            self.branch = []
            self.depth = depth

        if self.get_stats:
            self.valid_loads = 0
            self.address_seen = []
            self.load_addr_size = 0
            self.store_vals_size = 0

        self.curr_load = ''

    def add_child(self, child, branch=""):
        if self.backpressure_en:
            if child is not None:
                self.backpressure.append(child)
                self.branch.append(branch)

    def check_backpressure(self):
        if self.backpressure_en:
            j = 0
            for i in self.backpressure:
                if not i.fifo_available(self.branch[j]):
                    return False
                j += 1
        return True

    def update(self):
        self.update_done()
        if self.backpressure_en:
            self.data_ready = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_ready = True
            if (len(self.load_addrs) > 0 or len(self.store_vals) > 0):
                self.block_start = False

            if self.load_en and len(self.load_addrs) > 0:
                if self.get_stats:
                    self.load_addr_size = max(self.load_addr_size, len(self.load_addrs))
                self.curr_load = self.load(self.load_addrs.pop(0))
                self.load_en = False
            else:
                self.curr_load = ''

            if self.store_en and len(self.store_vals) > 0:
                if self.get_stats:
                    self.store_vals_size = max(self.store_vals_size, len(self.store_vals))
                store_tup = self.store_vals.pop(0)
                self.store(store_tup[0], store_tup[1])
                self.store_en = False

    def fifo_available(self, br=""):
        if self.backpressure_en:
            if len(self.load_addrs) > 1:
                return False
        return True

    def set_load(self, addr):
        if addr != '' and addr is not None:
            self.load_en = True
            self.load_addrs.append(addr)
        else:
            self.load_en = False

    def set_store(self, addr, vals):
        if addr != '' and vals != '' and addr is not None and vals is not None:
            self.store_en = True
            self.store_vals.append((addr, vals))
        else:
            self.store_en = False

    def get_arr(self):
        return self.arr

    def out_load(self):
        return self.curr_load

    def out_val(self):
        return self.curr_load

    def load(self, addr):
        # Special handling of loads of stop tokens
        if is_stkn(addr):
            val = addr
        # Special handling of loads of 'N' tokens
        elif is_0tkn(addr):
            val = 0
        elif addr == 'D':
            self.done = True
            val = 'D'
        elif addr >= self.size:
            raise Exception("Address (" + str(addr) + ") is out of array size (" +
                            str(self.size) + ") bounds, please resize")
        else:
            if self.get_stats:
                if addr not in self.address_seen:
                    self.address_seen.append(addr)
                self.valid_loads += 1

            val = self.arr[addr]

        if self.debug:
            print("DEBUG: ARRAY LD:", "\t Addr:", addr, "\t Val:", val)

        return val

    def store(self, addr, val):
        # Special handling of stores of stop tokens
        if is_stkn(addr) or is_stkn(val):
            return
        # Special handling of stores of 'N' tokens
        elif is_0tkn(addr) or is_0tkn(val):
            return
        elif addr == 'D' or val == 'D':
            self.done = True
            return
        elif addr >= self.size:
            self.resize(addr * 2)
            self.arr[addr] = val
            # raise Exception("Address (" + str(addr) + ") is out of array size (" +
            #                 str(self.size) + ") bounds, please resize")
        else:
            self.arr[addr] = val

        if self.debug:
            print("DEBUG: ARRAY ST:", "\t Addr:", addr, "\t Val:", val)

    def reinit(self, init_arr):
        self.arr = init_arr

    def resize(self, size):
        if self.size > size:
            self.arr = self.arr[:size]
        else:
            self.arr = self.arr + [self.fill] * (size - self.size)
        self.size = size

    def clear(self, fill=None):
        if fill is None:
            fill = self.fill
        self.arr = [fill for _ in range(self.size)]

    def return_statistics(self):
        if self.get_stats:
            stats_dict = {"array_size": self.size, "fifo_addr": self.load_addr_size, "fifo_vals": self.store_vals_size,
                          "elements_touched": len(self.address_seen), "valid_loads": self.valid_loads}
            stats_dict.update(super().return_statistics())
        else:
            stats_dict = {}
        return stats_dict

    def print_fifos(self):
        print("Arrayvals fifo addresses: ", self.load_addr_size)
        print("Arrayvals fifo vals: ", self.store_vals_size)
