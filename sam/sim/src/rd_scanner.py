from .base import *


#################
# Read Scanners
#################


class CrdRdScan(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = 'S0'
        self.curr_crd = 'S0'

        self.in_ref = []

    def set_in_ref(self, in_ref):
        if in_ref != '':
            self.in_ref.append(in_ref)

    def out_ref(self):
        return self.curr_ref

    def out_crd(self):
        return self.curr_crd


# TODO: figure out how uncompressed read scans work with 'N' tokens
class UncompressCrdRdScan(CrdRdScan):
    def __init__(self, dim=0, **kwargs):
        super().__init__(**kwargs)

        self.start_addr = 0
        self.stop_addr = dim

        self.curr_in_ref = 0

        self.meta_dim = dim

    def update(self):

        # run out of coordinates, move to next input reference
        if self.curr_crd == '' or self.curr_crd == 'D':
            self.curr_crd = ''
            self.curr_ref = ''
        elif is_stkn(self.curr_crd):
            self.curr_in_ref = self.in_ref.pop(0)
            if self.curr_in_ref == 'D':
                self.curr_crd = 'D'
                self.curr_ref = 'D'
                self.done = True
                return
            else:
                self.curr_crd = 0
                self.curr_ref = self.curr_crd + (self.curr_in_ref * self.meta_dim)
        elif self.curr_crd >= self.meta_dim - 1:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_crd = stkn
            self.curr_ref = stkn
        else:
            self.curr_crd += 1
            self.curr_ref = self.curr_crd + self.curr_in_ref * self.meta_dim

        if self.debug:
            print("DEBUG: U RD SCAN: \t "
                  "Curr crd:", self.curr_crd, "\t curr ref:", self.curr_ref)


def last_stkn(skiplist):
    max_ref = None
    for i, item in enumerate(skiplist):
        if is_stkn(item):
            max_ref = i
    return max_ref


class CompressedCrdRdScan(CrdRdScan):
    def __init__(self, crd_arr=[], seg_arr=[], skip=True, **kwargs):
        super().__init__(**kwargs)

        # Used for skip list
        self.skip = skip
        self.in_crd_skip = []
        self.curr_skip = None
        self.skip_processed = True
        self.prev_crd = 0
        # [Olivia]: I think this is needed to make sure we are looking at the correct fiber
        self.skip_stkn_cnt = 0
        self.out_stkn_cnt = 0

        self.crd_arr = crd_arr
        self.seg_arr = seg_arr

        self.start_addr = 0
        self.stop_addr = 0

        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = ''
        self.curr_crd = ''
        self.emit_fiber_stkn = False

        self.meta_clen = len(crd_arr)
        self.meta_slen = len(seg_arr)

        self.begin = True

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

    def update(self):
        # Process skip token first and save
        if len(self.in_crd_skip) > 0 and self.skip_processed:
            self.curr_skip = self.in_crd_skip.pop(0)
            if self.skip_stkn_cnt == self.out_stkn_cnt and isinstance(self.curr_skip, int) \
                    and self.curr_skip < self.prev_crd:
                # ignore the skip if it's too small
                self.skip_processed = True
            elif self.skip_stkn_cnt < self.out_stkn_cnt:
                # ignore the skip if it's a fiber behind
                self.skip_processed = True
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
                self._emit_stkn_code()

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

                            # Else emit smallest coordinate larger than the one provided by skip
                            else:
                                larger = [i for i in curr_range if i > self.curr_skip]
                                if not larger:
                                    self._emit_stkn_code()
                                else:
                                    val_larger = min(larger)
                                    self.curr_addr = curr_range.index(val_larger) + self.start_addr
                                    self._set_curr()

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

                    # Else emit smallest coordinate larger than the one provided by skip
                    else:
                        larger = [i for i in curr_range if i > self.curr_skip]
                        if not larger:
                            self._emit_stkn_code()
                        else:
                            val_larger = min(larger)
                            self.curr_addr = curr_range.index(val_larger) + self.start_addr
                            self._set_curr()

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
        assert is_valid_crd(in_crd)
        if in_crd != '':
            if is_stkn(in_crd):
                idx = last_stkn(self.in_crd_skip)
                if idx is not None:
                    # Flush coordinates
                    self.in_crd_skip = self.in_crd_skip[:idx + 1]
            self.in_crd_skip.append(in_crd)


# ---------------- BV --------------#

class BVRdScanSuper(Primitive, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.curr_ref = 'S0'
        self.curr_bv = 'S0'

        self.in_ref = []

    def set_in_ref(self, in_ref):
        if in_ref != '':
            self.in_ref.append(in_ref)

    def out_ref(self):
        return self.curr_ref

    def out_bv(self):
        return self.curr_bv


class BVRdScan(BVRdScanSuper):
    def __init__(self, bv_arr=None, dim=4, nbits=4, **kwargs):
        super().__init__(**kwargs)

        self.bv_arr = bv_arr if bv_arr is not None else [2 ** nbits - 1] * dim

        self.curr_addr = 0

        self.end_fiber = False
        self.curr_ref = ''
        self.curr_bv = ''
        self.emit_fiber_stkn = False
        self.begin = True

        self.meta_blen = len(bv_arr)
        self.meta_nbits = nbits
        self.meta_dim = dim

    def _get_bv_ref(self, addr):
        assert isinstance(addr, int), "Addresses must be integers"
        if addr <= 0:
            return 0
        bits = sum(map(popcount, self.bv_arr[:addr]))
        return bits

    def update(self):
        curr_in_ref = None
        if self.curr_bv == 'D' or self.curr_ref == 'D' or self.done:
            self.curr_addr = 0
            self.curr_bv = ''
            self.curr_ref = ''
        elif len(self.in_ref) > 0 and self.emit_fiber_stkn:
            next_in = self.in_ref[0]
            if is_stkn(next_in):
                self.in_ref.pop(0)
                stkn = increment_stkn(next_in)
            else:
                stkn = 'S0'
            self.curr_bv = stkn
            self.curr_ref = stkn

            self.curr_addr = 0
            self.emit_fiber_stkn = False
        # There exists another input reference at the segment and
        # either at the start of computation or end of fiber
        elif len(self.in_ref) > 0 and (self.end_fiber or self.begin):
            self.end_fiber = False
            self.begin = False

            curr_in_ref = self.in_ref.pop(0)

            if isinstance(curr_in_ref, int) and curr_in_ref + 1 > self.meta_blen:
                raise Exception('Not enough elements in bv array(' + str(self.meta_blen) + ')')

            # See 'N' 0-token which immediately emits a stop token and ends the fiber
            # TODO: need to figure out how this will work with bitvectors
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
                self.curr_bv = stkn
                self.curr_ref = stkn
                self.curr_addr = 0

            elif is_stkn(curr_in_ref) or curr_in_ref == 'D':
                self.curr_addr = 0
                self.curr_bv = curr_in_ref
                self.curr_ref = curr_in_ref
                self.end_fiber = True
                if curr_in_ref == 'D':
                    self.done = True
            else:
                self.curr_addr = curr_in_ref
                # End of fiber, get next input reference
                self.end_fiber = True

                self.emit_fiber_stkn = True
                self.curr_bv = self.bv_arr[self.curr_addr]
                self.curr_ref = self._get_bv_ref(self.curr_addr)

        else:   # elif self.curr_bv is not None and self.curr_ref is not None:
            # Default stall (when done)
            self.curr_ref = ''
            self.curr_bv = ''

        if self.debug:
            print("DEBUG: C RD SCAN: \t "
                  "Curr bv:", self.curr_bv, "\t curr ref:", self.curr_ref, "\t curr addr:", self.curr_addr,
                  "\t end fiber:", self.end_fiber, "\t curr input:", curr_in_ref)
