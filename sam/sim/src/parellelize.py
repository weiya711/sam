from .base import *


class Parellelize(Primitive):
    def __init__(self, parellelize_factor=16, depth=4, **kwargs):
        super().__init__(**kwargs)

        self.parellelize_factor = parellelize_factor
        self.in_token = []
        self.output_tokens = []
        self.output_ready = False
        self.done_received = False
        if self.backpressure_en:
            self.ready_backpressure = True
            self.data_valid = True
            self.depth = depth
            self.fifo_avail = True

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
            if len(self.in_token) > self.depth:
                self.fifo_avail = False
            else:
                self.fifo_avail = True
    
    def add_tokens(self, token):
        if token != None and token != "":
            self.in_token.append(token)


    def return_tokens(self):
        if self.output_ready:
            return self.output_tokens
        return [""]*self.parellelize_factor


    def update(self):
        self.update_done()
        self.update_ready()
        if len(self.in_token) > 0:
            self.block_start = False
        if self.backpressure_en:
            self.data_valid = False
        if (self.backpressure_en and self.check_backpressure()) or not self.backpressure_en:
            if self.backpressure_en:
                self.data_valid = True
            
            if self.done:
                self.output_ready = False
                return
            
            if len(self.output_tokens) == self.parellelize_factor:
                self.output_tokens = []
                if self.done_received:
                    self.done = True

            if len(self.in_token) > 0:
                token = self.in_token.pop(0)
                #if not is_stkn(token):
                self.output_tokens.append(token)
                if token == "D":
                    self.done_received = True
            else:
                token = ""

            if len(self.output_tokens) == self.parellelize_factor or self.done_received:
                self.output_ready = True
                if self.done_received:
                    while len(self.output_tokens) < self.parellelize_factor:
                        self.output_tokens.append('D')
            else:
                self.output_ready = False
        if self.debug:
            print("DEBUG: Parellize Block input_stream", self.in_token, " current token ", token, "output_tokens", self.output_tokens, "length", self.parellelize_factor)
