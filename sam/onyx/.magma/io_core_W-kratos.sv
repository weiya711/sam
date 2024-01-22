module io_core (
  input logic clk,
  input logic clk_en,
  input logic f2io_1,
  input logic [16:0] f2io_17,
  input logic f2io_17_valid,
  input logic f2io_1_valid,
  input logic flush,
  input logic glb2io_1,
  input logic [16:0] glb2io_17,
  input logic glb2io_17_valid,
  input logic glb2io_1_valid,
  input logic io2f_17_ready,
  input logic io2f_1_ready,
  input logic io2glb_17_ready,
  input logic io2glb_1_ready,
  input logic rst_n,
  input logic tile_en,
  output logic f2io_17_ready,
  output logic f2io_1_ready,
  output logic glb2io_17_ready,
  output logic glb2io_1_ready,
  output logic io2f_1,
  output logic [16:0] io2f_17,
  output logic io2f_17_valid,
  output logic io2f_1_valid,
  output logic io2glb_1,
  output logic [16:0] io2glb_17,
  output logic io2glb_17_valid,
  output logic io2glb_1_valid
);

logic [0:0][16:0] f2io_2_io2glb_17_data_out;
logic f2io_2_io2glb_17_empty;
logic f2io_2_io2glb_17_full;
logic [0:0] f2io_2_io2glb_1_data_out;
logic f2io_2_io2glb_1_empty;
logic f2io_2_io2glb_1_full;
logic gclk;
logic glb2io_2_io2f_17_empty;
logic glb2io_2_io2f_17_full;
logic glb2io_2_io2f_1_empty;
logic glb2io_2_io2f_1_full;
assign gclk = clk & tile_en;
assign io2glb_1 = f2io_2_io2glb_1_data_out;
assign f2io_1_ready = ~f2io_2_io2glb_1_full;
assign io2glb_1_valid = ~f2io_2_io2glb_1_empty;
assign glb2io_1_ready = ~glb2io_2_io2f_1_full;
assign io2f_1_valid = ~glb2io_2_io2f_1_empty;
assign io2glb_17 = f2io_2_io2glb_17_data_out[0][16:0];
assign f2io_17_ready = ~f2io_2_io2glb_17_full;
assign io2glb_17_valid = ~f2io_2_io2glb_17_empty;
assign glb2io_17_ready = ~glb2io_2_io2f_17_full;
assign io2f_17_valid = ~glb2io_2_io2f_17_empty;
reg_fifo_depth_2_w_1_afd_1_iocore_nof f2io_2_io2glb_1 (
  .clk(clk),
  .clk_en(clk_en),
  .data_in(f2io_1),
  .flush(flush),
  .pop(io2glb_1_ready),
  .push(f2io_1_valid),
  .rst_n(rst_n),
  .data_out(f2io_2_io2glb_1_data_out),
  .empty(f2io_2_io2glb_1_empty),
  .full(f2io_2_io2glb_1_full)
);

reg_fifo_depth_2_w_1_afd_1_iocore_nof glb2io_2_io2f_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(glb2io_1),
  .flush(flush),
  .pop(io2f_1_ready),
  .push(glb2io_1_valid),
  .rst_n(rst_n),
  .data_out(io2f_1),
  .empty(glb2io_2_io2f_1_empty),
  .full(glb2io_2_io2f_1_full)
);

reg_fifo_depth_2_w_17_afd_1_iocore_nof f2io_2_io2glb_17 (
  .clk(clk),
  .clk_en(clk_en),
  .data_in(f2io_17),
  .flush(flush),
  .pop(io2glb_17_ready),
  .push(f2io_17_valid),
  .rst_n(rst_n),
  .data_out(f2io_2_io2glb_17_data_out),
  .empty(f2io_2_io2glb_17_empty),
  .full(f2io_2_io2glb_17_full)
);

reg_fifo_depth_2_w_17_afd_1_iocore_nof glb2io_2_io2f_17 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(glb2io_17),
  .flush(flush),
  .pop(io2f_17_ready),
  .push(glb2io_17_valid),
  .rst_n(rst_n),
  .data_out(io2f_17),
  .empty(glb2io_2_io2f_17_empty),
  .full(glb2io_2_io2f_17_full)
);

endmodule   // io_core

module io_core_W (
  input logic clk,
  input logic clk_en,
  input logic f2io_1,
  input logic [16:0] f2io_17,
  input logic f2io_17_valid,
  input logic f2io_1_valid,
  input logic flush,
  input logic glb2io_1,
  input logic [16:0] glb2io_17,
  input logic glb2io_17_valid,
  input logic glb2io_1_valid,
  input logic io2f_17_ready,
  input logic io2f_1_ready,
  input logic io2glb_17_ready,
  input logic io2glb_1_ready,
  input logic rst_n,
  input logic tile_en,
  output logic f2io_17_ready,
  output logic f2io_1_ready,
  output logic glb2io_17_ready,
  output logic glb2io_1_ready,
  output logic io2f_1,
  output logic [16:0] io2f_17,
  output logic io2f_17_valid,
  output logic io2f_1_valid,
  output logic io2glb_1,
  output logic [16:0] io2glb_17,
  output logic io2glb_17_valid,
  output logic io2glb_1_valid
);

io_core io_core (
  .clk(clk),
  .clk_en(clk_en),
  .f2io_1(f2io_1),
  .f2io_17(f2io_17),
  .f2io_17_valid(f2io_17_valid),
  .f2io_1_valid(f2io_1_valid),
  .flush(flush),
  .glb2io_1(glb2io_1),
  .glb2io_17(glb2io_17),
  .glb2io_17_valid(glb2io_17_valid),
  .glb2io_1_valid(glb2io_1_valid),
  .io2f_17_ready(io2f_17_ready),
  .io2f_1_ready(io2f_1_ready),
  .io2glb_17_ready(io2glb_17_ready),
  .io2glb_1_ready(io2glb_1_ready),
  .rst_n(rst_n),
  .tile_en(tile_en),
  .f2io_17_ready(f2io_17_ready),
  .f2io_1_ready(f2io_1_ready),
  .glb2io_17_ready(glb2io_17_ready),
  .glb2io_1_ready(glb2io_1_ready),
  .io2f_1(io2f_1),
  .io2f_17(io2f_17),
  .io2f_17_valid(io2f_17_valid),
  .io2f_1_valid(io2f_1_valid),
  .io2glb_1(io2glb_1),
  .io2glb_17(io2glb_17),
  .io2glb_17_valid(io2glb_17_valid),
  .io2glb_1_valid(io2glb_1_valid)
);

endmodule   // io_core_W

module reg_fifo_depth_2_w_17_afd_1_iocore_nof (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] data_in,
  input logic flush,
  input logic pop,
  input logic push,
  input logic rst_n,
  output logic almost_full,
  output logic [0:0] [16:0] data_out,
  output logic empty,
  output logic full,
  output logic valid
);

logic [1:0] num_items;
logic passthru;
logic rd_ptr;
logic read;
logic [1:0][0:0][16:0] reg_array;
logic wr_ptr;
logic write;
assign full = num_items == 2'h2;
assign almost_full = num_items >= 2'h1;
assign empty = num_items == 2'h0;
assign read = pop & (~passthru) & (~empty);
assign passthru = 1'h0;
assign write = push & (~passthru) & (~full);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_items <= 2'h0;
  end
  else if (flush) begin
    num_items <= 2'h0;
  end
  else if (clk_en) begin
    if (write & (~read)) begin
      num_items <= num_items + 2'h1;
    end
    else if ((~write) & read) begin
      num_items <= num_items - 2'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reg_array <= 34'h0;
  end
  else if (flush) begin
    reg_array <= 34'h0;
  end
  else if (clk_en) begin
    if (write) begin
      reg_array[wr_ptr] <= data_in;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr <= 1'h0;
  end
  else if (flush) begin
    wr_ptr <= 1'h0;
  end
  else if (clk_en) begin
    if (write) begin
      if (wr_ptr == 1'h1) begin
        wr_ptr <= 1'h0;
      end
      else wr_ptr <= wr_ptr + 1'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rd_ptr <= 1'h0;
  end
  else if (flush) begin
    rd_ptr <= 1'h0;
  end
  else if (clk_en) begin
    if (read) begin
      rd_ptr <= rd_ptr + 1'h1;
    end
  end
end
always_comb begin
  if (passthru) begin
    data_out = data_in;
  end
  else data_out = reg_array[rd_ptr];
end
always_comb begin
  valid = (~empty) | passthru;
end
endmodule   // reg_fifo_depth_2_w_17_afd_1_iocore_nof

module reg_fifo_depth_2_w_1_afd_1_iocore_nof (
  input logic clk,
  input logic clk_en,
  input logic [0:0] data_in,
  input logic flush,
  input logic pop,
  input logic push,
  input logic rst_n,
  output logic almost_full,
  output logic [0:0] data_out,
  output logic empty,
  output logic full,
  output logic valid
);

logic [1:0] num_items;
logic passthru;
logic rd_ptr;
logic read;
logic [1:0][0:0] reg_array;
logic wr_ptr;
logic write;
assign full = num_items == 2'h2;
assign almost_full = num_items >= 2'h1;
assign empty = num_items == 2'h0;
assign read = pop & (~passthru) & (~empty);
assign passthru = 1'h0;
assign write = push & (~passthru) & (~full);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_items <= 2'h0;
  end
  else if (flush) begin
    num_items <= 2'h0;
  end
  else if (clk_en) begin
    if (write & (~read)) begin
      num_items <= num_items + 2'h1;
    end
    else if ((~write) & read) begin
      num_items <= num_items - 2'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reg_array <= 2'h0;
  end
  else if (flush) begin
    reg_array <= 2'h0;
  end
  else if (clk_en) begin
    if (write) begin
      reg_array[wr_ptr] <= data_in;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr <= 1'h0;
  end
  else if (flush) begin
    wr_ptr <= 1'h0;
  end
  else if (clk_en) begin
    if (write) begin
      if (wr_ptr == 1'h1) begin
        wr_ptr <= 1'h0;
      end
      else wr_ptr <= wr_ptr + 1'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rd_ptr <= 1'h0;
  end
  else if (flush) begin
    rd_ptr <= 1'h0;
  end
  else if (clk_en) begin
    if (read) begin
      rd_ptr <= rd_ptr + 1'h1;
    end
  end
end
always_comb begin
  if (passthru) begin
    data_out = data_in;
  end
  else data_out = reg_array[rd_ptr];
end
always_comb begin
  valid = (~empty) | passthru;
end
endmodule   // reg_fifo_depth_2_w_1_afd_1_iocore_nof

