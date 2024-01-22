module SplitFifo_17 (
  input logic clk,
  input logic clk_en,
  input logic [16:0] data_in,
  input logic end_fifo,
  input logic fifo_en,
  input logic ready1,
  input logic rst,
  input logic start_fifo,
  input logic valid0,
  output logic [16:0] data_out,
  output logic ready0,
  output logic valid1
);

logic empty;
logic empty_n;
logic ready_in;
logic valid_in;
logic [16:0] value;
assign empty = ~empty_n;
assign ready_in = ready1 && (~start_fifo);
assign ready0 = fifo_en ? empty || ready_in: clk_en;
assign valid_in = valid0 && (~end_fifo);
assign valid1 = fifo_en ? (~empty) || valid_in: clk_en;
assign data_out = (empty && fifo_en) ? data_in: value;

always_ff @(posedge clk, posedge rst) begin
  if (rst) begin
    value <= 17'h0;
  end
  else if (clk_en) begin
    if ((~fifo_en) || (valid0 && ready0 && (~(empty && ready1 && valid1)))) begin
      value <= data_in;
    end
  end
end

always_ff @(posedge clk, posedge rst) begin
  if (rst) begin
    empty_n <= 1'h0;
  end
  else if (clk_en) begin
    if (fifo_en) begin
      if (valid1 && ready1) begin
        if (~(valid0 && ready0)) begin
          empty_n <= 1'h0;
        end
      end
      else if (valid0 && ready0) begin
        empty_n <= 1'h1;
      end
    end
  end
end
endmodule   // SplitFifo_17

