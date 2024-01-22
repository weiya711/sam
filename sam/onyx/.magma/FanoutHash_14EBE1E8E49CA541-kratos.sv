module FanoutHash_14EBE1E8E49CA541 (
  input logic E0,
  input logic E1,
  input logic E2,
  input logic I0,
  input logic I1,
  input logic I2,
  input logic [31:0] S0,
  input logic [31:0] S1,
  input logic [31:0] S2,
  output logic O
);

logic sel0;
logic sel1;
logic sel2;
assign sel0 = (~E0) | (~S0[20]) | I0;
assign sel1 = (~E1) | (~S1[20]) | I1;
assign sel2 = (~E2) | (~S2[20]) | I2;
assign O = sel0 & sel1 & sel2;
endmodule   // FanoutHash_14EBE1E8E49CA541

