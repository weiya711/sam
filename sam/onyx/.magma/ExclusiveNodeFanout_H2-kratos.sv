module ExclusiveNodeFanout_H2 (
  input logic [1:0] I,
  input logic [1:0] S,
  output logic O
);

assign O = (I[0] & S[0]) | (I[1] & S[1]);
endmodule   // ExclusiveNodeFanout_H2

