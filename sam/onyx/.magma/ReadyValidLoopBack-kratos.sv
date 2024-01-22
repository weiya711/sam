module ReadyValidLoopBack (
  input logic ready_in,
  input logic valid_in,
  output logic valid_out
);

assign valid_out = ready_in & valid_in;
endmodule   // ReadyValidLoopBack

