module PE_inner (
  input logic [31:0] CONFIG_SPACE_0,
  input logic [31:0] CONFIG_SPACE_1,
  input logic [31:0] CONFIG_SPACE_2,
  input logic [27:0] CONFIG_SPACE_3,
  input logic [0:0] [16:0] PE_input_width_17_num_0,
  input logic PE_input_width_17_num_0_dense,
  input logic PE_input_width_17_num_0_valid,
  input logic [0:0] [16:0] PE_input_width_17_num_1,
  input logic PE_input_width_17_num_1_dense,
  input logic PE_input_width_17_num_1_valid,
  input logic [0:0] [16:0] PE_input_width_17_num_2,
  input logic PE_input_width_17_num_2_dense,
  input logic PE_input_width_17_num_2_valid,
  input logic [0:0] [16:0] PE_input_width_17_num_3,
  input logic PE_input_width_17_num_3_valid,
  input logic PE_input_width_1_num_0,
  input logic PE_input_width_1_num_1,
  input logic PE_input_width_1_num_2,
  input logic PE_output_width_17_num_0_ready,
  input logic PE_output_width_17_num_1_dense,
  input logic PE_output_width_17_num_1_ready,
  input logic PE_output_width_17_num_2_ready,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mode,
  input logic rst_n,
  input logic tile_en,
  output logic PE_input_width_17_num_0_ready,
  output logic PE_input_width_17_num_1_ready,
  output logic PE_input_width_17_num_2_ready,
  output logic PE_input_width_17_num_3_ready,
  output logic [0:0] [16:0] PE_output_width_17_num_0,
  output logic PE_output_width_17_num_0_valid,
  output logic [0:0] [16:0] PE_output_width_17_num_1,
  output logic PE_output_width_17_num_1_valid,
  output logic [0:0] [16:0] PE_output_width_17_num_2,
  output logic PE_output_width_17_num_2_valid,
  output logic PE_output_width_1_num_0,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O2,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O3,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O4
);

logic [123:0] CONFIG_SPACE;
logic gclk;
logic [0:0][16:0] input_width_17_num_0_fifo_out;
logic input_width_17_num_0_fifo_out_ready;
logic input_width_17_num_0_fifo_out_valid;
logic input_width_17_num_0_input_fifo_empty;
logic input_width_17_num_0_input_fifo_full;
logic [0:0][16:0] input_width_17_num_1_fifo_out;
logic input_width_17_num_1_fifo_out_ready;
logic input_width_17_num_1_fifo_out_valid;
logic input_width_17_num_1_input_fifo_empty;
logic input_width_17_num_1_input_fifo_full;
logic [0:0][16:0] input_width_17_num_2_fifo_out;
logic input_width_17_num_2_fifo_out_ready;
logic input_width_17_num_2_fifo_out_valid;
logic input_width_17_num_2_input_fifo_empty;
logic input_width_17_num_2_input_fifo_full;
logic [0:0][16:0] input_width_17_num_3_fifo_out;
logic input_width_17_num_3_fifo_out_ready;
logic input_width_17_num_3_fifo_out_valid;
logic input_width_17_num_3_input_fifo_empty;
logic input_width_17_num_3_input_fifo_full;
logic [15:0] mem_ctrl_RepeatSignalGenerator_flat_RepeatSignalGenerator_inst_stop_lvl;
logic mem_ctrl_RepeatSignalGenerator_flat_RepeatSignalGenerator_inst_tile_en;
logic mem_ctrl_RepeatSignalGenerator_flat_base_data_in_ready_f_;
logic mem_ctrl_RepeatSignalGenerator_flat_clk;
logic [0:0][16:0] mem_ctrl_RepeatSignalGenerator_flat_repsig_data_out_f_;
logic mem_ctrl_RepeatSignalGenerator_flat_repsig_data_out_valid_f_;
logic mem_ctrl_Repeat_flat_Repeat_inst_root;
logic mem_ctrl_Repeat_flat_Repeat_inst_spacc_mode;
logic [15:0] mem_ctrl_Repeat_flat_Repeat_inst_stop_lvl;
logic mem_ctrl_Repeat_flat_Repeat_inst_tile_en;
logic mem_ctrl_Repeat_flat_clk;
logic mem_ctrl_Repeat_flat_proc_data_in_ready_f_;
logic [0:0][16:0] mem_ctrl_Repeat_flat_ref_data_out_f_;
logic mem_ctrl_Repeat_flat_ref_data_out_valid_f_;
logic mem_ctrl_Repeat_flat_repsig_data_in_ready_f_;
logic mem_ctrl_crddrop_flat_clk;
logic mem_ctrl_crddrop_flat_cmrg_coord_in_0_ready_f_;
logic mem_ctrl_crddrop_flat_cmrg_coord_in_1_ready_f_;
logic [0:0][16:0] mem_ctrl_crddrop_flat_cmrg_coord_out_0_f_;
logic mem_ctrl_crddrop_flat_cmrg_coord_out_0_valid_f_;
logic [0:0][16:0] mem_ctrl_crddrop_flat_cmrg_coord_out_1_f_;
logic mem_ctrl_crddrop_flat_cmrg_coord_out_1_valid_f_;
logic mem_ctrl_crddrop_flat_crddrop_inst_cmrg_enable;
logic mem_ctrl_crddrop_flat_crddrop_inst_cmrg_mode;
logic [15:0] mem_ctrl_crddrop_flat_crddrop_inst_cmrg_stop_lvl;
logic mem_ctrl_crddrop_flat_crddrop_inst_tile_en;
logic mem_ctrl_crdhold_flat_clk;
logic mem_ctrl_crdhold_flat_cmrg_coord_in_0_ready_f_;
logic mem_ctrl_crdhold_flat_cmrg_coord_in_1_ready_f_;
logic [0:0][16:0] mem_ctrl_crdhold_flat_cmrg_coord_out_0_f_;
logic mem_ctrl_crdhold_flat_cmrg_coord_out_0_valid_f_;
logic [0:0][16:0] mem_ctrl_crdhold_flat_cmrg_coord_out_1_f_;
logic mem_ctrl_crdhold_flat_cmrg_coord_out_1_valid_f_;
logic mem_ctrl_crdhold_flat_crdhold_inst_cmrg_enable;
logic [15:0] mem_ctrl_crdhold_flat_crdhold_inst_cmrg_stop_lvl;
logic mem_ctrl_crdhold_flat_crdhold_inst_tile_en;
logic mem_ctrl_intersect_unit_flat_clk;
logic mem_ctrl_intersect_unit_flat_coord_in_0_ready_f_;
logic mem_ctrl_intersect_unit_flat_coord_in_1_ready_f_;
logic [0:0][16:0] mem_ctrl_intersect_unit_flat_coord_out_f_;
logic mem_ctrl_intersect_unit_flat_coord_out_valid_f_;
logic mem_ctrl_intersect_unit_flat_intersect_unit_inst_joiner_op;
logic mem_ctrl_intersect_unit_flat_intersect_unit_inst_tile_en;
logic mem_ctrl_intersect_unit_flat_intersect_unit_inst_vector_reduce_mode;
logic mem_ctrl_intersect_unit_flat_pos_in_0_ready_f_;
logic mem_ctrl_intersect_unit_flat_pos_in_1_ready_f_;
logic [0:0][16:0] mem_ctrl_intersect_unit_flat_pos_out_0_f_;
logic mem_ctrl_intersect_unit_flat_pos_out_0_valid_f_;
logic [0:0][16:0] mem_ctrl_intersect_unit_flat_pos_out_1_f_;
logic mem_ctrl_intersect_unit_flat_pos_out_1_valid_f_;
logic mem_ctrl_reduce_pe_cluster_flat_clk;
logic [0:0][16:0] mem_ctrl_reduce_pe_cluster_flat_data0_f_;
logic mem_ctrl_reduce_pe_cluster_flat_data0_ready_f_;
logic mem_ctrl_reduce_pe_cluster_flat_data0_valid_f_;
logic [0:0][16:0] mem_ctrl_reduce_pe_cluster_flat_data1_f_;
logic mem_ctrl_reduce_pe_cluster_flat_data1_ready_f_;
logic mem_ctrl_reduce_pe_cluster_flat_data1_valid_f_;
logic [0:0][16:0] mem_ctrl_reduce_pe_cluster_flat_data2_f_;
logic mem_ctrl_reduce_pe_cluster_flat_data2_ready_f_;
logic mem_ctrl_reduce_pe_cluster_flat_data2_valid_f_;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_data_in_ready_f_;
logic [0:0][16:0] mem_ctrl_reduce_pe_cluster_flat_reduce_data_out_f_;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_data_out_valid_f_;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_dense_mode;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_in_external;
logic [83:0] mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_onyxpeintf_inst;
logic [2:0] mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_sparse_num_inputs;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_tile_en;
logic [15:0] mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_default_value;
logic [15:0] mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_stop_lvl;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_tile_en;
logic mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_tile_en;
logic [0:0][16:0] mem_ctrl_reduce_pe_cluster_flat_res_f_;
logic mem_ctrl_reduce_pe_cluster_flat_res_p_f_;
logic mem_ctrl_reduce_pe_cluster_flat_res_ready_f_;
logic mem_ctrl_reduce_pe_cluster_flat_res_valid_f_;
logic [0:0][16:0] output_width_17_num_0_fifo_in;
logic output_width_17_num_0_fifo_in_ready;
logic output_width_17_num_0_fifo_in_valid;
logic [0:0][16:0] output_width_17_num_0_output_fifo_data_out;
logic output_width_17_num_0_output_fifo_empty;
logic output_width_17_num_0_output_fifo_full;
logic [0:0][16:0] output_width_17_num_1_fifo_in;
logic output_width_17_num_1_fifo_in_ready;
logic output_width_17_num_1_fifo_in_valid;
logic [0:0][16:0] output_width_17_num_1_output_fifo_data_out;
logic output_width_17_num_1_output_fifo_empty;
logic output_width_17_num_1_output_fifo_full;
logic [0:0][16:0] output_width_17_num_2_fifo_in;
logic output_width_17_num_2_fifo_in_ready;
logic output_width_17_num_2_fifo_in_valid;
logic [0:0][16:0] output_width_17_num_2_output_fifo_data_out;
logic output_width_17_num_2_output_fifo_empty;
logic output_width_17_num_2_output_fifo_full;
assign gclk = clk & tile_en;
assign mem_ctrl_intersect_unit_flat_clk = gclk & (mode == 3'h0);
assign mem_ctrl_crddrop_flat_clk = gclk & (mode == 3'h1);
assign mem_ctrl_crdhold_flat_clk = gclk & (mode == 3'h2);
assign mem_ctrl_Repeat_flat_clk = gclk & (mode == 3'h3);
assign mem_ctrl_RepeatSignalGenerator_flat_clk = gclk & (mode == 3'h4);
assign mem_ctrl_reduce_pe_cluster_flat_clk = gclk & (mode == 3'h5);
assign input_width_17_num_0_fifo_out_valid = ~input_width_17_num_0_input_fifo_empty;
always_comb begin
  input_width_17_num_0_fifo_out_ready = 1'h1;
  if (mode == 3'h0) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_intersect_unit_flat_coord_in_0_ready_f_;
  end
  else if (mode == 3'h1) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_crddrop_flat_cmrg_coord_in_0_ready_f_;
  end
  else if (mode == 3'h2) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_crdhold_flat_cmrg_coord_in_0_ready_f_;
  end
  else if (mode == 3'h3) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_Repeat_flat_proc_data_in_ready_f_;
  end
  else if (mode == 3'h4) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_RepeatSignalGenerator_flat_base_data_in_ready_f_;
  end
  else if (mode == 3'h5) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_reduce_pe_cluster_flat_data0_ready_f_;
  end
end
assign mem_ctrl_reduce_pe_cluster_flat_data0_f_ = PE_input_width_17_num_0_dense ? PE_input_width_17_num_0:
    input_width_17_num_0_fifo_out;
assign mem_ctrl_reduce_pe_cluster_flat_data0_valid_f_ = PE_input_width_17_num_0_dense ? 1'h1: input_width_17_num_0_fifo_out_valid;
always_comb begin
  PE_input_width_17_num_0_ready = 1'h1;
  if (mode == 3'h0) begin
    PE_input_width_17_num_0_ready = ~input_width_17_num_0_input_fifo_full;
  end
  else if (mode == 3'h1) begin
    PE_input_width_17_num_0_ready = ~input_width_17_num_0_input_fifo_full;
  end
  else if (mode == 3'h2) begin
    PE_input_width_17_num_0_ready = ~input_width_17_num_0_input_fifo_full;
  end
  else if (mode == 3'h3) begin
    PE_input_width_17_num_0_ready = ~input_width_17_num_0_input_fifo_full;
  end
  else if (mode == 3'h4) begin
    PE_input_width_17_num_0_ready = ~input_width_17_num_0_input_fifo_full;
  end
  else if (mode == 3'h5) begin
    PE_input_width_17_num_0_ready = PE_input_width_17_num_0_dense ? 1'h1: ~input_width_17_num_0_input_fifo_full;
  end
end
assign input_width_17_num_1_fifo_out_valid = ~input_width_17_num_1_input_fifo_empty;
always_comb begin
  input_width_17_num_1_fifo_out_ready = 1'h1;
  if (mode == 3'h0) begin
    input_width_17_num_1_fifo_out_ready = mem_ctrl_intersect_unit_flat_coord_in_1_ready_f_;
  end
  else if (mode == 3'h1) begin
    input_width_17_num_1_fifo_out_ready = mem_ctrl_crddrop_flat_cmrg_coord_in_1_ready_f_;
  end
  else if (mode == 3'h2) begin
    input_width_17_num_1_fifo_out_ready = mem_ctrl_crdhold_flat_cmrg_coord_in_1_ready_f_;
  end
  else if (mode == 3'h3) begin
    input_width_17_num_1_fifo_out_ready = mem_ctrl_Repeat_flat_repsig_data_in_ready_f_;
  end
  else if (mode == 3'h5) begin
    input_width_17_num_1_fifo_out_ready = mem_ctrl_reduce_pe_cluster_flat_data1_ready_f_;
  end
end
assign mem_ctrl_reduce_pe_cluster_flat_data1_f_ = PE_input_width_17_num_1_dense ? PE_input_width_17_num_1:
    input_width_17_num_1_fifo_out;
assign mem_ctrl_reduce_pe_cluster_flat_data1_valid_f_ = PE_input_width_17_num_1_dense ? 1'h1: input_width_17_num_1_fifo_out_valid;
always_comb begin
  PE_input_width_17_num_1_ready = 1'h1;
  if (mode == 3'h0) begin
    PE_input_width_17_num_1_ready = ~input_width_17_num_1_input_fifo_full;
  end
  else if (mode == 3'h1) begin
    PE_input_width_17_num_1_ready = ~input_width_17_num_1_input_fifo_full;
  end
  else if (mode == 3'h2) begin
    PE_input_width_17_num_1_ready = ~input_width_17_num_1_input_fifo_full;
  end
  else if (mode == 3'h3) begin
    PE_input_width_17_num_1_ready = ~input_width_17_num_1_input_fifo_full;
  end
  else if (mode == 3'h5) begin
    PE_input_width_17_num_1_ready = PE_input_width_17_num_1_dense ? 1'h1: ~input_width_17_num_1_input_fifo_full;
  end
end
assign input_width_17_num_2_fifo_out_valid = ~input_width_17_num_2_input_fifo_empty;
always_comb begin
  input_width_17_num_2_fifo_out_ready = 1'h1;
  if (mode == 3'h0) begin
    input_width_17_num_2_fifo_out_ready = mem_ctrl_intersect_unit_flat_pos_in_0_ready_f_;
  end
  else if (mode == 3'h5) begin
    input_width_17_num_2_fifo_out_ready = mem_ctrl_reduce_pe_cluster_flat_data2_ready_f_;
  end
end
assign mem_ctrl_reduce_pe_cluster_flat_data2_f_ = PE_input_width_17_num_2_dense ? PE_input_width_17_num_2:
    input_width_17_num_2_fifo_out;
assign mem_ctrl_reduce_pe_cluster_flat_data2_valid_f_ = PE_input_width_17_num_2_dense ? 1'h1: input_width_17_num_2_fifo_out_valid;
always_comb begin
  PE_input_width_17_num_2_ready = 1'h1;
  if (mode == 3'h0) begin
    PE_input_width_17_num_2_ready = ~input_width_17_num_2_input_fifo_full;
  end
  else if (mode == 3'h5) begin
    PE_input_width_17_num_2_ready = PE_input_width_17_num_2_dense ? 1'h1: ~input_width_17_num_2_input_fifo_full;
  end
end
assign input_width_17_num_3_fifo_out_valid = ~input_width_17_num_3_input_fifo_empty;
always_comb begin
  input_width_17_num_3_fifo_out_ready = 1'h1;
  if (mode == 3'h0) begin
    input_width_17_num_3_fifo_out_ready = mem_ctrl_intersect_unit_flat_pos_in_1_ready_f_;
  end
  else if (mode == 3'h5) begin
    input_width_17_num_3_fifo_out_ready = mem_ctrl_reduce_pe_cluster_flat_reduce_data_in_ready_f_;
  end
end
always_comb begin
  PE_input_width_17_num_3_ready = 1'h1;
  if (mode == 3'h0) begin
    PE_input_width_17_num_3_ready = ~input_width_17_num_3_input_fifo_full;
  end
  else if (mode == 3'h5) begin
    PE_input_width_17_num_3_ready = ~input_width_17_num_3_input_fifo_full;
  end
end
assign output_width_17_num_0_fifo_in_ready = ~output_width_17_num_0_output_fifo_full;
always_comb begin
  output_width_17_num_0_fifo_in = 17'h0;
  output_width_17_num_0_fifo_in_valid = 1'h0;
  if (mode == 3'h0) begin
    output_width_17_num_0_fifo_in = mem_ctrl_intersect_unit_flat_coord_out_f_;
    output_width_17_num_0_fifo_in_valid = mem_ctrl_intersect_unit_flat_coord_out_valid_f_;
  end
  else if (mode == 3'h1) begin
    output_width_17_num_0_fifo_in = mem_ctrl_crddrop_flat_cmrg_coord_out_0_f_;
    output_width_17_num_0_fifo_in_valid = mem_ctrl_crddrop_flat_cmrg_coord_out_0_valid_f_;
  end
  else if (mode == 3'h2) begin
    output_width_17_num_0_fifo_in = mem_ctrl_crdhold_flat_cmrg_coord_out_0_f_;
    output_width_17_num_0_fifo_in_valid = mem_ctrl_crdhold_flat_cmrg_coord_out_0_valid_f_;
  end
  else if (mode == 3'h3) begin
    output_width_17_num_0_fifo_in = mem_ctrl_Repeat_flat_ref_data_out_f_;
    output_width_17_num_0_fifo_in_valid = mem_ctrl_Repeat_flat_ref_data_out_valid_f_;
  end
  else if (mode == 3'h4) begin
    output_width_17_num_0_fifo_in = mem_ctrl_RepeatSignalGenerator_flat_repsig_data_out_f_;
    output_width_17_num_0_fifo_in_valid = mem_ctrl_RepeatSignalGenerator_flat_repsig_data_out_valid_f_;
  end
  else if (mode == 3'h5) begin
    output_width_17_num_0_fifo_in = mem_ctrl_reduce_pe_cluster_flat_reduce_data_out_f_;
    output_width_17_num_0_fifo_in_valid = mem_ctrl_reduce_pe_cluster_flat_reduce_data_out_valid_f_;
  end
end
always_comb begin
  PE_output_width_17_num_0 = 17'h0;
  if (mode == 3'h0) begin
    PE_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
  else if (mode == 3'h1) begin
    PE_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
  else if (mode == 3'h2) begin
    PE_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
  else if (mode == 3'h3) begin
    PE_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
  else if (mode == 3'h4) begin
    PE_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
  else if (mode == 3'h5) begin
    PE_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
end
always_comb begin
  PE_output_width_17_num_0_valid = 1'h0;
  if (mode == 3'h0) begin
    PE_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
  else if (mode == 3'h1) begin
    PE_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
  else if (mode == 3'h2) begin
    PE_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
  else if (mode == 3'h3) begin
    PE_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
  else if (mode == 3'h4) begin
    PE_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
  else if (mode == 3'h5) begin
    PE_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
end
assign output_width_17_num_1_fifo_in_ready = ~output_width_17_num_1_output_fifo_full;
always_comb begin
  output_width_17_num_1_fifo_in = 17'h0;
  output_width_17_num_1_fifo_in_valid = 1'h0;
  if (mode == 3'h0) begin
    output_width_17_num_1_fifo_in = mem_ctrl_intersect_unit_flat_pos_out_0_f_;
    output_width_17_num_1_fifo_in_valid = mem_ctrl_intersect_unit_flat_pos_out_0_valid_f_;
  end
  else if (mode == 3'h1) begin
    output_width_17_num_1_fifo_in = mem_ctrl_crddrop_flat_cmrg_coord_out_1_f_;
    output_width_17_num_1_fifo_in_valid = mem_ctrl_crddrop_flat_cmrg_coord_out_1_valid_f_;
  end
  else if (mode == 3'h2) begin
    output_width_17_num_1_fifo_in = mem_ctrl_crdhold_flat_cmrg_coord_out_1_f_;
    output_width_17_num_1_fifo_in_valid = mem_ctrl_crdhold_flat_cmrg_coord_out_1_valid_f_;
  end
  else if (mode == 3'h5) begin
    output_width_17_num_1_fifo_in = mem_ctrl_reduce_pe_cluster_flat_res_f_;
    output_width_17_num_1_fifo_in_valid = mem_ctrl_reduce_pe_cluster_flat_res_valid_f_;
  end
end
assign mem_ctrl_reduce_pe_cluster_flat_res_ready_f_ = PE_output_width_17_num_1_dense ? 1'h1: output_width_17_num_1_fifo_in_ready;
always_comb begin
  PE_output_width_17_num_1 = 17'h0;
  if (mode == 3'h0) begin
    PE_output_width_17_num_1 = output_width_17_num_1_output_fifo_data_out;
  end
  else if (mode == 3'h1) begin
    PE_output_width_17_num_1 = output_width_17_num_1_output_fifo_data_out;
  end
  else if (mode == 3'h2) begin
    PE_output_width_17_num_1 = output_width_17_num_1_output_fifo_data_out;
  end
  else if (mode == 3'h5) begin
    PE_output_width_17_num_1 = PE_output_width_17_num_1_dense ? mem_ctrl_reduce_pe_cluster_flat_res_f_:
        output_width_17_num_1_output_fifo_data_out;
  end
end
always_comb begin
  PE_output_width_17_num_1_valid = 1'h0;
  if (mode == 3'h0) begin
    PE_output_width_17_num_1_valid = ~output_width_17_num_1_output_fifo_empty;
  end
  else if (mode == 3'h1) begin
    PE_output_width_17_num_1_valid = ~output_width_17_num_1_output_fifo_empty;
  end
  else if (mode == 3'h2) begin
    PE_output_width_17_num_1_valid = ~output_width_17_num_1_output_fifo_empty;
  end
  else if (mode == 3'h5) begin
    PE_output_width_17_num_1_valid = PE_output_width_17_num_1_dense ? 1'h1: ~output_width_17_num_1_output_fifo_empty;
  end
end
assign output_width_17_num_2_fifo_in_ready = ~output_width_17_num_2_output_fifo_full;
always_comb begin
  output_width_17_num_2_fifo_in = 17'h0;
  output_width_17_num_2_fifo_in_valid = 1'h0;
  output_width_17_num_2_fifo_in = mem_ctrl_intersect_unit_flat_pos_out_1_f_;
  output_width_17_num_2_fifo_in_valid = mem_ctrl_intersect_unit_flat_pos_out_1_valid_f_;
end
always_comb begin
  PE_output_width_17_num_2 = 17'h0;
  if (mode == 3'h0) begin
    PE_output_width_17_num_2 = output_width_17_num_2_output_fifo_data_out;
  end
  else PE_output_width_17_num_2 = 17'h0;
end
always_comb begin
  PE_output_width_17_num_2_valid = 1'h0;
  if (mode == 3'h0) begin
    PE_output_width_17_num_2_valid = ~output_width_17_num_2_output_fifo_empty;
  end
  else PE_output_width_17_num_2_valid = 1'h0;
end
always_comb begin
  PE_output_width_1_num_0 = 1'h0;
  if (mode == 3'h5) begin
    PE_output_width_1_num_0 = mem_ctrl_reduce_pe_cluster_flat_res_p_f_;
  end
  else PE_output_width_1_num_0 = 1'h0;
end
assign {mem_ctrl_intersect_unit_flat_intersect_unit_inst_joiner_op, mem_ctrl_intersect_unit_flat_intersect_unit_inst_tile_en, mem_ctrl_intersect_unit_flat_intersect_unit_inst_vector_reduce_mode} = CONFIG_SPACE[2:0];
assign {mem_ctrl_crddrop_flat_crddrop_inst_cmrg_enable, mem_ctrl_crddrop_flat_crddrop_inst_cmrg_mode, mem_ctrl_crddrop_flat_crddrop_inst_cmrg_stop_lvl, mem_ctrl_crddrop_flat_crddrop_inst_tile_en} = CONFIG_SPACE[18:0];
assign {mem_ctrl_crdhold_flat_crdhold_inst_cmrg_enable, mem_ctrl_crdhold_flat_crdhold_inst_cmrg_stop_lvl, mem_ctrl_crdhold_flat_crdhold_inst_tile_en} = CONFIG_SPACE[17:0];
assign {mem_ctrl_Repeat_flat_Repeat_inst_root, mem_ctrl_Repeat_flat_Repeat_inst_spacc_mode, mem_ctrl_Repeat_flat_Repeat_inst_stop_lvl, mem_ctrl_Repeat_flat_Repeat_inst_tile_en} = CONFIG_SPACE[18:0];
assign {mem_ctrl_RepeatSignalGenerator_flat_RepeatSignalGenerator_inst_stop_lvl, mem_ctrl_RepeatSignalGenerator_flat_RepeatSignalGenerator_inst_tile_en} = CONFIG_SPACE[16:0];
assign {mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_dense_mode, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_in_external, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_onyxpeintf_inst, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_sparse_num_inputs, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_tile_en, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_default_value, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_stop_lvl, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_tile_en, mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_tile_en} = CONFIG_SPACE[123:0];
assign CONFIG_SPACE[31:0] = CONFIG_SPACE_0;
assign CONFIG_SPACE[63:32] = CONFIG_SPACE_1;
assign CONFIG_SPACE[95:64] = CONFIG_SPACE_2;
assign CONFIG_SPACE[123:96] = CONFIG_SPACE_3;
intersect_unit_flat mem_ctrl_intersect_unit_flat (
  .clk(mem_ctrl_intersect_unit_flat_clk),
  .clk_en(clk_en),
  .coord_in_0_f_(input_width_17_num_0_fifo_out),
  .coord_in_0_valid_f_(input_width_17_num_0_fifo_out_valid),
  .coord_in_1_f_(input_width_17_num_1_fifo_out),
  .coord_in_1_valid_f_(input_width_17_num_1_fifo_out_valid),
  .coord_out_ready_f_(output_width_17_num_0_fifo_in_ready),
  .flush(flush),
  .intersect_unit_inst_joiner_op(mem_ctrl_intersect_unit_flat_intersect_unit_inst_joiner_op),
  .intersect_unit_inst_tile_en(mem_ctrl_intersect_unit_flat_intersect_unit_inst_tile_en),
  .intersect_unit_inst_vector_reduce_mode(mem_ctrl_intersect_unit_flat_intersect_unit_inst_vector_reduce_mode),
  .pos_in_0_f_(input_width_17_num_2_fifo_out),
  .pos_in_0_valid_f_(input_width_17_num_2_fifo_out_valid),
  .pos_in_1_f_(input_width_17_num_3_fifo_out),
  .pos_in_1_valid_f_(input_width_17_num_3_fifo_out_valid),
  .pos_out_0_ready_f_(output_width_17_num_1_fifo_in_ready),
  .pos_out_1_ready_f_(output_width_17_num_2_fifo_in_ready),
  .rst_n(rst_n),
  .coord_in_0_ready_f_(mem_ctrl_intersect_unit_flat_coord_in_0_ready_f_),
  .coord_in_1_ready_f_(mem_ctrl_intersect_unit_flat_coord_in_1_ready_f_),
  .coord_out_f_(mem_ctrl_intersect_unit_flat_coord_out_f_),
  .coord_out_valid_f_(mem_ctrl_intersect_unit_flat_coord_out_valid_f_),
  .pos_in_0_ready_f_(mem_ctrl_intersect_unit_flat_pos_in_0_ready_f_),
  .pos_in_1_ready_f_(mem_ctrl_intersect_unit_flat_pos_in_1_ready_f_),
  .pos_out_0_f_(mem_ctrl_intersect_unit_flat_pos_out_0_f_),
  .pos_out_0_valid_f_(mem_ctrl_intersect_unit_flat_pos_out_0_valid_f_),
  .pos_out_1_f_(mem_ctrl_intersect_unit_flat_pos_out_1_f_),
  .pos_out_1_valid_f_(mem_ctrl_intersect_unit_flat_pos_out_1_valid_f_)
);

crddrop_flat mem_ctrl_crddrop_flat (
  .clk(mem_ctrl_crddrop_flat_clk),
  .clk_en(clk_en),
  .cmrg_coord_in_0_f_(input_width_17_num_0_fifo_out),
  .cmrg_coord_in_0_valid_f_(input_width_17_num_0_fifo_out_valid),
  .cmrg_coord_in_1_f_(input_width_17_num_1_fifo_out),
  .cmrg_coord_in_1_valid_f_(input_width_17_num_1_fifo_out_valid),
  .cmrg_coord_out_0_ready_f_(output_width_17_num_0_fifo_in_ready),
  .cmrg_coord_out_1_ready_f_(output_width_17_num_1_fifo_in_ready),
  .crddrop_inst_cmrg_enable(mem_ctrl_crddrop_flat_crddrop_inst_cmrg_enable),
  .crddrop_inst_cmrg_mode(mem_ctrl_crddrop_flat_crddrop_inst_cmrg_mode),
  .crddrop_inst_cmrg_stop_lvl(mem_ctrl_crddrop_flat_crddrop_inst_cmrg_stop_lvl),
  .crddrop_inst_tile_en(mem_ctrl_crddrop_flat_crddrop_inst_tile_en),
  .flush(flush),
  .rst_n(rst_n),
  .cmrg_coord_in_0_ready_f_(mem_ctrl_crddrop_flat_cmrg_coord_in_0_ready_f_),
  .cmrg_coord_in_1_ready_f_(mem_ctrl_crddrop_flat_cmrg_coord_in_1_ready_f_),
  .cmrg_coord_out_0_f_(mem_ctrl_crddrop_flat_cmrg_coord_out_0_f_),
  .cmrg_coord_out_0_valid_f_(mem_ctrl_crddrop_flat_cmrg_coord_out_0_valid_f_),
  .cmrg_coord_out_1_f_(mem_ctrl_crddrop_flat_cmrg_coord_out_1_f_),
  .cmrg_coord_out_1_valid_f_(mem_ctrl_crddrop_flat_cmrg_coord_out_1_valid_f_)
);

crdhold_flat mem_ctrl_crdhold_flat (
  .clk(mem_ctrl_crdhold_flat_clk),
  .clk_en(clk_en),
  .cmrg_coord_in_0_f_(input_width_17_num_0_fifo_out),
  .cmrg_coord_in_0_valid_f_(input_width_17_num_0_fifo_out_valid),
  .cmrg_coord_in_1_f_(input_width_17_num_1_fifo_out),
  .cmrg_coord_in_1_valid_f_(input_width_17_num_1_fifo_out_valid),
  .cmrg_coord_out_0_ready_f_(output_width_17_num_0_fifo_in_ready),
  .cmrg_coord_out_1_ready_f_(output_width_17_num_1_fifo_in_ready),
  .crdhold_inst_cmrg_enable(mem_ctrl_crdhold_flat_crdhold_inst_cmrg_enable),
  .crdhold_inst_cmrg_stop_lvl(mem_ctrl_crdhold_flat_crdhold_inst_cmrg_stop_lvl),
  .crdhold_inst_tile_en(mem_ctrl_crdhold_flat_crdhold_inst_tile_en),
  .flush(flush),
  .rst_n(rst_n),
  .cmrg_coord_in_0_ready_f_(mem_ctrl_crdhold_flat_cmrg_coord_in_0_ready_f_),
  .cmrg_coord_in_1_ready_f_(mem_ctrl_crdhold_flat_cmrg_coord_in_1_ready_f_),
  .cmrg_coord_out_0_f_(mem_ctrl_crdhold_flat_cmrg_coord_out_0_f_),
  .cmrg_coord_out_0_valid_f_(mem_ctrl_crdhold_flat_cmrg_coord_out_0_valid_f_),
  .cmrg_coord_out_1_f_(mem_ctrl_crdhold_flat_cmrg_coord_out_1_f_),
  .cmrg_coord_out_1_valid_f_(mem_ctrl_crdhold_flat_cmrg_coord_out_1_valid_f_)
);

Repeat_flat mem_ctrl_Repeat_flat (
  .Repeat_inst_root(mem_ctrl_Repeat_flat_Repeat_inst_root),
  .Repeat_inst_spacc_mode(mem_ctrl_Repeat_flat_Repeat_inst_spacc_mode),
  .Repeat_inst_stop_lvl(mem_ctrl_Repeat_flat_Repeat_inst_stop_lvl),
  .Repeat_inst_tile_en(mem_ctrl_Repeat_flat_Repeat_inst_tile_en),
  .clk(mem_ctrl_Repeat_flat_clk),
  .clk_en(clk_en),
  .flush(flush),
  .proc_data_in_f_(input_width_17_num_0_fifo_out),
  .proc_data_in_valid_f_(input_width_17_num_0_fifo_out_valid),
  .ref_data_out_ready_f_(output_width_17_num_0_fifo_in_ready),
  .repsig_data_in_f_(input_width_17_num_1_fifo_out),
  .repsig_data_in_valid_f_(input_width_17_num_1_fifo_out_valid),
  .rst_n(rst_n),
  .proc_data_in_ready_f_(mem_ctrl_Repeat_flat_proc_data_in_ready_f_),
  .ref_data_out_f_(mem_ctrl_Repeat_flat_ref_data_out_f_),
  .ref_data_out_valid_f_(mem_ctrl_Repeat_flat_ref_data_out_valid_f_),
  .repsig_data_in_ready_f_(mem_ctrl_Repeat_flat_repsig_data_in_ready_f_)
);

RepeatSignalGenerator_flat mem_ctrl_RepeatSignalGenerator_flat (
  .RepeatSignalGenerator_inst_stop_lvl(mem_ctrl_RepeatSignalGenerator_flat_RepeatSignalGenerator_inst_stop_lvl),
  .RepeatSignalGenerator_inst_tile_en(mem_ctrl_RepeatSignalGenerator_flat_RepeatSignalGenerator_inst_tile_en),
  .base_data_in_f_(input_width_17_num_0_fifo_out),
  .base_data_in_valid_f_(input_width_17_num_0_fifo_out_valid),
  .clk(mem_ctrl_RepeatSignalGenerator_flat_clk),
  .clk_en(clk_en),
  .flush(flush),
  .repsig_data_out_ready_f_(output_width_17_num_0_fifo_in_ready),
  .rst_n(rst_n),
  .base_data_in_ready_f_(mem_ctrl_RepeatSignalGenerator_flat_base_data_in_ready_f_),
  .repsig_data_out_f_(mem_ctrl_RepeatSignalGenerator_flat_repsig_data_out_f_),
  .repsig_data_out_valid_f_(mem_ctrl_RepeatSignalGenerator_flat_repsig_data_out_valid_f_)
);

reduce_pe_cluster_flat mem_ctrl_reduce_pe_cluster_flat (
  .bit0_f_(PE_input_width_1_num_0),
  .bit1_f_(PE_input_width_1_num_1),
  .bit2_f_(PE_input_width_1_num_2),
  .clk(mem_ctrl_reduce_pe_cluster_flat_clk),
  .clk_en(clk_en),
  .data0_f_(mem_ctrl_reduce_pe_cluster_flat_data0_f_),
  .data0_valid_f_(mem_ctrl_reduce_pe_cluster_flat_data0_valid_f_),
  .data1_f_(mem_ctrl_reduce_pe_cluster_flat_data1_f_),
  .data1_valid_f_(mem_ctrl_reduce_pe_cluster_flat_data1_valid_f_),
  .data2_f_(mem_ctrl_reduce_pe_cluster_flat_data2_f_),
  .data2_valid_f_(mem_ctrl_reduce_pe_cluster_flat_data2_valid_f_),
  .flush(flush),
  .reduce_data_in_f_(input_width_17_num_3_fifo_out),
  .reduce_data_in_valid_f_(input_width_17_num_3_fifo_out_valid),
  .reduce_data_out_ready_f_(output_width_17_num_0_fifo_in_ready),
  .reduce_pe_cluster_inst_pe_dense_mode(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_dense_mode),
  .reduce_pe_cluster_inst_pe_in_external(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_in_external),
  .reduce_pe_cluster_inst_pe_onyxpeintf_inst(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_onyxpeintf_inst),
  .reduce_pe_cluster_inst_pe_sparse_num_inputs(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_sparse_num_inputs),
  .reduce_pe_cluster_inst_pe_tile_en(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_pe_tile_en),
  .reduce_pe_cluster_inst_reduce_default_value(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_default_value),
  .reduce_pe_cluster_inst_reduce_stop_lvl(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_stop_lvl),
  .reduce_pe_cluster_inst_reduce_tile_en(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_reduce_tile_en),
  .reduce_pe_cluster_inst_tile_en(mem_ctrl_reduce_pe_cluster_flat_reduce_pe_cluster_inst_tile_en),
  .res_ready_f_(mem_ctrl_reduce_pe_cluster_flat_res_ready_f_),
  .rst_n(rst_n),
  .data0_ready_f_(mem_ctrl_reduce_pe_cluster_flat_data0_ready_f_),
  .data1_ready_f_(mem_ctrl_reduce_pe_cluster_flat_data1_ready_f_),
  .data2_ready_f_(mem_ctrl_reduce_pe_cluster_flat_data2_ready_f_),
  .reduce_data_in_ready_f_(mem_ctrl_reduce_pe_cluster_flat_reduce_data_in_ready_f_),
  .reduce_data_out_f_(mem_ctrl_reduce_pe_cluster_flat_reduce_data_out_f_),
  .reduce_data_out_valid_f_(mem_ctrl_reduce_pe_cluster_flat_reduce_data_out_valid_f_),
  .reduce_pe_cluster_inst_pe_onyxpeintf_O2(reduce_pe_cluster_inst_pe_onyxpeintf_O2),
  .reduce_pe_cluster_inst_pe_onyxpeintf_O3(reduce_pe_cluster_inst_pe_onyxpeintf_O3),
  .reduce_pe_cluster_inst_pe_onyxpeintf_O4(reduce_pe_cluster_inst_pe_onyxpeintf_O4),
  .res_f_(mem_ctrl_reduce_pe_cluster_flat_res_f_),
  .res_p_f_(mem_ctrl_reduce_pe_cluster_flat_res_p_f_),
  .res_valid_f_(mem_ctrl_reduce_pe_cluster_flat_res_valid_f_)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_0_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(PE_input_width_17_num_0),
  .flush(flush),
  .pop(input_width_17_num_0_fifo_out_ready),
  .push(PE_input_width_17_num_0_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_0_fifo_out),
  .empty(input_width_17_num_0_input_fifo_empty),
  .full(input_width_17_num_0_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_1_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(PE_input_width_17_num_1),
  .flush(flush),
  .pop(input_width_17_num_1_fifo_out_ready),
  .push(PE_input_width_17_num_1_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_1_fifo_out),
  .empty(input_width_17_num_1_input_fifo_empty),
  .full(input_width_17_num_1_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_2_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(PE_input_width_17_num_2),
  .flush(flush),
  .pop(input_width_17_num_2_fifo_out_ready),
  .push(PE_input_width_17_num_2_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_2_fifo_out),
  .empty(input_width_17_num_2_input_fifo_empty),
  .full(input_width_17_num_2_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_3_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(PE_input_width_17_num_3),
  .flush(flush),
  .pop(input_width_17_num_3_fifo_out_ready),
  .push(PE_input_width_17_num_3_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_3_fifo_out),
  .empty(input_width_17_num_3_input_fifo_empty),
  .full(input_width_17_num_3_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 output_width_17_num_0_output_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(output_width_17_num_0_fifo_in),
  .flush(flush),
  .pop(PE_output_width_17_num_0_ready),
  .push(output_width_17_num_0_fifo_in_valid),
  .rst_n(rst_n),
  .data_out(output_width_17_num_0_output_fifo_data_out),
  .empty(output_width_17_num_0_output_fifo_empty),
  .full(output_width_17_num_0_output_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 output_width_17_num_1_output_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(output_width_17_num_1_fifo_in),
  .flush(flush),
  .pop(PE_output_width_17_num_1_ready),
  .push(output_width_17_num_1_fifo_in_valid),
  .rst_n(rst_n),
  .data_out(output_width_17_num_1_output_fifo_data_out),
  .empty(output_width_17_num_1_output_fifo_empty),
  .full(output_width_17_num_1_output_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 output_width_17_num_2_output_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(output_width_17_num_2_fifo_in),
  .flush(flush),
  .pop(PE_output_width_17_num_2_ready),
  .push(output_width_17_num_2_fifo_in_valid),
  .rst_n(rst_n),
  .data_out(output_width_17_num_2_output_fifo_data_out),
  .empty(output_width_17_num_2_output_fifo_empty),
  .full(output_width_17_num_2_output_fifo_full)
);

endmodule   // PE_inner

module PE_inner_W (
  input logic [31:0] CONFIG_SPACE_0,
  input logic [31:0] CONFIG_SPACE_1,
  input logic [31:0] CONFIG_SPACE_2,
  input logic [27:0] CONFIG_SPACE_3,
  input logic [0:0] [16:0] PE_input_width_17_num_0,
  input logic PE_input_width_17_num_0_dense,
  input logic PE_input_width_17_num_0_valid,
  input logic [0:0] [16:0] PE_input_width_17_num_1,
  input logic PE_input_width_17_num_1_dense,
  input logic PE_input_width_17_num_1_valid,
  input logic [0:0] [16:0] PE_input_width_17_num_2,
  input logic PE_input_width_17_num_2_dense,
  input logic PE_input_width_17_num_2_valid,
  input logic [0:0] [16:0] PE_input_width_17_num_3,
  input logic PE_input_width_17_num_3_valid,
  input logic PE_input_width_1_num_0,
  input logic PE_input_width_1_num_1,
  input logic PE_input_width_1_num_2,
  input logic PE_output_width_17_num_0_ready,
  input logic PE_output_width_17_num_1_dense,
  input logic PE_output_width_17_num_1_ready,
  input logic PE_output_width_17_num_2_ready,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mode,
  input logic rst_n,
  input logic tile_en,
  output logic PE_input_width_17_num_0_ready,
  output logic PE_input_width_17_num_1_ready,
  output logic PE_input_width_17_num_2_ready,
  output logic PE_input_width_17_num_3_ready,
  output logic [0:0] [16:0] PE_output_width_17_num_0,
  output logic PE_output_width_17_num_0_valid,
  output logic [0:0] [16:0] PE_output_width_17_num_1,
  output logic PE_output_width_17_num_1_valid,
  output logic [0:0] [16:0] PE_output_width_17_num_2,
  output logic PE_output_width_17_num_2_valid,
  output logic PE_output_width_1_num_0,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O2,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O3,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O4
);

PE_inner PE_inner (
  .CONFIG_SPACE_0(CONFIG_SPACE_0),
  .CONFIG_SPACE_1(CONFIG_SPACE_1),
  .CONFIG_SPACE_2(CONFIG_SPACE_2),
  .CONFIG_SPACE_3(CONFIG_SPACE_3),
  .PE_input_width_17_num_0(PE_input_width_17_num_0),
  .PE_input_width_17_num_0_dense(PE_input_width_17_num_0_dense),
  .PE_input_width_17_num_0_valid(PE_input_width_17_num_0_valid),
  .PE_input_width_17_num_1(PE_input_width_17_num_1),
  .PE_input_width_17_num_1_dense(PE_input_width_17_num_1_dense),
  .PE_input_width_17_num_1_valid(PE_input_width_17_num_1_valid),
  .PE_input_width_17_num_2(PE_input_width_17_num_2),
  .PE_input_width_17_num_2_dense(PE_input_width_17_num_2_dense),
  .PE_input_width_17_num_2_valid(PE_input_width_17_num_2_valid),
  .PE_input_width_17_num_3(PE_input_width_17_num_3),
  .PE_input_width_17_num_3_valid(PE_input_width_17_num_3_valid),
  .PE_input_width_1_num_0(PE_input_width_1_num_0),
  .PE_input_width_1_num_1(PE_input_width_1_num_1),
  .PE_input_width_1_num_2(PE_input_width_1_num_2),
  .PE_output_width_17_num_0_ready(PE_output_width_17_num_0_ready),
  .PE_output_width_17_num_1_dense(PE_output_width_17_num_1_dense),
  .PE_output_width_17_num_1_ready(PE_output_width_17_num_1_ready),
  .PE_output_width_17_num_2_ready(PE_output_width_17_num_2_ready),
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mode(mode),
  .rst_n(rst_n),
  .tile_en(tile_en),
  .PE_input_width_17_num_0_ready(PE_input_width_17_num_0_ready),
  .PE_input_width_17_num_1_ready(PE_input_width_17_num_1_ready),
  .PE_input_width_17_num_2_ready(PE_input_width_17_num_2_ready),
  .PE_input_width_17_num_3_ready(PE_input_width_17_num_3_ready),
  .PE_output_width_17_num_0(PE_output_width_17_num_0),
  .PE_output_width_17_num_0_valid(PE_output_width_17_num_0_valid),
  .PE_output_width_17_num_1(PE_output_width_17_num_1),
  .PE_output_width_17_num_1_valid(PE_output_width_17_num_1_valid),
  .PE_output_width_17_num_2(PE_output_width_17_num_2),
  .PE_output_width_17_num_2_valid(PE_output_width_17_num_2_valid),
  .PE_output_width_1_num_0(PE_output_width_1_num_0),
  .reduce_pe_cluster_inst_pe_onyxpeintf_O2(reduce_pe_cluster_inst_pe_onyxpeintf_O2),
  .reduce_pe_cluster_inst_pe_onyxpeintf_O3(reduce_pe_cluster_inst_pe_onyxpeintf_O3),
  .reduce_pe_cluster_inst_pe_onyxpeintf_O4(reduce_pe_cluster_inst_pe_onyxpeintf_O4)
);

endmodule   // PE_inner_W

module PE_onyx (
  input logic bit0,
  input logic bit1,
  input logic bit2,
  input logic clk,
  input logic clk_en,
  input logic [16:0] data0,
  input logic data0_valid,
  input logic [16:0] data1,
  input logic data1_valid,
  input logic [16:0] data2,
  input logic data2_valid,
  input logic dense_mode,
  input logic flush,
  input logic [83:0] onyxpeintf_inst,
  input logic res_ready,
  input logic rst_n,
  input logic [2:0] sparse_num_inputs,
  input logic tile_en,
  output logic data0_ready,
  output logic data1_ready,
  output logic data2_ready,
  output logic [15:0] onyxpeintf_O2,
  output logic [15:0] onyxpeintf_O3,
  output logic [15:0] onyxpeintf_O4,
  output logic [16:0] res,
  output logic res_p,
  output logic res_valid
);

logic [15:0] data_to_fifo;
logic gclk;
logic [2:0][16:0] infifo_in_packed;
logic [2:0][15:0] infifo_out_data;
logic [2:0] infifo_out_eos;
logic infifo_out_maybe_0;
logic infifo_out_maybe_1;
logic infifo_out_maybe_2;
logic [2:0][16:0] infifo_out_packed;
logic [2:0] infifo_out_valid;
logic [2:0] infifo_pop;
logic infifo_push_0;
logic infifo_push_1;
logic infifo_push_2;
logic [0:0][16:0] input_fifo_0_data_out;
logic input_fifo_0_empty;
logic input_fifo_0_full;
logic [0:0][16:0] input_fifo_1_data_out;
logic input_fifo_1_empty;
logic input_fifo_1_full;
logic [0:0][16:0] input_fifo_2_data_out;
logic input_fifo_2_empty;
logic input_fifo_2_full;
logic onyxpeintf_ASYNCRESET;
logic [15:0] onyxpeintf_data0;
logic [15:0] onyxpeintf_data1;
logic [15:0] onyxpeintf_data2;
logic outfifo_full;
logic outfifo_in_eos;
logic [16:0] outfifo_in_packed;
logic [16:0] outfifo_out_packed;
logic outfifo_pop;
logic outfifo_push;
logic output_fifo_empty;
logic [15:0] pe_output;
assign gclk = clk & tile_en;
assign data0_ready = dense_mode ? 1'h1: ~input_fifo_0_full;
assign data1_ready = dense_mode ? 1'h1: ~input_fifo_1_full;
assign data2_ready = dense_mode ? 1'h1: ~input_fifo_2_full;
assign infifo_in_packed[0] = data0;
assign infifo_out_eos[0] = infifo_out_packed[0][16];
assign infifo_out_data[0] = infifo_out_packed[0][15:0];
assign infifo_in_packed[1] = data1;
assign infifo_out_eos[1] = infifo_out_packed[1][16];
assign infifo_out_data[1] = infifo_out_packed[1][15:0];
assign infifo_in_packed[2] = data2;
assign infifo_out_eos[2] = infifo_out_packed[2][16];
assign infifo_out_data[2] = infifo_out_packed[2][15:0];
assign infifo_push_0 = data0_valid;
assign infifo_push_1 = data1_valid;
assign infifo_push_2 = data2_valid;
assign infifo_out_packed[0] = input_fifo_0_data_out;
assign infifo_out_packed[1] = input_fifo_1_data_out;
assign infifo_out_packed[2] = input_fifo_2_data_out;
assign infifo_out_valid[0] = ~input_fifo_0_empty;
assign infifo_out_valid[1] = ~input_fifo_1_empty;
assign infifo_out_valid[2] = ~input_fifo_2_empty;
assign outfifo_in_packed[16] = outfifo_in_eos;
assign outfifo_in_packed[15:0] = data_to_fifo;
assign res = dense_mode ? 17'(pe_output): outfifo_out_packed;
assign res_valid = dense_mode ? 1'h1: ~output_fifo_empty;
assign outfifo_pop = res_ready;
assign infifo_out_maybe_0 = infifo_out_eos[0] & infifo_out_valid[0] & (infifo_out_data[0][9:8] == 2'h2);
assign infifo_out_maybe_1 = infifo_out_eos[1] & infifo_out_valid[1] & (infifo_out_data[1][9:8] == 2'h2);
assign infifo_out_maybe_2 = infifo_out_eos[2] & infifo_out_valid[2] & (infifo_out_data[2][9:8] == 2'h2);
assign onyxpeintf_ASYNCRESET = ~rst_n;
assign onyxpeintf_data0 = dense_mode ? data0[15:0]: infifo_out_maybe_0 ? 16'h0: infifo_out_data[0];
assign onyxpeintf_data1 = dense_mode ? data1[15:0]: infifo_out_maybe_1 ? 16'h0: infifo_out_data[1];
assign onyxpeintf_data2 = dense_mode ? data2[15:0]: infifo_out_maybe_2 ? 16'h0: infifo_out_data[2];
always_comb begin
  outfifo_push = 1'h0;
  outfifo_in_eos = 1'h0;
  data_to_fifo = 16'h0;
  infifo_pop[0] = 1'h0;
  infifo_pop[1] = 1'h0;
  infifo_pop[2] = 1'h0;
  if (((infifo_out_valid & sparse_num_inputs) == sparse_num_inputs) & (~outfifo_full) & (~dense_mode)) begin
    if (~((infifo_out_eos & sparse_num_inputs) == sparse_num_inputs)) begin
      outfifo_push = 1'h1;
      outfifo_in_eos = 1'h0;
      data_to_fifo = pe_output;
      infifo_pop[0] = infifo_out_valid[0] & sparse_num_inputs[0];
      infifo_pop[1] = infifo_out_valid[1] & sparse_num_inputs[1];
      infifo_pop[2] = infifo_out_valid[2] & sparse_num_inputs[2];
    end
    else begin
      outfifo_push = 1'h1;
      outfifo_in_eos = 1'h1;
      data_to_fifo = sparse_num_inputs[0] ? infifo_out_data[0]: sparse_num_inputs[1] ?
          infifo_out_data[1]: infifo_out_data[2];
      infifo_pop[0] = infifo_out_valid[0] & sparse_num_inputs[0];
      infifo_pop[1] = infifo_out_valid[1] & sparse_num_inputs[1];
      infifo_pop[2] = infifo_out_valid[2] & sparse_num_inputs[2];
    end
  end
end
reg_fifo_depth_0_w_17_afd_2 input_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(infifo_in_packed[0]),
  .flush(flush),
  .pop(infifo_pop[0]),
  .push(infifo_push_0),
  .rst_n(rst_n),
  .data_out(input_fifo_0_data_out),
  .empty(input_fifo_0_empty),
  .full(input_fifo_0_full)
);

reg_fifo_depth_0_w_17_afd_2 input_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(infifo_in_packed[1]),
  .flush(flush),
  .pop(infifo_pop[1]),
  .push(infifo_push_1),
  .rst_n(rst_n),
  .data_out(input_fifo_1_data_out),
  .empty(input_fifo_1_empty),
  .full(input_fifo_1_full)
);

reg_fifo_depth_0_w_17_afd_2 input_fifo_2 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(infifo_in_packed[2]),
  .flush(flush),
  .pop(infifo_pop[2]),
  .push(infifo_push_2),
  .rst_n(rst_n),
  .data_out(input_fifo_2_data_out),
  .empty(input_fifo_2_empty),
  .full(input_fifo_2_full)
);

reg_fifo_depth_2_w_17_afd_2 output_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(outfifo_in_packed),
  .flush(flush),
  .pop(outfifo_pop),
  .push(outfifo_push),
  .rst_n(rst_n),
  .data_out(outfifo_out_packed),
  .empty(output_fifo_empty),
  .full(outfifo_full)
);

PEGEN_PE onyxpeintf (
  .ASYNCRESET(onyxpeintf_ASYNCRESET),
  .CLK(gclk),
  .bit0(bit0),
  .bit1(bit1),
  .bit2(bit2),
  .clk_en(clk_en),
  .data0(onyxpeintf_data0),
  .data1(onyxpeintf_data1),
  .data2(onyxpeintf_data2),
  .inst(onyxpeintf_inst),
  .O0(pe_output),
  .O1(res_p),
  .O2(onyxpeintf_O2),
  .O3(onyxpeintf_O3),
  .O4(onyxpeintf_O4)
);

endmodule   // PE_onyx

module Repeat (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [16:0] proc_data_in,
  input logic proc_data_in_valid,
  input logic ref_data_out_ready,
  input logic [16:0] repsig_data_in,
  input logic repsig_data_in_valid,
  input logic root,
  input logic rst_n,
  input logic spacc_mode,
  input logic [15:0] stop_lvl,
  input logic tile_en,
  output logic proc_data_in_ready,
  output logic [16:0] ref_data_out,
  output logic ref_data_out_valid,
  output logic repsig_data_in_ready
);

typedef enum logic[1:0] {
  INJECT0 = 2'h0,
  INJECT1 = 2'h1,
  PASS_REPEAT = 2'h2,
  START = 2'h3
} repeat_fsm_state;
logic blank_repeat;
logic blank_repeat_stop;
logic clr_last_pushed_data;
logic gclk;
logic proc_data;
logic proc_done;
logic proc_fifo_full;
logic [15:0] proc_fifo_inject_data;
logic proc_fifo_inject_eos;
logic proc_fifo_inject_push;
logic [15:0] proc_fifo_out_data;
logic proc_fifo_out_eos;
logic proc_fifo_pop;
logic proc_fifo_push;
logic proc_fifo_valid;
logic [0:0][16:0] proc_in_fifo_data_in;
logic [0:0][16:0] proc_in_fifo_data_out;
logic proc_in_fifo_empty;
logic proc_in_fifo_full;
logic proc_stop;
logic pushed_data_sticky_sticky;
logic pushed_data_sticky_was_high;
logic ref_fifo_full;
logic [15:0] ref_fifo_in_data;
logic ref_fifo_in_eos;
logic ref_fifo_push;
logic ref_maybe;
logic [0:0][16:0] ref_out_fifo_data_in;
logic ref_out_fifo_empty;
repeat_fsm_state repeat_fsm_current_state;
repeat_fsm_state repeat_fsm_next_state;
logic repsig_done;
logic [15:0] repsig_fifo_out_data;
logic repsig_fifo_out_eos;
logic repsig_fifo_pop;
logic repsig_fifo_valid;
logic [0:0][16:0] repsig_in_fifo_data_out;
logic repsig_in_fifo_empty;
logic repsig_in_fifo_full;
logic repsig_sig;
logic repsig_stop;
logic seen_root_eos_sticky;
logic seen_root_eos_was_high;
logic set_last_pushed_data;
assign gclk = clk & tile_en;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_data_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_data_sticky_was_high <= 1'h0;
    end
    else if (clr_last_pushed_data) begin
      pushed_data_sticky_was_high <= 1'h0;
    end
    else if (set_last_pushed_data) begin
      pushed_data_sticky_was_high <= 1'h1;
    end
  end
end
assign pushed_data_sticky_sticky = pushed_data_sticky_was_high;
assign {repsig_fifo_out_eos, repsig_fifo_out_data} = repsig_in_fifo_data_out;
assign repsig_data_in_ready = ~repsig_in_fifo_full;
assign repsig_fifo_valid = ~repsig_in_fifo_empty;
assign proc_fifo_push = root ? proc_fifo_inject_push: proc_data_in_valid;
assign proc_in_fifo_data_in = root ? {proc_fifo_inject_eos, proc_fifo_inject_data}: proc_data_in;
assign {proc_fifo_out_eos, proc_fifo_out_data} = proc_in_fifo_data_out;
assign proc_data_in_ready = ~proc_in_fifo_full;
assign proc_fifo_full = proc_in_fifo_full;
assign proc_fifo_valid = ~proc_in_fifo_empty;
assign ref_out_fifo_data_in = {ref_fifo_in_eos, ref_fifo_in_data};
assign ref_data_out_valid = ~ref_out_fifo_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    seen_root_eos_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      seen_root_eos_was_high <= 1'h0;
    end
    else if (1'h0) begin
      seen_root_eos_was_high <= 1'h0;
    end
    else if ((proc_fifo_out_data == 16'h0) & proc_fifo_out_eos & proc_fifo_valid) begin
      seen_root_eos_was_high <= 1'h1;
    end
  end
end
assign seen_root_eos_sticky = ((proc_fifo_out_data == 16'h0) & proc_fifo_out_eos & proc_fifo_valid) |
    seen_root_eos_was_high;
assign ref_maybe = proc_fifo_valid & proc_fifo_out_eos & (proc_fifo_out_data[9:8] == 2'h2);
assign proc_data = ((~proc_fifo_out_eos) | ref_maybe) & proc_fifo_valid;
assign proc_stop = (proc_fifo_out_data[9:8] == 2'h0) & proc_fifo_out_eos & proc_fifo_valid;
assign proc_done = (proc_fifo_out_data[9:8] == 2'h1) & proc_fifo_out_eos & proc_fifo_valid;
assign repsig_stop = (repsig_fifo_out_data[9:8] == 2'h0) & repsig_fifo_out_eos & repsig_fifo_valid;
assign repsig_sig = (~repsig_fifo_out_eos) & repsig_fifo_valid;
assign repsig_done = (repsig_fifo_out_data[9:8] == 2'h1) & repsig_fifo_out_eos & repsig_fifo_valid;
assign blank_repeat = proc_stop & (~repsig_stop) & repsig_fifo_valid;
assign blank_repeat_stop = proc_stop & repsig_stop & pushed_data_sticky_sticky;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    repeat_fsm_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      repeat_fsm_current_state <= START;
    end
    else repeat_fsm_current_state <= repeat_fsm_next_state;
  end
end
always_comb begin
  repeat_fsm_next_state = repeat_fsm_current_state;
  unique case (repeat_fsm_current_state)
    INJECT0: begin
        if (~proc_fifo_full) begin
          repeat_fsm_next_state = INJECT1;
        end
        else repeat_fsm_next_state = INJECT0;
      end
    INJECT1: begin
        if (~proc_fifo_full) begin
          repeat_fsm_next_state = PASS_REPEAT;
        end
        else repeat_fsm_next_state = INJECT1;
      end
    PASS_REPEAT: begin
        if (proc_done & repsig_done & (~ref_fifo_full)) begin
          repeat_fsm_next_state = START;
        end
        else repeat_fsm_next_state = PASS_REPEAT;
      end
    START: begin
        if (root & tile_en) begin
          repeat_fsm_next_state = INJECT0;
        end
        else if ((~root) & tile_en) begin
          repeat_fsm_next_state = PASS_REPEAT;
        end
        else repeat_fsm_next_state = START;
      end
    default: begin end
  endcase
end
always_comb begin
  unique case (repeat_fsm_current_state)
    INJECT0: begin :repeat_fsm_INJECT0_Output
        ref_fifo_in_data = 16'h0;
        ref_fifo_in_eos = 1'h0;
        ref_fifo_push = 1'h0;
        proc_fifo_pop = 1'h0;
        repsig_fifo_pop = 1'h0;
        proc_fifo_inject_push = 1'h1;
        proc_fifo_inject_data = 16'h0;
        proc_fifo_inject_eos = 1'h0;
        set_last_pushed_data = 1'h0;
        clr_last_pushed_data = 1'h0;
      end :repeat_fsm_INJECT0_Output
    INJECT1: begin :repeat_fsm_INJECT1_Output
        ref_fifo_in_data = 16'h0;
        ref_fifo_in_eos = 1'h0;
        ref_fifo_push = 1'h0;
        proc_fifo_pop = 1'h0;
        repsig_fifo_pop = 1'h0;
        proc_fifo_inject_push = 1'h1;
        proc_fifo_inject_data = 16'h100;
        proc_fifo_inject_eos = 1'h1;
        set_last_pushed_data = 1'h0;
        clr_last_pushed_data = 1'h0;
      end :repeat_fsm_INJECT1_Output
    PASS_REPEAT: begin :repeat_fsm_PASS_REPEAT_Output
        ref_fifo_in_data = repsig_stop ? repsig_fifo_out_data: proc_fifo_out_data;
        ref_fifo_in_eos = ref_maybe | repsig_done | repsig_stop;
        ref_fifo_push = (~ref_fifo_full) & ((proc_done & repsig_done) | (proc_data & repsig_fifo_valid)
            | (proc_stop & (~pushed_data_sticky_sticky) & repsig_stop));
        proc_fifo_pop = proc_done ? repsig_done & (~ref_fifo_full): proc_stop ?
            pushed_data_sticky_sticky | (repsig_stop & (~ref_fifo_full) &
            (~pushed_data_sticky_sticky)): repsig_stop & (~ref_fifo_full);
        repsig_fifo_pop = repsig_done ? proc_done & (~ref_fifo_full): repsig_stop ? (proc_data |
            (proc_stop & (~pushed_data_sticky_sticky))) & (~ref_fifo_full): (proc_data &
            (~ref_fifo_full)) | (proc_stop & (~pushed_data_sticky_sticky));
        proc_fifo_inject_push = 1'h0;
        proc_fifo_inject_data = 16'h0;
        proc_fifo_inject_eos = 1'h0;
        set_last_pushed_data = proc_data;
        clr_last_pushed_data = proc_stop | (proc_done & repsig_done & (~ref_fifo_full));
      end :repeat_fsm_PASS_REPEAT_Output
    START: begin :repeat_fsm_START_Output
        ref_fifo_in_data = 16'h0;
        ref_fifo_in_eos = 1'h0;
        ref_fifo_push = 1'h0;
        proc_fifo_pop = 1'h0;
        repsig_fifo_pop = 1'h0;
        proc_fifo_inject_push = 1'h0;
        proc_fifo_inject_data = 16'h0;
        proc_fifo_inject_eos = 1'h0;
        set_last_pushed_data = 1'h0;
        clr_last_pushed_data = 1'h0;
      end :repeat_fsm_START_Output
    default: begin end
  endcase
end
reg_fifo_depth_0_w_17_afd_2 repsig_in_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(repsig_data_in),
  .flush(flush),
  .pop(repsig_fifo_pop),
  .push(repsig_data_in_valid),
  .rst_n(rst_n),
  .data_out(repsig_in_fifo_data_out),
  .empty(repsig_in_fifo_empty),
  .full(repsig_in_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 proc_in_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(proc_in_fifo_data_in),
  .flush(flush),
  .pop(proc_fifo_pop),
  .push(proc_fifo_push),
  .rst_n(rst_n),
  .data_out(proc_in_fifo_data_out),
  .empty(proc_in_fifo_empty),
  .full(proc_in_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 ref_out_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(ref_out_fifo_data_in),
  .flush(flush),
  .pop(ref_data_out_ready),
  .push(ref_fifo_push),
  .rst_n(rst_n),
  .data_out(ref_data_out),
  .empty(ref_out_fifo_empty),
  .full(ref_fifo_full)
);

endmodule   // Repeat

module RepeatSignalGenerator (
  input logic [16:0] base_data_in,
  input logic base_data_in_valid,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic repsig_data_out_ready,
  input logic rst_n,
  input logic [15:0] stop_lvl,
  input logic tile_en,
  output logic base_data_in_ready,
  output logic [16:0] repsig_data_out,
  output logic repsig_data_out_valid
);

typedef enum logic[1:0] {
  DONE = 2'h0,
  PASS_REPEAT = 2'h1,
  PASS_STOP = 2'h2,
  START = 2'h3
} rsg_fsm_state;
logic already_pushed_repsig_eos_sticky;
logic already_pushed_repsig_eos_was_high;
logic [15:0] base_fifo_out_data;
logic base_fifo_out_eos;
logic base_fifo_pop;
logic base_fifo_valid;
logic [0:0][16:0] base_in_fifo_data_out;
logic base_in_fifo_empty;
logic base_in_fifo_full;
logic clr_already_pushed_repsig_eos;
logic gclk;
logic repsig_fifo_full;
logic [15:0] repsig_fifo_in_data;
logic repsig_fifo_in_eos;
logic repsig_fifo_push;
logic [0:0][16:0] repsig_out_fifo_data_in;
logic repsig_out_fifo_empty;
rsg_fsm_state rsg_fsm_current_state;
rsg_fsm_state rsg_fsm_next_state;
logic seen_root_eos_sticky;
logic seen_root_eos_was_high;
assign gclk = clk & tile_en;
assign {base_fifo_out_eos, base_fifo_out_data} = base_in_fifo_data_out;
assign base_data_in_ready = ~base_in_fifo_full;
assign base_fifo_valid = ~base_in_fifo_empty;
assign repsig_out_fifo_data_in = {repsig_fifo_in_eos, repsig_fifo_in_data};
assign repsig_data_out_valid = ~repsig_out_fifo_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    seen_root_eos_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      seen_root_eos_was_high <= 1'h0;
    end
    else if (1'h0) begin
      seen_root_eos_was_high <= 1'h0;
    end
    else if ((base_fifo_out_data[9:8] == 2'h1) & base_fifo_out_eos & base_fifo_valid) begin
      seen_root_eos_was_high <= 1'h1;
    end
  end
end
assign seen_root_eos_sticky = ((base_fifo_out_data[9:8] == 2'h1) & base_fifo_out_eos & base_fifo_valid) |
    seen_root_eos_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    already_pushed_repsig_eos_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      already_pushed_repsig_eos_was_high <= 1'h0;
    end
    else if (clr_already_pushed_repsig_eos) begin
      already_pushed_repsig_eos_was_high <= 1'h0;
    end
    else if (repsig_fifo_push & (~repsig_fifo_full)) begin
      already_pushed_repsig_eos_was_high <= 1'h1;
    end
  end
end
assign already_pushed_repsig_eos_sticky = already_pushed_repsig_eos_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    rsg_fsm_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      rsg_fsm_current_state <= START;
    end
    else rsg_fsm_current_state <= rsg_fsm_next_state;
  end
end
always_comb begin
  rsg_fsm_next_state = rsg_fsm_current_state;
  unique case (rsg_fsm_current_state)
    DONE: rsg_fsm_next_state = START;
    PASS_REPEAT: begin
        if (base_fifo_out_eos & base_fifo_valid) begin
          rsg_fsm_next_state = PASS_STOP;
        end
        else rsg_fsm_next_state = PASS_REPEAT;
      end
    PASS_STOP: begin
        if (base_fifo_valid & base_fifo_out_eos & (base_fifo_out_data[9:8] == 2'h1) & (~repsig_fifo_full)) begin
          rsg_fsm_next_state = DONE;
        end
        else if (base_fifo_valid & (~base_fifo_out_eos)) begin
          rsg_fsm_next_state = PASS_REPEAT;
        end
        else rsg_fsm_next_state = PASS_STOP;
      end
    START: begin
        if (tile_en) begin
          rsg_fsm_next_state = PASS_REPEAT;
        end
        else rsg_fsm_next_state = START;
      end
    default: begin end
  endcase
end
always_comb begin
  unique case (rsg_fsm_current_state)
    DONE: begin :rsg_fsm_DONE_Output
        repsig_fifo_in_data = 16'h0;
        repsig_fifo_in_eos = 1'h0;
        repsig_fifo_push = 1'h0;
        base_fifo_pop = 1'h0;
        clr_already_pushed_repsig_eos = 1'h0;
      end :rsg_fsm_DONE_Output
    PASS_REPEAT: begin :rsg_fsm_PASS_REPEAT_Output
        repsig_fifo_in_data = 16'h1;
        repsig_fifo_in_eos = 1'h0;
        repsig_fifo_push = (~base_fifo_out_eos) & base_fifo_valid;
        clr_already_pushed_repsig_eos = 1'h1;
        base_fifo_pop = (~base_fifo_out_eos) & base_fifo_valid & (~repsig_fifo_full);
      end :rsg_fsm_PASS_REPEAT_Output
    PASS_STOP: begin :rsg_fsm_PASS_STOP_Output
        repsig_fifo_in_data = (base_fifo_out_data[9:8] == 2'h1) ? base_fifo_out_data: base_fifo_out_data;
        repsig_fifo_in_eos = 1'h1;
        repsig_fifo_push = base_fifo_out_eos & base_fifo_valid;
        clr_already_pushed_repsig_eos = 1'h0;
        base_fifo_pop = base_fifo_out_eos & base_fifo_valid & (~repsig_fifo_full);
      end :rsg_fsm_PASS_STOP_Output
    START: begin :rsg_fsm_START_Output
        repsig_fifo_in_data = 16'h0;
        repsig_fifo_in_eos = 1'h0;
        repsig_fifo_push = 1'h0;
        base_fifo_pop = 1'h0;
        clr_already_pushed_repsig_eos = 1'h0;
      end :rsg_fsm_START_Output
    default: begin end
  endcase
end
reg_fifo_depth_0_w_17_afd_2 base_in_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(base_data_in),
  .flush(flush),
  .pop(base_fifo_pop),
  .push(base_data_in_valid),
  .rst_n(rst_n),
  .data_out(base_in_fifo_data_out),
  .empty(base_in_fifo_empty),
  .full(base_in_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 repsig_out_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(repsig_out_fifo_data_in),
  .flush(flush),
  .pop(repsig_data_out_ready),
  .push(repsig_fifo_push),
  .rst_n(rst_n),
  .data_out(repsig_data_out),
  .empty(repsig_out_fifo_empty),
  .full(repsig_fifo_full)
);

endmodule   // RepeatSignalGenerator

module RepeatSignalGenerator_flat (
  input logic [15:0] RepeatSignalGenerator_inst_stop_lvl,
  input logic RepeatSignalGenerator_inst_tile_en,
  input logic [0:0] [16:0] base_data_in_f_,
  input logic base_data_in_valid_f_,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic repsig_data_out_ready_f_,
  input logic rst_n,
  output logic base_data_in_ready_f_,
  output logic [0:0] [16:0] repsig_data_out_f_,
  output logic repsig_data_out_valid_f_
);

RepeatSignalGenerator RepeatSignalGenerator_inst (
  .base_data_in(base_data_in_f_),
  .base_data_in_valid(base_data_in_valid_f_),
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .repsig_data_out_ready(repsig_data_out_ready_f_),
  .rst_n(rst_n),
  .stop_lvl(RepeatSignalGenerator_inst_stop_lvl),
  .tile_en(RepeatSignalGenerator_inst_tile_en),
  .base_data_in_ready(base_data_in_ready_f_),
  .repsig_data_out(repsig_data_out_f_),
  .repsig_data_out_valid(repsig_data_out_valid_f_)
);

endmodule   // RepeatSignalGenerator_flat

module Repeat_flat (
  input logic Repeat_inst_root,
  input logic Repeat_inst_spacc_mode,
  input logic [15:0] Repeat_inst_stop_lvl,
  input logic Repeat_inst_tile_en,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [0:0] [16:0] proc_data_in_f_,
  input logic proc_data_in_valid_f_,
  input logic ref_data_out_ready_f_,
  input logic [0:0] [16:0] repsig_data_in_f_,
  input logic repsig_data_in_valid_f_,
  input logic rst_n,
  output logic proc_data_in_ready_f_,
  output logic [0:0] [16:0] ref_data_out_f_,
  output logic ref_data_out_valid_f_,
  output logic repsig_data_in_ready_f_
);

Repeat Repeat_inst (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .proc_data_in(proc_data_in_f_),
  .proc_data_in_valid(proc_data_in_valid_f_),
  .ref_data_out_ready(ref_data_out_ready_f_),
  .repsig_data_in(repsig_data_in_f_),
  .repsig_data_in_valid(repsig_data_in_valid_f_),
  .root(Repeat_inst_root),
  .rst_n(rst_n),
  .spacc_mode(Repeat_inst_spacc_mode),
  .stop_lvl(Repeat_inst_stop_lvl),
  .tile_en(Repeat_inst_tile_en),
  .proc_data_in_ready(proc_data_in_ready_f_),
  .ref_data_out(ref_data_out_f_),
  .ref_data_out_valid(ref_data_out_valid_f_),
  .repsig_data_in_ready(repsig_data_in_ready_f_)
);

endmodule   // Repeat_flat

module crddrop (
  input logic clk,
  input logic clk_en,
  input logic [16:0] cmrg_coord_in_0,
  input logic cmrg_coord_in_0_valid,
  input logic [16:0] cmrg_coord_in_1,
  input logic cmrg_coord_in_1_valid,
  input logic cmrg_coord_out_0_ready,
  input logic cmrg_coord_out_1_ready,
  input logic cmrg_enable,
  input logic cmrg_mode,
  input logic [15:0] cmrg_stop_lvl,
  input logic flush,
  input logic rst_n,
  input logic tile_en,
  output logic cmrg_coord_in_0_ready,
  output logic cmrg_coord_in_1_ready,
  output logic [16:0] cmrg_coord_out_0,
  output logic cmrg_coord_out_0_valid,
  output logic [16:0] cmrg_coord_out_1,
  output logic cmrg_coord_out_1_valid
);

typedef enum logic[1:0] {
  DROPZERO = 2'h0,
  PROCESS = 2'h1,
  START = 2'h2
} proc_seq_state;
logic base_data_seen;
logic [16:0] base_delay;
logic base_done;
logic base_done_seen;
logic base_eos_seen;
logic base_infifo_empty;
logic base_infifo_full;
logic [15:0] base_infifo_in_data;
logic base_infifo_in_eos;
logic [16:0] base_infifo_in_packed;
logic base_infifo_in_valid;
logic [16:0] base_infifo_out_packed;
logic base_infifo_true_pop;
logic base_outfifo_empty;
logic base_outfifo_full;
logic [16:0] base_outfifo_in_packed;
logic base_outfifo_in_ready;
logic [16:0] base_outfifo_out_packed;
logic base_valid_delay;
logic both_done;
logic clr_pushed_data_lower;
logic clr_pushed_proc;
logic clr_pushed_stop_lvl;
logic cmrg_base_fifo_pop;
logic cmrg_base_fifo_push;
logic cmrg_coord_in_0_eos;
logic cmrg_coord_in_1_eos;
logic [1:0] cmrg_fifo_pop;
logic [1:0] cmrg_fifo_push;
logic cmrg_proc_fifo_pop;
logic cmrg_proc_fifo_push;
logic delay_data;
logic delay_done;
logic delay_eos;
logic delay_stop;
logic gclk;
logic proc_data_seen;
logic proc_done;
logic proc_infifo_empty;
logic proc_infifo_full;
logic [15:0] proc_infifo_in_data;
logic proc_infifo_in_eos;
logic [16:0] proc_infifo_in_packed;
logic proc_infifo_in_valid;
logic [16:0] proc_infifo_out_packed;
logic proc_outfifo_empty;
logic proc_outfifo_full;
logic [16:0] proc_outfifo_in_packed;
logic proc_outfifo_in_ready;
logic [16:0] proc_outfifo_out_packed;
proc_seq_state proc_seq_current_state;
proc_seq_state proc_seq_next_state;
logic pushed_data_sticky_sticky;
logic pushed_data_sticky_was_high;
logic pushed_proc_sticky;
logic pushed_proc_was_high;
logic pushed_stop_lvl_sticky;
logic pushed_stop_lvl_was_high;
logic pushing_done;
logic set_pushed_data_lower;
assign gclk = clk & tile_en;
assign cmrg_coord_in_0_eos = cmrg_coord_in_0[16];
assign cmrg_coord_in_1_eos = cmrg_coord_in_1[16];
assign delay_eos = base_valid_delay & base_delay[16];
assign delay_data = base_valid_delay & (~delay_eos);
assign delay_done = delay_eos & (base_delay[9:8] == 2'h1);
assign delay_stop = delay_eos & (base_delay[9:8] == 2'h0);
assign base_infifo_in_packed[16] = cmrg_coord_in_0_eos;
assign base_infifo_in_packed[15:0] = cmrg_coord_in_0[15:0];
assign base_infifo_in_eos = base_infifo_out_packed[16];
assign base_infifo_in_data = base_infifo_out_packed[15:0];
assign base_infifo_in_valid = ~base_infifo_empty;
assign cmrg_coord_in_0_ready = ~base_infifo_full;
assign proc_infifo_in_packed[16] = cmrg_coord_in_1_eos;
assign proc_infifo_in_packed[15:0] = cmrg_coord_in_1[15:0];
assign proc_infifo_in_eos = proc_infifo_out_packed[16];
assign proc_infifo_in_data = proc_infifo_out_packed[15:0];
assign proc_infifo_in_valid = ~proc_infifo_empty;
assign cmrg_coord_in_1_ready = ~proc_infifo_full;
assign base_data_seen = base_infifo_in_valid & (~base_infifo_in_eos);
assign proc_data_seen = proc_infifo_in_valid & (~proc_infifo_in_eos);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_data_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_data_sticky_was_high <= 1'h0;
    end
    else if (clr_pushed_data_lower) begin
      pushed_data_sticky_was_high <= 1'h0;
    end
    else if (set_pushed_data_lower) begin
      pushed_data_sticky_was_high <= 1'h1;
    end
  end
end
assign pushed_data_sticky_sticky = pushed_data_sticky_was_high;
assign base_eos_seen = base_infifo_in_valid & base_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h0);
assign base_done_seen = base_infifo_in_valid & base_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1);
assign base_done = base_infifo_in_valid & base_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1);
assign proc_done = proc_infifo_in_valid & proc_infifo_in_eos & (proc_infifo_in_data[9:8] == 2'h1);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_proc_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_proc_was_high <= 1'h0;
    end
    else if (clr_pushed_proc) begin
      pushed_proc_was_high <= 1'h0;
    end
    else if (cmrg_fifo_push[1]) begin
      pushed_proc_was_high <= 1'h1;
    end
  end
end
assign pushed_proc_sticky = pushed_proc_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_stop_lvl_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_stop_lvl_was_high <= 1'h0;
    end
    else if (clr_pushed_stop_lvl) begin
      pushed_stop_lvl_was_high <= 1'h0;
    end
    else if (cmrg_fifo_push[0] & base_infifo_in_valid & base_infifo_in_eos) begin
      pushed_stop_lvl_was_high <= 1'h1;
    end
  end
end
assign pushed_stop_lvl_sticky = pushed_stop_lvl_was_high;
assign both_done = base_infifo_in_valid & base_infifo_in_eos & proc_infifo_in_valid &
    proc_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1) &
    (proc_infifo_in_data[9:8] == 2'h1);
assign pushing_done = base_infifo_in_valid & base_infifo_in_eos & proc_infifo_in_valid &
    proc_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1) &
    (proc_infifo_in_data[9:8] == 2'h1) & (~base_outfifo_full) & (~proc_outfifo_full);
assign base_infifo_true_pop = cmrg_mode ? cmrg_fifo_pop[0] & (~(delay_stop & base_done_seen)):
    cmrg_fifo_pop[0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    base_delay <= 17'h0;
    base_valid_delay <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      base_delay <= 17'h0;
      base_valid_delay <= 1'h0;
    end
    else if (cmrg_fifo_pop[0] & (~(delay_done & base_done_seen))) begin
      base_delay <= ((~base_valid_delay) | base_data_seen | base_done_seen | delay_data) ?
          base_infifo_out_packed: (base_infifo_out_packed < base_delay) ? base_delay:
          base_infifo_out_packed;
      base_valid_delay <= base_infifo_in_valid;
    end
    else if (cmrg_fifo_pop[0] & delay_done & base_done_seen) begin
      base_delay <= 17'h0;
      base_valid_delay <= 1'h0;
    end
    else begin
      base_delay <= base_delay;
      base_valid_delay <= base_valid_delay;
    end
  end
end
assign base_outfifo_in_packed[16] = cmrg_mode ? base_delay[16]: base_infifo_in_eos;
assign base_outfifo_in_packed[15:0] = cmrg_mode ? base_delay[15:0]: base_infifo_in_data;
assign cmrg_coord_out_0[16] = base_outfifo_out_packed[16];
assign cmrg_coord_out_0[15:0] = base_outfifo_out_packed[15:0];
assign cmrg_coord_out_0_valid = ~base_outfifo_empty;
assign base_outfifo_in_ready = ~base_outfifo_full;
assign proc_outfifo_in_packed[16] = proc_infifo_in_eos;
assign proc_outfifo_in_packed[15:0] = proc_infifo_in_data;
assign cmrg_coord_out_1[16] = proc_outfifo_out_packed[16];
assign cmrg_coord_out_1[15:0] = proc_outfifo_out_packed[15:0];
assign cmrg_coord_out_1_valid = ~proc_outfifo_empty;
assign proc_outfifo_in_ready = ~proc_outfifo_full;
always_comb begin
  if (base_infifo_in_valid & proc_infifo_in_valid) begin
    if ((base_infifo_in_data == 16'h0) & (~base_eos_seen) & (~base_done)) begin
      cmrg_base_fifo_pop = 1'h1;
      cmrg_proc_fifo_pop = 1'h1;
      cmrg_base_fifo_push = 1'h0;
      cmrg_proc_fifo_push = 1'h0;
    end
    else if (base_outfifo_in_ready & proc_outfifo_in_ready) begin
      cmrg_base_fifo_pop = 1'h1;
      cmrg_proc_fifo_pop = 1'h1;
      cmrg_base_fifo_push = 1'h1;
      cmrg_proc_fifo_push = 1'h1;
    end
    else begin
      cmrg_base_fifo_pop = 1'h0;
      cmrg_proc_fifo_pop = 1'h0;
      cmrg_base_fifo_push = 1'h0;
      cmrg_proc_fifo_push = 1'h0;
    end
  end
  else begin
    cmrg_base_fifo_pop = 1'h0;
    cmrg_proc_fifo_pop = 1'h0;
    cmrg_base_fifo_push = 1'h0;
    cmrg_proc_fifo_push = 1'h0;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    proc_seq_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      proc_seq_current_state <= START;
    end
    else proc_seq_current_state <= proc_seq_next_state;
  end
end
always_comb begin
  proc_seq_next_state = proc_seq_current_state;
  unique case (proc_seq_current_state)
    DROPZERO: proc_seq_next_state = DROPZERO;
    PROCESS: proc_seq_next_state = PROCESS;
    START: begin
        if (tile_en & cmrg_mode) begin
          proc_seq_next_state = PROCESS;
        end
        else if (tile_en & (~cmrg_mode)) begin
          proc_seq_next_state = DROPZERO;
        end
        else proc_seq_next_state = START;
      end
    default: proc_seq_next_state = proc_seq_current_state;
  endcase
end
always_comb begin
  unique case (proc_seq_current_state)
    DROPZERO: begin :proc_seq_DROPZERO_Output
        cmrg_fifo_pop[0] = cmrg_base_fifo_pop;
        cmrg_fifo_pop[1] = cmrg_proc_fifo_pop;
        cmrg_fifo_push[0] = cmrg_base_fifo_push;
        cmrg_fifo_push[1] = cmrg_proc_fifo_push;
        clr_pushed_proc = 1'h1;
        clr_pushed_stop_lvl = 1'h1;
        set_pushed_data_lower = 1'h0;
        clr_pushed_data_lower = 1'h1;
      end :proc_seq_DROPZERO_Output
    PROCESS: begin :proc_seq_PROCESS_Output
        cmrg_fifo_pop[0] = (~base_valid_delay) | (delay_done ? proc_done & (~base_outfifo_full) &
            (~proc_outfifo_full): delay_data ? (~base_outfifo_full) & base_infifo_in_valid &
            (base_data_seen | (base_eos_seen & proc_infifo_in_valid & (~proc_infifo_in_eos)
            & (~proc_outfifo_full))): delay_eos ? base_infifo_in_valid &
            ((proc_infifo_in_valid & (~proc_infifo_in_eos)) | proc_done) & (((base_data_seen
            | base_done) & ((pushed_data_sticky_sticky & (~base_outfifo_full)) |
            (~pushed_data_sticky_sticky))) | base_eos_seen): 1'h0);
        cmrg_fifo_pop[1] = proc_done ? delay_done & (~base_outfifo_full) & (~proc_outfifo_full):
            (proc_infifo_in_valid & (~proc_infifo_in_eos)) ? base_eos_seen &
            (((~proc_outfifo_full) & delay_data & (~base_outfifo_full)) | delay_eos |
            (~base_valid_delay)): (proc_infifo_in_valid & proc_infifo_in_eos) ?
            ~proc_outfifo_full: 1'h0;
        cmrg_fifo_push[0] = delay_done ? proc_done & (~base_outfifo_full) & (~proc_outfifo_full): delay_data
            ? (~base_outfifo_full) & base_infifo_in_valid & (base_data_seen | (base_eos_seen
            & proc_infifo_in_valid & (~proc_infifo_in_eos) & (~proc_outfifo_full))):
            delay_eos ? base_infifo_in_valid & ((proc_infifo_in_valid &
            (~proc_infifo_in_eos)) | proc_done) & (base_data_seen | base_done) &
            pushed_data_sticky_sticky & (~base_outfifo_full): 1'h0;
        cmrg_fifo_push[1] = proc_done ? delay_done & (~base_outfifo_full) & (~proc_outfifo_full):
            (proc_infifo_in_valid & (~proc_infifo_in_eos)) ? base_eos_seen &
            (~proc_outfifo_full) & delay_data & (~base_outfifo_full): (proc_infifo_in_valid
            & proc_infifo_in_eos) ? ~proc_outfifo_full: 1'h0;
        clr_pushed_proc = 1'h0;
        clr_pushed_stop_lvl = 1'h0;
        set_pushed_data_lower = delay_data & (~base_outfifo_full) & base_infifo_in_valid & (base_data_seen |
            (base_eos_seen & proc_infifo_in_valid & (~proc_infifo_in_eos) &
            (~proc_outfifo_full)));
        clr_pushed_data_lower = delay_done | (delay_eos & base_infifo_in_valid & ((proc_infifo_in_valid &
            (~proc_infifo_in_eos)) | proc_done) & (base_data_seen | base_done) &
            pushed_data_sticky_sticky & (~base_outfifo_full));
      end :proc_seq_PROCESS_Output
    START: begin :proc_seq_START_Output
        cmrg_fifo_pop[0] = 1'h0;
        cmrg_fifo_pop[1] = 1'h0;
        cmrg_fifo_push[0] = 1'h0;
        cmrg_fifo_push[1] = 1'h0;
        clr_pushed_proc = 1'h1;
        clr_pushed_stop_lvl = 1'h1;
        set_pushed_data_lower = 1'h0;
        clr_pushed_data_lower = 1'h1;
      end :proc_seq_START_Output
    default: begin :proc_seq_default_Output
        cmrg_fifo_pop[0] = 1'h0;
        cmrg_fifo_pop[1] = 1'h0;
        cmrg_fifo_push[0] = 1'h0;
        cmrg_fifo_push[1] = 1'h0;
        clr_pushed_proc = 1'h1;
        clr_pushed_stop_lvl = 1'h1;
        set_pushed_data_lower = 1'h0;
        clr_pushed_data_lower = 1'h1;
      end :proc_seq_default_Output
  endcase
end
reg_fifo_depth_0_w_17_afd_2 base_infifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(base_infifo_in_packed),
  .flush(flush),
  .pop(base_infifo_true_pop),
  .push(cmrg_coord_in_0_valid),
  .rst_n(rst_n),
  .data_out(base_infifo_out_packed),
  .empty(base_infifo_empty),
  .full(base_infifo_full)
);

reg_fifo_depth_0_w_17_afd_2 proc_infifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(proc_infifo_in_packed),
  .flush(flush),
  .pop(cmrg_fifo_pop[1]),
  .push(cmrg_coord_in_1_valid),
  .rst_n(rst_n),
  .data_out(proc_infifo_out_packed),
  .empty(proc_infifo_empty),
  .full(proc_infifo_full)
);

reg_fifo_depth_0_w_17_afd_2 base_outfifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(base_outfifo_in_packed),
  .flush(flush),
  .pop(cmrg_coord_out_0_ready),
  .push(cmrg_fifo_push[0]),
  .rst_n(rst_n),
  .data_out(base_outfifo_out_packed),
  .empty(base_outfifo_empty),
  .full(base_outfifo_full)
);

reg_fifo_depth_0_w_17_afd_2 proc_outfifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(proc_outfifo_in_packed),
  .flush(flush),
  .pop(cmrg_coord_out_1_ready),
  .push(cmrg_fifo_push[1]),
  .rst_n(rst_n),
  .data_out(proc_outfifo_out_packed),
  .empty(proc_outfifo_empty),
  .full(proc_outfifo_full)
);

endmodule   // crddrop

module crddrop_flat (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] cmrg_coord_in_0_f_,
  input logic cmrg_coord_in_0_valid_f_,
  input logic [0:0] [16:0] cmrg_coord_in_1_f_,
  input logic cmrg_coord_in_1_valid_f_,
  input logic cmrg_coord_out_0_ready_f_,
  input logic cmrg_coord_out_1_ready_f_,
  input logic crddrop_inst_cmrg_enable,
  input logic crddrop_inst_cmrg_mode,
  input logic [15:0] crddrop_inst_cmrg_stop_lvl,
  input logic crddrop_inst_tile_en,
  input logic flush,
  input logic rst_n,
  output logic cmrg_coord_in_0_ready_f_,
  output logic cmrg_coord_in_1_ready_f_,
  output logic [0:0] [16:0] cmrg_coord_out_0_f_,
  output logic cmrg_coord_out_0_valid_f_,
  output logic [0:0] [16:0] cmrg_coord_out_1_f_,
  output logic cmrg_coord_out_1_valid_f_
);

crddrop crddrop_inst (
  .clk(clk),
  .clk_en(clk_en),
  .cmrg_coord_in_0(cmrg_coord_in_0_f_),
  .cmrg_coord_in_0_valid(cmrg_coord_in_0_valid_f_),
  .cmrg_coord_in_1(cmrg_coord_in_1_f_),
  .cmrg_coord_in_1_valid(cmrg_coord_in_1_valid_f_),
  .cmrg_coord_out_0_ready(cmrg_coord_out_0_ready_f_),
  .cmrg_coord_out_1_ready(cmrg_coord_out_1_ready_f_),
  .cmrg_enable(crddrop_inst_cmrg_enable),
  .cmrg_mode(crddrop_inst_cmrg_mode),
  .cmrg_stop_lvl(crddrop_inst_cmrg_stop_lvl),
  .flush(flush),
  .rst_n(rst_n),
  .tile_en(crddrop_inst_tile_en),
  .cmrg_coord_in_0_ready(cmrg_coord_in_0_ready_f_),
  .cmrg_coord_in_1_ready(cmrg_coord_in_1_ready_f_),
  .cmrg_coord_out_0(cmrg_coord_out_0_f_),
  .cmrg_coord_out_0_valid(cmrg_coord_out_0_valid_f_),
  .cmrg_coord_out_1(cmrg_coord_out_1_f_),
  .cmrg_coord_out_1_valid(cmrg_coord_out_1_valid_f_)
);

endmodule   // crddrop_flat

module crdhold (
  input logic clk,
  input logic clk_en,
  input logic [16:0] cmrg_coord_in_0,
  input logic cmrg_coord_in_0_valid,
  input logic [16:0] cmrg_coord_in_1,
  input logic cmrg_coord_in_1_valid,
  input logic cmrg_coord_out_0_ready,
  input logic cmrg_coord_out_1_ready,
  input logic cmrg_enable,
  input logic [15:0] cmrg_stop_lvl,
  input logic flush,
  input logic rst_n,
  input logic tile_en,
  output logic cmrg_coord_in_0_ready,
  output logic cmrg_coord_in_1_ready,
  output logic [16:0] cmrg_coord_out_0,
  output logic cmrg_coord_out_0_valid,
  output logic [16:0] cmrg_coord_out_1,
  output logic cmrg_coord_out_1_valid
);

typedef enum logic[1:0] {
  DATA_SEEN = 2'h0,
  DONE = 2'h1,
  START = 2'h2
} proc_seq_state;
logic base_data_seen;
logic base_done_seen;
logic base_eos_seen;
logic base_infifo_empty;
logic base_infifo_full;
logic [15:0] base_infifo_in_data;
logic base_infifo_in_eos;
logic [16:0] base_infifo_in_packed;
logic base_infifo_in_valid;
logic [16:0] base_infifo_out_packed;
logic base_outfifo_empty;
logic base_outfifo_full;
logic [16:0] base_outfifo_in_packed;
logic [16:0] base_outfifo_out_packed;
logic both_done;
logic clr_pushed_base;
logic clr_pushed_proc;
logic cmrg_coord_in_0_eos;
logic cmrg_coord_in_1_eos;
logic [1:0] cmrg_fifo_pop;
logic [1:0] cmrg_fifo_push;
logic [15:0] data_to_fifo;
logic eos_to_fifo;
logic gclk;
logic [15:0] hold_reg;
logic proc_data_seen;
logic proc_done_seen;
logic proc_eos_seen;
logic proc_infifo_empty;
logic proc_infifo_full;
logic [15:0] proc_infifo_in_data;
logic proc_infifo_in_eos;
logic [16:0] proc_infifo_in_packed;
logic proc_infifo_in_valid;
logic [16:0] proc_infifo_out_packed;
logic proc_outfifo_empty;
logic proc_outfifo_full;
logic [16:0] proc_outfifo_in_packed;
logic [16:0] proc_outfifo_out_packed;
proc_seq_state proc_seq_current_state;
proc_seq_state proc_seq_next_state;
logic pushed_base_sticky;
logic pushed_base_was_high;
logic pushed_proc_sticky;
logic pushed_proc_was_high;
logic pushing_done;
logic reg_clr;
logic reg_hold;
assign gclk = clk & tile_en;
assign cmrg_coord_in_0_eos = cmrg_coord_in_0[16];
assign cmrg_coord_in_1_eos = cmrg_coord_in_1[16];
assign base_infifo_in_packed[16] = cmrg_coord_in_0_eos;
assign base_infifo_in_packed[15:0] = cmrg_coord_in_0[15:0];
assign base_infifo_in_eos = base_infifo_out_packed[16];
assign base_infifo_in_data = base_infifo_out_packed[15:0];
assign base_infifo_in_valid = ~base_infifo_empty;
assign cmrg_coord_in_0_ready = ~base_infifo_full;
assign proc_infifo_in_packed[16] = cmrg_coord_in_1_eos;
assign proc_infifo_in_packed[15:0] = cmrg_coord_in_1[15:0];
assign proc_infifo_in_eos = proc_infifo_out_packed[16];
assign proc_infifo_in_data = proc_infifo_out_packed[15:0];
assign proc_infifo_in_valid = ~proc_infifo_empty;
assign cmrg_coord_in_1_ready = ~proc_infifo_full;
assign base_data_seen = base_infifo_in_valid & (~base_infifo_in_eos);
assign proc_data_seen = proc_infifo_in_valid & (~proc_infifo_in_eos);
assign base_eos_seen = base_infifo_in_valid & base_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h0);
assign proc_eos_seen = proc_infifo_in_valid & proc_infifo_in_eos & (proc_infifo_in_data[9:8] == 2'h0);
assign base_done_seen = base_infifo_in_valid & base_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1);
assign proc_done_seen = proc_infifo_in_valid & proc_infifo_in_eos & (proc_infifo_in_data[9:8] == 2'h1);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_proc_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_proc_was_high <= 1'h0;
    end
    else if (clr_pushed_proc) begin
      pushed_proc_was_high <= 1'h0;
    end
    else if (cmrg_fifo_push[1]) begin
      pushed_proc_was_high <= 1'h1;
    end
  end
end
assign pushed_proc_sticky = pushed_proc_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_base_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_base_was_high <= 1'h0;
    end
    else if (clr_pushed_base) begin
      pushed_base_was_high <= 1'h0;
    end
    else if (cmrg_fifo_push[0]) begin
      pushed_base_was_high <= 1'h1;
    end
  end
end
assign pushed_base_sticky = pushed_base_was_high;
assign both_done = base_infifo_in_valid & base_infifo_in_eos & proc_infifo_in_valid &
    proc_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1) &
    (proc_infifo_in_data[9:8] == 2'h1);
assign pushing_done = base_infifo_in_valid & base_infifo_in_eos & proc_infifo_in_valid &
    proc_infifo_in_eos & (base_infifo_in_data[9:8] == 2'h1) &
    (proc_infifo_in_data[9:8] == 2'h1) & (~base_outfifo_full) & (~proc_outfifo_full);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    hold_reg <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      hold_reg <= 16'h0;
    end
    else if (reg_clr) begin
      hold_reg <= 16'h0;
    end
    else if (reg_hold) begin
      hold_reg <= proc_infifo_in_data;
    end
  end
end
assign base_outfifo_in_packed[16] = base_infifo_in_eos;
assign base_outfifo_in_packed[15:0] = base_infifo_in_data;
assign cmrg_coord_out_0[16] = base_outfifo_out_packed[16];
assign cmrg_coord_out_0[15:0] = base_outfifo_out_packed[15:0];
assign cmrg_coord_out_0_valid = ~base_outfifo_empty;
assign proc_outfifo_in_packed[16] = eos_to_fifo;
assign proc_outfifo_in_packed[15:0] = data_to_fifo;
assign cmrg_coord_out_1[16] = proc_outfifo_out_packed[16];
assign cmrg_coord_out_1[15:0] = proc_outfifo_out_packed[15:0];
assign cmrg_coord_out_1_valid = ~proc_outfifo_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    proc_seq_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      proc_seq_current_state <= START;
    end
    else proc_seq_current_state <= proc_seq_next_state;
  end
end
always_comb begin
  proc_seq_next_state = proc_seq_current_state;
  unique case (proc_seq_current_state)
    DATA_SEEN: begin
        if (both_done) begin
          proc_seq_next_state = DONE;
        end
        else proc_seq_next_state = DATA_SEEN;
      end
    DONE: begin
        if ((~base_outfifo_full) & (~proc_outfifo_full)) begin
          proc_seq_next_state = START;
        end
      end
    START: begin
        if (tile_en) begin
          proc_seq_next_state = DATA_SEEN;
        end
        else proc_seq_next_state = START;
      end
    default: proc_seq_next_state = proc_seq_current_state;
  endcase
end
always_comb begin
  unique case (proc_seq_current_state)
    DATA_SEEN: begin :proc_seq_DATA_SEEN_Output
        cmrg_fifo_pop[0] = (base_eos_seen ? 1'h1 & (proc_data_seen | proc_done_seen): base_infifo_in_valid
            & (~base_infifo_in_eos) & proc_infifo_in_valid & (~proc_infifo_in_eos)) &
            (~base_outfifo_full) & (~proc_outfifo_full) & (~base_done_seen);
        cmrg_fifo_pop[1] = (proc_eos_seen ? 1'h1: base_eos_seen & (~base_outfifo_full) &
            (~proc_outfifo_full)) & (~proc_done_seen);
        cmrg_fifo_push[0] = (base_eos_seen ? 1'h1 & (proc_data_seen | proc_done_seen): base_infifo_in_valid
            & (~base_infifo_in_eos) & proc_infifo_in_valid & (~proc_infifo_in_eos)) &
            (~base_outfifo_full) & (~proc_outfifo_full) & (~base_done_seen);
        cmrg_fifo_push[1] = (base_eos_seen ? 1'h1 & (proc_data_seen | proc_done_seen): base_infifo_in_valid
            & (~base_infifo_in_eos) & proc_infifo_in_valid & (~proc_infifo_in_eos)) &
            (~base_outfifo_full) & (~proc_outfifo_full) & (~base_done_seen);
        data_to_fifo = base_infifo_in_eos ? base_infifo_in_data: proc_infifo_in_data;
        eos_to_fifo = base_infifo_in_eos;
        clr_pushed_proc = 1'h1;
        clr_pushed_base = 1'h1;
        reg_clr = 1'h1;
        reg_hold = 1'h0;
      end :proc_seq_DATA_SEEN_Output
    DONE: begin :proc_seq_DONE_Output
        cmrg_fifo_pop[0] = (~proc_outfifo_full) & (~base_outfifo_full);
        cmrg_fifo_pop[1] = (~proc_outfifo_full) & (~base_outfifo_full);
        cmrg_fifo_push[0] = (~proc_outfifo_full) & (~base_outfifo_full);
        cmrg_fifo_push[1] = (~proc_outfifo_full) & (~base_outfifo_full);
        data_to_fifo = base_infifo_in_data;
        eos_to_fifo = 1'h1;
        clr_pushed_proc = 1'h1;
        clr_pushed_base = 1'h1;
        reg_clr = 1'h1;
        reg_hold = 1'h0;
      end :proc_seq_DONE_Output
    START: begin :proc_seq_START_Output
        cmrg_fifo_pop[0] = 1'h0;
        cmrg_fifo_pop[1] = 1'h0;
        cmrg_fifo_push[0] = 1'h0;
        cmrg_fifo_push[1] = 1'h0;
        data_to_fifo = 16'h0;
        eos_to_fifo = 1'h0;
        clr_pushed_proc = 1'h1;
        clr_pushed_base = 1'h1;
        reg_clr = 1'h0;
        reg_hold = 1'h0;
      end :proc_seq_START_Output
    default: begin :proc_seq_default_Output
        cmrg_fifo_pop[0] = 1'h0;
        cmrg_fifo_pop[1] = 1'h0;
        cmrg_fifo_push[0] = 1'h0;
        cmrg_fifo_push[1] = 1'h0;
        data_to_fifo = 16'h0;
        eos_to_fifo = 1'h0;
        clr_pushed_proc = 1'h1;
        clr_pushed_base = 1'h1;
        reg_clr = 1'h0;
        reg_hold = 1'h0;
      end :proc_seq_default_Output
  endcase
end
reg_fifo_depth_0_w_17_afd_2 base_infifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(base_infifo_in_packed),
  .flush(flush),
  .pop(cmrg_fifo_pop[0]),
  .push(cmrg_coord_in_0_valid),
  .rst_n(rst_n),
  .data_out(base_infifo_out_packed),
  .empty(base_infifo_empty),
  .full(base_infifo_full)
);

reg_fifo_depth_0_w_17_afd_2 proc_infifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(proc_infifo_in_packed),
  .flush(flush),
  .pop(cmrg_fifo_pop[1]),
  .push(cmrg_coord_in_1_valid),
  .rst_n(rst_n),
  .data_out(proc_infifo_out_packed),
  .empty(proc_infifo_empty),
  .full(proc_infifo_full)
);

reg_fifo_depth_0_w_17_afd_2 base_outfifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(base_outfifo_in_packed),
  .flush(flush),
  .pop(cmrg_coord_out_0_ready),
  .push(cmrg_fifo_push[0]),
  .rst_n(rst_n),
  .data_out(base_outfifo_out_packed),
  .empty(base_outfifo_empty),
  .full(base_outfifo_full)
);

reg_fifo_depth_0_w_17_afd_2 proc_outfifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(proc_outfifo_in_packed),
  .flush(flush),
  .pop(cmrg_coord_out_1_ready),
  .push(cmrg_fifo_push[1]),
  .rst_n(rst_n),
  .data_out(proc_outfifo_out_packed),
  .empty(proc_outfifo_empty),
  .full(proc_outfifo_full)
);

endmodule   // crdhold

module crdhold_flat (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] cmrg_coord_in_0_f_,
  input logic cmrg_coord_in_0_valid_f_,
  input logic [0:0] [16:0] cmrg_coord_in_1_f_,
  input logic cmrg_coord_in_1_valid_f_,
  input logic cmrg_coord_out_0_ready_f_,
  input logic cmrg_coord_out_1_ready_f_,
  input logic crdhold_inst_cmrg_enable,
  input logic [15:0] crdhold_inst_cmrg_stop_lvl,
  input logic crdhold_inst_tile_en,
  input logic flush,
  input logic rst_n,
  output logic cmrg_coord_in_0_ready_f_,
  output logic cmrg_coord_in_1_ready_f_,
  output logic [0:0] [16:0] cmrg_coord_out_0_f_,
  output logic cmrg_coord_out_0_valid_f_,
  output logic [0:0] [16:0] cmrg_coord_out_1_f_,
  output logic cmrg_coord_out_1_valid_f_
);

crdhold crdhold_inst (
  .clk(clk),
  .clk_en(clk_en),
  .cmrg_coord_in_0(cmrg_coord_in_0_f_),
  .cmrg_coord_in_0_valid(cmrg_coord_in_0_valid_f_),
  .cmrg_coord_in_1(cmrg_coord_in_1_f_),
  .cmrg_coord_in_1_valid(cmrg_coord_in_1_valid_f_),
  .cmrg_coord_out_0_ready(cmrg_coord_out_0_ready_f_),
  .cmrg_coord_out_1_ready(cmrg_coord_out_1_ready_f_),
  .cmrg_enable(crdhold_inst_cmrg_enable),
  .cmrg_stop_lvl(crdhold_inst_cmrg_stop_lvl),
  .flush(flush),
  .rst_n(rst_n),
  .tile_en(crdhold_inst_tile_en),
  .cmrg_coord_in_0_ready(cmrg_coord_in_0_ready_f_),
  .cmrg_coord_in_1_ready(cmrg_coord_in_1_ready_f_),
  .cmrg_coord_out_0(cmrg_coord_out_0_f_),
  .cmrg_coord_out_0_valid(cmrg_coord_out_0_valid_f_),
  .cmrg_coord_out_1(cmrg_coord_out_1_f_),
  .cmrg_coord_out_1_valid(cmrg_coord_out_1_valid_f_)
);

endmodule   // crdhold_flat

module intersect_unit (
  input logic clk,
  input logic clk_en,
  input logic [16:0] coord_in_0,
  input logic coord_in_0_valid,
  input logic [16:0] coord_in_1,
  input logic coord_in_1_valid,
  input logic coord_out_ready,
  input logic flush,
  input logic joiner_op,
  input logic [16:0] pos_in_0,
  input logic pos_in_0_valid,
  input logic [16:0] pos_in_1,
  input logic pos_in_1_valid,
  input logic pos_out_0_ready,
  input logic pos_out_1_ready,
  input logic rst_n,
  input logic tile_en,
  input logic vector_reduce_mode,
  output logic coord_in_0_ready,
  output logic coord_in_1_ready,
  output logic [16:0] coord_out,
  output logic coord_out_valid,
  output logic pos_in_0_ready,
  output logic pos_in_1_ready,
  output logic [16:0] pos_out_0,
  output logic pos_out_0_valid,
  output logic [16:0] pos_out_1,
  output logic pos_out_1_valid
);

typedef enum logic[2:0] {
  ALIGN = 3'h0,
  DONE = 3'h1,
  DRAIN = 3'h2,
  IDLE = 3'h3,
  ITER = 3'h4,
  PASS_DONE = 3'h5,
  UNION = 3'h6,
  WAIT_FOR_VALID = 3'h7
} intersect_seq_state;
logic all_are_valid;
logic all_are_valid_but_no_eos;
logic all_have_eos;
logic [1:0] all_have_eos_and_all_valid;
logic any_has_eos;
logic [1:0] clr_eos_sticky;
logic [16:0] coord_fifo_in_packed;
logic [16:0] coord_fifo_out_packed;
logic coord_in_0_fifo_eos_in;
logic [16:0] coord_in_0_fifo_in;
logic coord_in_0_fifo_valid_in;
logic coord_in_1_fifo_eos_in;
logic [16:0] coord_in_1_fifo_in;
logic coord_in_1_fifo_valid_in;
logic coord_in_fifo_0_empty;
logic coord_in_fifo_0_full;
logic coord_in_fifo_1_empty;
logic coord_in_fifo_1_full;
logic [15:0] coord_to_fifo;
logic coord_to_fifo_eos;
logic coordinate_fifo_empty;
logic coordinate_fifo_full;
logic [16:0] done_token;
logic [1:0] eos_in_sticky;
logic eos_sticky_0_sticky;
logic eos_sticky_0_was_high;
logic eos_sticky_1_sticky;
logic eos_sticky_1_was_high;
logic [2:0] fifo_full;
logic fifo_push;
logic gclk;
intersect_seq_state intersect_seq_current_state;
intersect_seq_state intersect_seq_next_state;
logic [15:0] maybe;
logic [1:0] pop_fifo;
logic pos0_fifo_empty;
logic pos0_fifo_full;
logic [16:0] pos0_fifo_in_packed;
logic [16:0] pos0_fifo_out_packed;
logic pos1_fifo_empty;
logic pos1_fifo_full;
logic [16:0] pos1_fifo_in_packed;
logic [16:0] pos1_fifo_out_packed;
logic pos_in_0_fifo_eos_in;
logic [16:0] pos_in_0_fifo_in;
logic pos_in_0_fifo_valid_in;
logic pos_in_1_fifo_eos_in;
logic [16:0] pos_in_1_fifo_in;
logic pos_in_1_fifo_valid_in;
logic pos_in_fifo_0_empty;
logic pos_in_fifo_0_full;
logic pos_in_fifo_1_empty;
logic pos_in_fifo_1_full;
logic [1:0][15:0] pos_to_fifo;
logic [1:0] pos_to_fifo_eos;
logic [16:0] semi_done_token;
assign gclk = clk & tile_en;
assign coord_in_0_fifo_eos_in = coord_in_0_fifo_in[16];
assign coord_in_0_ready = ~coord_in_fifo_0_full;
assign coord_in_0_fifo_valid_in = ~coord_in_fifo_0_empty;
assign pos_in_0_fifo_eos_in = pos_in_0_fifo_in[16];
assign pos_in_0_ready = ~pos_in_fifo_0_full;
assign pos_in_0_fifo_valid_in = ~pos_in_fifo_0_empty;
assign coord_in_1_fifo_eos_in = coord_in_1_fifo_in[16];
assign coord_in_1_ready = ~coord_in_fifo_1_full;
assign coord_in_1_fifo_valid_in = ~coord_in_fifo_1_empty;
assign pos_in_1_fifo_eos_in = pos_in_1_fifo_in[16];
assign pos_in_1_ready = ~pos_in_fifo_1_full;
assign pos_in_1_fifo_valid_in = ~pos_in_fifo_1_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    eos_sticky_0_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      eos_sticky_0_was_high <= 1'h0;
    end
    else if (clr_eos_sticky[0]) begin
      eos_sticky_0_was_high <= 1'h0;
    end
    else if (coord_in_0_fifo_eos_in & coord_in_0_fifo_valid_in & pos_in_0_fifo_eos_in & pos_in_0_fifo_valid_in) begin
      eos_sticky_0_was_high <= 1'h1;
    end
  end
end
assign eos_sticky_0_sticky = (coord_in_0_fifo_eos_in & coord_in_0_fifo_valid_in & pos_in_0_fifo_eos_in &
    pos_in_0_fifo_valid_in) | eos_sticky_0_was_high;
assign eos_in_sticky[0] = eos_sticky_0_sticky;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    eos_sticky_1_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      eos_sticky_1_was_high <= 1'h0;
    end
    else if (clr_eos_sticky[1]) begin
      eos_sticky_1_was_high <= 1'h0;
    end
    else if (coord_in_1_fifo_eos_in & coord_in_1_fifo_valid_in & pos_in_1_fifo_eos_in & pos_in_1_fifo_valid_in) begin
      eos_sticky_1_was_high <= 1'h1;
    end
  end
end
assign eos_sticky_1_sticky = (coord_in_1_fifo_eos_in & coord_in_1_fifo_valid_in & pos_in_1_fifo_eos_in &
    pos_in_1_fifo_valid_in) | eos_sticky_1_was_high;
assign eos_in_sticky[1] = eos_sticky_1_sticky;
assign all_are_valid_but_no_eos = (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in,
    pos_in_1_fifo_valid_in}) & (~any_has_eos);
assign all_are_valid = &{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in,
    pos_in_1_fifo_valid_in};
assign all_have_eos_and_all_valid[0] = coord_in_0_fifo_eos_in & pos_in_0_fifo_eos_in & coord_in_0_fifo_valid_in &
    pos_in_0_fifo_valid_in;
assign all_have_eos_and_all_valid[1] = coord_in_1_fifo_eos_in & pos_in_1_fifo_eos_in & coord_in_1_fifo_valid_in &
    pos_in_1_fifo_valid_in;
assign any_has_eos = |({coord_in_0_fifo_eos_in, coord_in_1_fifo_eos_in, pos_in_0_fifo_eos_in,
    pos_in_1_fifo_eos_in} & {coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in,
    pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in});
assign all_have_eos = &({coord_in_0_fifo_eos_in, coord_in_1_fifo_eos_in, pos_in_0_fifo_eos_in,
    pos_in_1_fifo_eos_in} & {coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in,
    pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in});
assign maybe = vector_reduce_mode ? 16'h0: {6'h0, 2'h2, 8'h0};
assign semi_done_token = {1'h1, 11'h0, 1'h1, 4'h0};
assign done_token = {1'h1, 7'h0, 1'h1, 8'h0};
assign coord_fifo_in_packed[16] = coord_to_fifo_eos;
assign coord_fifo_in_packed[15:0] = coord_to_fifo;
assign coord_out[16] = coord_fifo_out_packed[16];
assign coord_out[15:0] = coord_fifo_out_packed[15:0];
assign pos0_fifo_in_packed[16] = pos_to_fifo_eos[0];
assign pos0_fifo_in_packed[15:0] = pos_to_fifo[0];
assign pos_out_0[16] = pos0_fifo_out_packed[16];
assign pos_out_0[15:0] = pos0_fifo_out_packed[15:0];
assign pos1_fifo_in_packed[16] = pos_to_fifo_eos[1];
assign pos1_fifo_in_packed[15:0] = pos_to_fifo[1];
assign pos_out_1[16] = pos1_fifo_out_packed[16];
assign pos_out_1[15:0] = pos1_fifo_out_packed[15:0];
assign fifo_full[0] = coordinate_fifo_full;
assign fifo_full[1] = pos0_fifo_full;
assign fifo_full[2] = pos1_fifo_full;
assign coord_out_valid = ~coordinate_fifo_empty;
assign pos_out_0_valid = ~pos0_fifo_empty;
assign pos_out_1_valid = ~pos1_fifo_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    intersect_seq_current_state <= IDLE;
  end
  else if (clk_en) begin
    if (flush) begin
      intersect_seq_current_state <= IDLE;
    end
    else intersect_seq_current_state <= intersect_seq_next_state;
  end
end
always_comb begin
  intersect_seq_next_state = intersect_seq_current_state;
  unique case (intersect_seq_current_state)
    ALIGN: begin
        if (all_have_eos) begin
          intersect_seq_next_state = ITER;
        end
        else intersect_seq_next_state = ALIGN;
      end
    DONE: intersect_seq_next_state = IDLE;
    DRAIN: begin
        if (vector_reduce_mode & (~(|fifo_full)) & (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in})) begin
          intersect_seq_next_state = PASS_DONE;
        end
        else if ((~vector_reduce_mode) & (~all_have_eos) & (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in})) begin
          intersect_seq_next_state = DONE;
        end
        else intersect_seq_next_state = DRAIN;
      end
    IDLE: begin
        if (all_are_valid & (joiner_op == 1'h1) & tile_en) begin
          intersect_seq_next_state = UNION;
        end
        else if (any_has_eos & (joiner_op == 1'h0) & tile_en) begin
          intersect_seq_next_state = ALIGN;
        end
        else if (all_are_valid_but_no_eos & (joiner_op == 1'h0) & tile_en) begin
          intersect_seq_next_state = ITER;
        end
        else intersect_seq_next_state = IDLE;
      end
    ITER: begin
        if (any_has_eos & (~all_have_eos)) begin
          intersect_seq_next_state = ALIGN;
        end
        else intersect_seq_next_state = ITER;
      end
    PASS_DONE: begin
        if ((~(|fifo_full)) & (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in})) begin
          intersect_seq_next_state = WAIT_FOR_VALID;
        end
        else intersect_seq_next_state = PASS_DONE;
      end
    UNION: begin
        if (&eos_in_sticky) begin
          intersect_seq_next_state = DRAIN;
        end
        else intersect_seq_next_state = UNION;
      end
    WAIT_FOR_VALID: begin
        if (vector_reduce_mode & (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in})) begin
          intersect_seq_next_state = DONE;
        end
        else if ((~all_have_eos) & (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in})) begin
          intersect_seq_next_state = DONE;
        end
        else intersect_seq_next_state = WAIT_FOR_VALID;
      end
    default: begin end
  endcase
end
always_comb begin
  unique case (intersect_seq_current_state)
    ALIGN: begin :intersect_seq_ALIGN_Output
        pop_fifo[0] = ((~eos_in_sticky[0]) & coord_in_0_fifo_valid_in & pos_in_0_fifo_valid_in) |
            (all_have_eos & (~(|fifo_full)));
        pop_fifo[1] = ((~eos_in_sticky[1]) & coord_in_1_fifo_valid_in & pos_in_1_fifo_valid_in) |
            (all_have_eos & (~(|fifo_full)));
        fifo_push = all_have_eos & (~(|fifo_full));
        clr_eos_sticky[0] = all_have_eos & (~(|fifo_full));
        clr_eos_sticky[1] = all_have_eos & (~(|fifo_full));
        coord_to_fifo = coord_in_0_fifo_in[15:0];
        pos_to_fifo[0] = pos_in_0_fifo_in[15:0];
        pos_to_fifo[1] = pos_in_1_fifo_in[15:0];
        coord_to_fifo_eos = 1'h1;
        pos_to_fifo_eos[0] = 1'h1;
        pos_to_fifo_eos[1] = 1'h1;
      end :intersect_seq_ALIGN_Output
    DONE: begin :intersect_seq_DONE_Output
        pop_fifo[0] = 1'h0;
        pop_fifo[1] = 1'h0;
        fifo_push = 1'h0;
        clr_eos_sticky[0] = 1'h1;
        clr_eos_sticky[1] = 1'h1;
        coord_to_fifo = 16'h0;
        pos_to_fifo[0] = 16'h0;
        pos_to_fifo[1] = 16'h0;
        coord_to_fifo_eos = 1'h0;
        pos_to_fifo_eos[0] = 1'h0;
        pos_to_fifo_eos[1] = 1'h0;
      end :intersect_seq_DONE_Output
    DRAIN: begin :intersect_seq_DRAIN_Output
        pop_fifo[0] = (~(|fifo_full)) & all_have_eos & (&{coord_in_0_fifo_valid_in,
            coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in});
        pop_fifo[1] = (~(|fifo_full)) & all_have_eos & (&{coord_in_0_fifo_valid_in,
            coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in});
        fifo_push = (~(|fifo_full)) & all_have_eos & (&{coord_in_0_fifo_valid_in,
            coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in});
        clr_eos_sticky[0] = 1'h0;
        clr_eos_sticky[1] = 1'h0;
        coord_to_fifo = coord_in_0_fifo_in[15:0];
        pos_to_fifo[0] = pos_in_0_fifo_in[15:0];
        pos_to_fifo[1] = pos_in_0_fifo_in[15:0];
        coord_to_fifo_eos = any_has_eos;
        pos_to_fifo_eos[0] = any_has_eos;
        pos_to_fifo_eos[1] = any_has_eos;
      end :intersect_seq_DRAIN_Output
    IDLE: begin :intersect_seq_IDLE_Output
        pop_fifo[0] = 1'h0;
        pop_fifo[1] = 1'h0;
        fifo_push = 1'h0;
        clr_eos_sticky[0] = 1'h0;
        clr_eos_sticky[1] = 1'h0;
        coord_to_fifo = 16'h0;
        pos_to_fifo[0] = 16'h0;
        pos_to_fifo[1] = 16'h0;
        coord_to_fifo_eos = 1'h0;
        pos_to_fifo_eos[0] = 1'h0;
        pos_to_fifo_eos[1] = 1'h0;
      end :intersect_seq_IDLE_Output
    ITER: begin :intersect_seq_ITER_Output
        pop_fifo[0] = (all_are_valid_but_no_eos | (all_are_valid & all_have_eos)) &
            (coord_in_0_fifo_in <= coord_in_1_fifo_in) & (~(|fifo_full));
        pop_fifo[1] = (all_are_valid_but_no_eos | (all_are_valid & all_have_eos)) &
            (coord_in_0_fifo_in >= coord_in_1_fifo_in) & (~(|fifo_full));
        fifo_push = all_are_valid & (((coord_in_0_fifo_in == coord_in_1_fifo_in) & (~any_has_eos)) |
            all_have_eos) & (~(|fifo_full));
        clr_eos_sticky[0] = all_have_eos & (~(|fifo_full));
        clr_eos_sticky[1] = all_have_eos & (~(|fifo_full));
        coord_to_fifo = coord_in_0_fifo_in[15:0];
        pos_to_fifo[0] = pos_in_0_fifo_in[15:0];
        pos_to_fifo[1] = pos_in_1_fifo_in[15:0];
        coord_to_fifo_eos = all_have_eos;
        pos_to_fifo_eos[0] = all_have_eos;
        pos_to_fifo_eos[1] = all_have_eos;
      end :intersect_seq_ITER_Output
    PASS_DONE: begin :intersect_seq_PASS_DONE_Output
        pop_fifo[0] = (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in,
            pos_in_1_fifo_valid_in}) & (coord_in_0_fifo_in == done_token);
        pop_fifo[1] = (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in, pos_in_0_fifo_valid_in,
            pos_in_1_fifo_valid_in}) & (coord_in_1_fifo_in == done_token);
        fifo_push = (~(|fifo_full)) & (&{coord_in_0_fifo_valid_in, coord_in_1_fifo_valid_in,
            pos_in_0_fifo_valid_in, pos_in_1_fifo_valid_in});
        clr_eos_sticky[0] = 1'h0;
        clr_eos_sticky[1] = 1'h0;
        coord_to_fifo = (coord_in_0_fifo_in == done_token) ? done_token[15:0]: semi_done_token[15:0];
        pos_to_fifo[0] = (coord_in_0_fifo_in == done_token) ? done_token[15:0]: semi_done_token[15:0];
        pos_to_fifo[1] = (coord_in_0_fifo_in == done_token) ? done_token[15:0]: semi_done_token[15:0];
        coord_to_fifo_eos = 1'h1;
        pos_to_fifo_eos[0] = 1'h1;
        pos_to_fifo_eos[1] = 1'h1;
      end :intersect_seq_PASS_DONE_Output
    UNION: begin :intersect_seq_UNION_Output
        pop_fifo[0] = all_are_valid & ((coord_in_0_fifo_in <= coord_in_1_fifo_in) |
            coord_in_1_fifo_eos_in) & (~(|fifo_full)) & (~coord_in_0_fifo_eos_in);
        pop_fifo[1] = all_are_valid & ((coord_in_0_fifo_in >= coord_in_1_fifo_in) |
            coord_in_0_fifo_eos_in) & (~(|fifo_full)) & (~coord_in_1_fifo_eos_in);
        fifo_push = all_are_valid & (~(|fifo_full)) & (~all_have_eos);
        clr_eos_sticky[0] = 1'h0;
        clr_eos_sticky[1] = 1'h0;
        coord_to_fifo = pop_fifo[0] ? coord_in_0_fifo_in[15:0]: coord_in_1_fifo_in[15:0];
        pos_to_fifo[0] = pop_fifo[0] ? pos_in_0_fifo_in[15:0]: maybe;
        pos_to_fifo[1] = pop_fifo[1] ? pos_in_1_fifo_in[15:0]: maybe;
        coord_to_fifo_eos = 1'h0;
        pos_to_fifo_eos[0] = (~vector_reduce_mode) & (~pop_fifo[0]);
        pos_to_fifo_eos[1] = (~vector_reduce_mode) & (~pop_fifo[1]);
      end :intersect_seq_UNION_Output
    WAIT_FOR_VALID: begin :intersect_seq_WAIT_FOR_VALID_Output
        pop_fifo[0] = 1'h0;
        pop_fifo[1] = 1'h0;
        fifo_push = 1'h0;
        clr_eos_sticky[0] = 1'h0;
        clr_eos_sticky[1] = 1'h0;
        coord_to_fifo = 16'h0;
        pos_to_fifo[0] = 16'h0;
        pos_to_fifo[1] = 16'h0;
        coord_to_fifo_eos = 1'h0;
        pos_to_fifo_eos[0] = 1'h0;
        pos_to_fifo_eos[1] = 1'h0;
      end :intersect_seq_WAIT_FOR_VALID_Output
    default: begin end
  endcase
end
reg_fifo_depth_0_w_17_afd_2 coord_in_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(coord_in_0),
  .flush(flush),
  .pop(pop_fifo[0]),
  .push(coord_in_0_valid),
  .rst_n(rst_n),
  .data_out(coord_in_0_fifo_in),
  .empty(coord_in_fifo_0_empty),
  .full(coord_in_fifo_0_full)
);

reg_fifo_depth_0_w_17_afd_2 pos_in_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(pos_in_0),
  .flush(flush),
  .pop(pop_fifo[0]),
  .push(pos_in_0_valid),
  .rst_n(rst_n),
  .data_out(pos_in_0_fifo_in),
  .empty(pos_in_fifo_0_empty),
  .full(pos_in_fifo_0_full)
);

reg_fifo_depth_0_w_17_afd_2 coord_in_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(coord_in_1),
  .flush(flush),
  .pop(pop_fifo[1]),
  .push(coord_in_1_valid),
  .rst_n(rst_n),
  .data_out(coord_in_1_fifo_in),
  .empty(coord_in_fifo_1_empty),
  .full(coord_in_fifo_1_full)
);

reg_fifo_depth_0_w_17_afd_2 pos_in_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(pos_in_1),
  .flush(flush),
  .pop(pop_fifo[1]),
  .push(pos_in_1_valid),
  .rst_n(rst_n),
  .data_out(pos_in_1_fifo_in),
  .empty(pos_in_fifo_1_empty),
  .full(pos_in_fifo_1_full)
);

reg_fifo_depth_0_w_17_afd_2 coordinate_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(coord_fifo_in_packed),
  .flush(flush),
  .pop(coord_out_ready),
  .push(fifo_push),
  .rst_n(rst_n),
  .data_out(coord_fifo_out_packed),
  .empty(coordinate_fifo_empty),
  .full(coordinate_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 pos0_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(pos0_fifo_in_packed),
  .flush(flush),
  .pop(pos_out_0_ready),
  .push(fifo_push),
  .rst_n(rst_n),
  .data_out(pos0_fifo_out_packed),
  .empty(pos0_fifo_empty),
  .full(pos0_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 pos1_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(pos1_fifo_in_packed),
  .flush(flush),
  .pop(pos_out_1_ready),
  .push(fifo_push),
  .rst_n(rst_n),
  .data_out(pos1_fifo_out_packed),
  .empty(pos1_fifo_empty),
  .full(pos1_fifo_full)
);

endmodule   // intersect_unit

module intersect_unit_flat (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] coord_in_0_f_,
  input logic coord_in_0_valid_f_,
  input logic [0:0] [16:0] coord_in_1_f_,
  input logic coord_in_1_valid_f_,
  input logic coord_out_ready_f_,
  input logic flush,
  input logic intersect_unit_inst_joiner_op,
  input logic intersect_unit_inst_tile_en,
  input logic intersect_unit_inst_vector_reduce_mode,
  input logic [0:0] [16:0] pos_in_0_f_,
  input logic pos_in_0_valid_f_,
  input logic [0:0] [16:0] pos_in_1_f_,
  input logic pos_in_1_valid_f_,
  input logic pos_out_0_ready_f_,
  input logic pos_out_1_ready_f_,
  input logic rst_n,
  output logic coord_in_0_ready_f_,
  output logic coord_in_1_ready_f_,
  output logic [0:0] [16:0] coord_out_f_,
  output logic coord_out_valid_f_,
  output logic pos_in_0_ready_f_,
  output logic pos_in_1_ready_f_,
  output logic [0:0] [16:0] pos_out_0_f_,
  output logic pos_out_0_valid_f_,
  output logic [0:0] [16:0] pos_out_1_f_,
  output logic pos_out_1_valid_f_
);

intersect_unit intersect_unit_inst (
  .clk(clk),
  .clk_en(clk_en),
  .coord_in_0(coord_in_0_f_),
  .coord_in_0_valid(coord_in_0_valid_f_),
  .coord_in_1(coord_in_1_f_),
  .coord_in_1_valid(coord_in_1_valid_f_),
  .coord_out_ready(coord_out_ready_f_),
  .flush(flush),
  .joiner_op(intersect_unit_inst_joiner_op),
  .pos_in_0(pos_in_0_f_),
  .pos_in_0_valid(pos_in_0_valid_f_),
  .pos_in_1(pos_in_1_f_),
  .pos_in_1_valid(pos_in_1_valid_f_),
  .pos_out_0_ready(pos_out_0_ready_f_),
  .pos_out_1_ready(pos_out_1_ready_f_),
  .rst_n(rst_n),
  .tile_en(intersect_unit_inst_tile_en),
  .vector_reduce_mode(intersect_unit_inst_vector_reduce_mode),
  .coord_in_0_ready(coord_in_0_ready_f_),
  .coord_in_1_ready(coord_in_1_ready_f_),
  .coord_out(coord_out_f_),
  .coord_out_valid(coord_out_valid_f_),
  .pos_in_0_ready(pos_in_0_ready_f_),
  .pos_in_1_ready(pos_in_1_ready_f_),
  .pos_out_0(pos_out_0_f_),
  .pos_out_0_valid(pos_out_0_valid_f_),
  .pos_out_1(pos_out_1_f_),
  .pos_out_1_valid(pos_out_1_valid_f_)
);

endmodule   // intersect_unit_flat

module reduce_pe_cluster (
  input logic bit0,
  input logic bit1,
  input logic bit2,
  input logic clk,
  input logic clk_en,
  input logic [16:0] data0,
  input logic data0_valid,
  input logic [16:0] data1,
  input logic data1_valid,
  input logic [16:0] data2,
  input logic data2_valid,
  input logic flush,
  input logic pe_dense_mode,
  input logic pe_in_external,
  input logic [83:0] pe_onyxpeintf_inst,
  input logic [2:0] pe_sparse_num_inputs,
  input logic pe_tile_en,
  input logic [16:0] reduce_data_in,
  input logic reduce_data_in_valid,
  input logic reduce_data_out_ready,
  input logic [15:0] reduce_default_value,
  input logic [15:0] reduce_stop_lvl,
  input logic reduce_tile_en,
  input logic res_ready,
  input logic rst_n,
  input logic tile_en,
  output logic data0_ready,
  output logic data1_ready,
  output logic data2_ready,
  output logic [15:0] pe_onyxpeintf_O2,
  output logic [15:0] pe_onyxpeintf_O3,
  output logic [15:0] pe_onyxpeintf_O4,
  output logic reduce_data_in_ready,
  output logic [16:0] reduce_data_out,
  output logic reduce_data_out_valid,
  output logic [16:0] res,
  output logic res_p,
  output logic res_valid
);

logic gclk;
logic [16:0] pe_data0;
logic [16:0] pe_data1;
logic [16:0] pe_data2;
logic [16:0] pe_data_to_reduce;
logic [16:0] pe_res;
logic [16:0] reduce_data_to_pe0;
logic [16:0] reduce_data_to_pe1;
assign gclk = clk & tile_en;
assign res = pe_res;
assign pe_data0 = pe_in_external ? data0: reduce_data_to_pe0;
assign pe_data1 = pe_in_external ? data1: reduce_data_to_pe1;
assign pe_data2 = data2;
assign pe_data_to_reduce = pe_res;
reg_cr reduce (
  .clk(gclk),
  .clk_en(clk_en),
  .data_from_pe(pe_data_to_reduce),
  .data_in(reduce_data_in),
  .data_in_valid(reduce_data_in_valid),
  .data_out_ready(reduce_data_out_ready),
  .default_value(reduce_default_value),
  .flush(flush),
  .rst_n(rst_n),
  .stop_lvl(reduce_stop_lvl),
  .tile_en(reduce_tile_en),
  .data_in_ready(reduce_data_in_ready),
  .data_out(reduce_data_out),
  .data_out_valid(reduce_data_out_valid),
  .data_to_pe0(reduce_data_to_pe0),
  .data_to_pe1(reduce_data_to_pe1)
);

PE_onyx pe (
  .bit0(bit0),
  .bit1(bit1),
  .bit2(bit2),
  .clk(gclk),
  .clk_en(clk_en),
  .data0(pe_data0),
  .data0_valid(data0_valid),
  .data1(pe_data1),
  .data1_valid(data1_valid),
  .data2(pe_data2),
  .data2_valid(data2_valid),
  .dense_mode(pe_dense_mode),
  .flush(flush),
  .onyxpeintf_inst(pe_onyxpeintf_inst),
  .res_ready(res_ready),
  .rst_n(rst_n),
  .sparse_num_inputs(pe_sparse_num_inputs),
  .tile_en(pe_tile_en),
  .data0_ready(data0_ready),
  .data1_ready(data1_ready),
  .data2_ready(data2_ready),
  .onyxpeintf_O2(pe_onyxpeintf_O2),
  .onyxpeintf_O3(pe_onyxpeintf_O3),
  .onyxpeintf_O4(pe_onyxpeintf_O4),
  .res(pe_res),
  .res_p(res_p),
  .res_valid(res_valid)
);

endmodule   // reduce_pe_cluster

module reduce_pe_cluster_flat (
  input logic bit0_f_,
  input logic bit1_f_,
  input logic bit2_f_,
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] data0_f_,
  input logic data0_valid_f_,
  input logic [0:0] [16:0] data1_f_,
  input logic data1_valid_f_,
  input logic [0:0] [16:0] data2_f_,
  input logic data2_valid_f_,
  input logic flush,
  input logic [0:0] [16:0] reduce_data_in_f_,
  input logic reduce_data_in_valid_f_,
  input logic reduce_data_out_ready_f_,
  input logic reduce_pe_cluster_inst_pe_dense_mode,
  input logic reduce_pe_cluster_inst_pe_in_external,
  input logic [83:0] reduce_pe_cluster_inst_pe_onyxpeintf_inst,
  input logic [2:0] reduce_pe_cluster_inst_pe_sparse_num_inputs,
  input logic reduce_pe_cluster_inst_pe_tile_en,
  input logic [15:0] reduce_pe_cluster_inst_reduce_default_value,
  input logic [15:0] reduce_pe_cluster_inst_reduce_stop_lvl,
  input logic reduce_pe_cluster_inst_reduce_tile_en,
  input logic reduce_pe_cluster_inst_tile_en,
  input logic res_ready_f_,
  input logic rst_n,
  output logic data0_ready_f_,
  output logic data1_ready_f_,
  output logic data2_ready_f_,
  output logic reduce_data_in_ready_f_,
  output logic [0:0] [16:0] reduce_data_out_f_,
  output logic reduce_data_out_valid_f_,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O2,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O3,
  output logic [15:0] reduce_pe_cluster_inst_pe_onyxpeintf_O4,
  output logic [0:0] [16:0] res_f_,
  output logic res_p_f_,
  output logic res_valid_f_
);

reduce_pe_cluster reduce_pe_cluster_inst (
  .bit0(bit0_f_),
  .bit1(bit1_f_),
  .bit2(bit2_f_),
  .clk(clk),
  .clk_en(clk_en),
  .data0(data0_f_),
  .data0_valid(data0_valid_f_),
  .data1(data1_f_),
  .data1_valid(data1_valid_f_),
  .data2(data2_f_),
  .data2_valid(data2_valid_f_),
  .flush(flush),
  .pe_dense_mode(reduce_pe_cluster_inst_pe_dense_mode),
  .pe_in_external(reduce_pe_cluster_inst_pe_in_external),
  .pe_onyxpeintf_inst(reduce_pe_cluster_inst_pe_onyxpeintf_inst),
  .pe_sparse_num_inputs(reduce_pe_cluster_inst_pe_sparse_num_inputs),
  .pe_tile_en(reduce_pe_cluster_inst_pe_tile_en),
  .reduce_data_in(reduce_data_in_f_),
  .reduce_data_in_valid(reduce_data_in_valid_f_),
  .reduce_data_out_ready(reduce_data_out_ready_f_),
  .reduce_default_value(reduce_pe_cluster_inst_reduce_default_value),
  .reduce_stop_lvl(reduce_pe_cluster_inst_reduce_stop_lvl),
  .reduce_tile_en(reduce_pe_cluster_inst_reduce_tile_en),
  .res_ready(res_ready_f_),
  .rst_n(rst_n),
  .tile_en(reduce_pe_cluster_inst_tile_en),
  .data0_ready(data0_ready_f_),
  .data1_ready(data1_ready_f_),
  .data2_ready(data2_ready_f_),
  .pe_onyxpeintf_O2(reduce_pe_cluster_inst_pe_onyxpeintf_O2),
  .pe_onyxpeintf_O3(reduce_pe_cluster_inst_pe_onyxpeintf_O3),
  .pe_onyxpeintf_O4(reduce_pe_cluster_inst_pe_onyxpeintf_O4),
  .reduce_data_in_ready(reduce_data_in_ready_f_),
  .reduce_data_out(reduce_data_out_f_),
  .reduce_data_out_valid(reduce_data_out_valid_f_),
  .res(res_f_),
  .res_p(res_p_f_),
  .res_valid(res_valid_f_)
);

endmodule   // reduce_pe_cluster_flat

module reg_cr (
  input logic clk,
  input logic clk_en,
  input logic [16:0] data_from_pe,
  input logic [16:0] data_in,
  input logic data_in_valid,
  input logic data_out_ready,
  input logic [15:0] default_value,
  input logic flush,
  input logic rst_n,
  input logic [15:0] stop_lvl,
  input logic tile_en,
  output logic data_in_ready,
  output logic [16:0] data_out,
  output logic data_out_valid,
  output logic [16:0] data_to_pe0,
  output logic [16:0] data_to_pe1
);

typedef enum logic[2:0] {
  ACCUM = 3'h0,
  DONE = 3'h1,
  OUTPUT = 3'h2,
  START = 3'h3,
  STOP_PASS = 3'h4
} accum_seq_state;
logic [15:0] accum_reg;
accum_seq_state accum_seq_current_state;
accum_seq_state accum_seq_next_state;
logic [15:0] data_to_fifo;
logic gclk;
logic [16:0] infifo_in_packed;
logic [15:0] infifo_out_data;
logic infifo_out_eos;
logic [16:0] infifo_out_packed;
logic infifo_out_valid;
logic infifo_pop;
logic infifo_push;
logic input_fifo_empty;
logic input_fifo_full;
logic outfifo_full;
logic outfifo_in_eos;
logic [16:0] outfifo_in_packed;
logic [16:0] outfifo_out_packed;
logic outfifo_pop;
logic outfifo_push;
logic output_fifo_empty;
logic reg_clr;
logic update_accum_reg;
assign gclk = clk & tile_en;
assign data_in_ready = ~input_fifo_full;
assign infifo_in_packed[16:0] = data_in;
assign infifo_out_eos = infifo_out_packed[16];
assign infifo_out_data = infifo_out_packed[15:0];
assign infifo_push = data_in_valid;
assign infifo_out_valid = ~input_fifo_empty;
assign data_to_pe0 = infifo_out_packed;
assign data_to_pe1[15:0] = accum_reg;
assign data_to_pe1[16] = 1'h0;
assign outfifo_in_packed[16] = outfifo_in_eos;
assign outfifo_in_packed[15:0] = data_to_fifo;
assign data_out = outfifo_out_packed[16:0];
assign data_out_valid = ~output_fifo_empty;
assign outfifo_pop = data_out_ready;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    accum_reg <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      accum_reg <= 16'h0;
    end
    else if (reg_clr) begin
      accum_reg <= default_value;
    end
    else if (update_accum_reg) begin
      accum_reg <= data_from_pe[15:0];
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    accum_seq_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      accum_seq_current_state <= START;
    end
    else accum_seq_current_state <= accum_seq_next_state;
  end
end
always_comb begin
  accum_seq_next_state = accum_seq_current_state;
  unique case (accum_seq_current_state)
    ACCUM: begin
        if (infifo_out_valid & infifo_out_eos) begin
          accum_seq_next_state = OUTPUT;
        end
        else accum_seq_next_state = ACCUM;
      end
    DONE: begin
        if (~outfifo_full) begin
          accum_seq_next_state = START;
        end
        else accum_seq_next_state = DONE;
      end
    OUTPUT: begin
        if (~outfifo_full) begin
          accum_seq_next_state = STOP_PASS;
        end
        else accum_seq_next_state = OUTPUT;
      end
    START: begin
        if (infifo_out_valid & (~infifo_out_eos)) begin
          accum_seq_next_state = ACCUM;
        end
        else if (infifo_out_valid & infifo_out_eos & (infifo_out_data[9:8] == 2'h1)) begin
          accum_seq_next_state = DONE;
        end
        else if (infifo_out_valid & infifo_out_eos & (infifo_out_data[9:8] == 2'h0)) begin
          accum_seq_next_state = OUTPUT;
        end
        else accum_seq_next_state = START;
      end
    STOP_PASS: begin
        if (~outfifo_full) begin
          accum_seq_next_state = START;
        end
        else accum_seq_next_state = STOP_PASS;
      end
    default: accum_seq_next_state = accum_seq_current_state;
  endcase
end
always_comb begin
  unique case (accum_seq_current_state)
    ACCUM: begin :accum_seq_ACCUM_Output
        infifo_pop = infifo_out_valid & (~infifo_out_eos);
        outfifo_push = 1'h0;
        data_to_fifo = 16'h0;
        outfifo_in_eos = 1'h0;
        reg_clr = 1'h0;
        update_accum_reg = infifo_out_valid & (~infifo_out_eos);
      end :accum_seq_ACCUM_Output
    DONE: begin :accum_seq_DONE_Output
        infifo_pop = ~outfifo_full;
        outfifo_push = ~outfifo_full;
        reg_clr = 1'h1;
        data_to_fifo = infifo_out_data;
        outfifo_in_eos = infifo_out_eos;
        update_accum_reg = 1'h0;
      end :accum_seq_DONE_Output
    OUTPUT: begin :accum_seq_OUTPUT_Output
        infifo_pop = 1'h0;
        outfifo_push = ~outfifo_full;
        reg_clr = 1'h0;
        data_to_fifo = accum_reg;
        outfifo_in_eos = 1'h0;
        update_accum_reg = 1'h0;
      end :accum_seq_OUTPUT_Output
    START: begin :accum_seq_START_Output
        infifo_pop = 1'h0;
        outfifo_push = 1'h0;
        data_to_fifo = 16'h0;
        outfifo_in_eos = 1'h0;
        reg_clr = 1'h0;
        update_accum_reg = 1'h0;
      end :accum_seq_START_Output
    STOP_PASS: begin :accum_seq_STOP_PASS_Output
        infifo_pop = (~outfifo_full) & infifo_out_valid & infifo_out_eos & (infifo_out_data[9:8] ==
            2'h0);
        outfifo_push = (~outfifo_full) & infifo_out_valid & infifo_out_eos & (infifo_out_data[9:8] ==
            2'h0) & (infifo_out_data[7:0] > 8'h0);
        reg_clr = 1'h1;
        data_to_fifo = infifo_out_data - 16'h1;
        outfifo_in_eos = 1'h1;
        update_accum_reg = 1'h0;
      end :accum_seq_STOP_PASS_Output
    default: begin :accum_seq_default_Output
        infifo_pop = 1'h0;
        outfifo_push = 1'h0;
        data_to_fifo = 16'h0;
        outfifo_in_eos = 1'h0;
        reg_clr = 1'h0;
        update_accum_reg = 1'h0;
      end :accum_seq_default_Output
  endcase
end
reg_fifo_depth_0_w_17_afd_2 input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(infifo_in_packed),
  .flush(flush),
  .pop(infifo_pop),
  .push(infifo_push),
  .rst_n(rst_n),
  .data_out(infifo_out_packed),
  .empty(input_fifo_empty),
  .full(input_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 output_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(outfifo_in_packed),
  .flush(flush),
  .pop(outfifo_pop),
  .push(outfifo_push),
  .rst_n(rst_n),
  .data_out(outfifo_out_packed),
  .empty(output_fifo_empty),
  .full(outfifo_full)
);

endmodule   // reg_cr

module reg_fifo_depth_0_w_17_afd_2 (
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

assign data_out = data_in;
assign valid = push;
assign empty = ~push;
assign full = ~pop;
assign almost_full = ~pop;
endmodule   // reg_fifo_depth_0_w_17_afd_2

module reg_fifo_depth_2_w_17_afd_2 (
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
assign almost_full = num_items >= 2'h0;
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
endmodule   // reg_fifo_depth_2_w_17_afd_2

