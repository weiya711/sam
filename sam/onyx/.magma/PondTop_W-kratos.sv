module PondTop (
  input logic [31:0] CONFIG_SPACE_0,
  input logic [31:0] CONFIG_SPACE_1,
  input logic [31:0] CONFIG_SPACE_10,
  input logic [31:0] CONFIG_SPACE_11,
  input logic [31:0] CONFIG_SPACE_12,
  input logic [31:0] CONFIG_SPACE_13,
  input logic [31:0] CONFIG_SPACE_14,
  input logic [31:0] CONFIG_SPACE_15,
  input logic [29:0] CONFIG_SPACE_16,
  input logic [31:0] CONFIG_SPACE_2,
  input logic [31:0] CONFIG_SPACE_3,
  input logic [31:0] CONFIG_SPACE_4,
  input logic [31:0] CONFIG_SPACE_5,
  input logic [31:0] CONFIG_SPACE_6,
  input logic [31:0] CONFIG_SPACE_7,
  input logic [31:0] CONFIG_SPACE_8,
  input logic [31:0] CONFIG_SPACE_9,
  input logic [0:0] [16:0] PondTop_input_width_17_num_0,
  input logic [0:0] [16:0] PondTop_input_width_17_num_1,
  input logic clk,
  input logic clk_en,
  input logic [7:0] config_addr_in,
  input logic [31:0] config_data_in,
  input logic config_en,
  input logic config_read,
  input logic config_write,
  input logic flush,
  input logic rst_n,
  input logic tile_en,
  output logic [0:0] [16:0] PondTop_output_width_17_num_0,
  output logic [0:0] [16:0] PondTop_output_width_17_num_1,
  output logic PondTop_output_width_1_num_0,
  output logic PondTop_output_width_1_num_1,
  output logic [0:0] [31:0] config_data_out
);

logic [541:0] CONFIG_SPACE;
logic [15:0] config_data_in_shrt;
logic [0:0][15:0] config_data_out_shrt;
logic [4:0] config_seq_addr_out;
logic config_seq_clk_en;
logic [0:0][0:0][15:0] config_seq_rd_data_stg;
logic config_seq_ren_out;
logic config_seq_wen_out;
logic [0:0][15:0] config_seq_wr_data;
logic gclk;
logic mem_ctrl_strg_ub_thin_PondTop_flat_clk;
logic [0:0][16:0] mem_ctrl_strg_ub_thin_PondTop_flat_data_out_f_0;
logic [0:0][16:0] mem_ctrl_strg_ub_thin_PondTop_flat_data_out_f_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_data_from_strg_lifted;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_data_to_strg_lifted;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr2;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_0;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_1;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_0;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_1;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_2;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_3;
logic [2:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality;
logic [1:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_3;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_3;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_rd_addr_out_lifted;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr2;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_0;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_1;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_0;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_1;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_2;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_3;
logic [2:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality;
logic [1:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_3;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_3;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_ren_to_strg_lifted;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_tmp0_rdaddr_lifted;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_tmp0_rden_lifted;
logic mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_wen_to_strg_lifted;
logic [4:0] mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_wr_addr_out_lifted;
logic mem_ctrl_strg_ub_thin_PondTop_flat_valid_out_f_b_0;
logic mem_ctrl_strg_ub_thin_PondTop_flat_valid_out_f_b_1;
logic memory_0_clk_en;
logic [15:0] memory_0_data_in_p0;
logic [15:0] memory_0_data_out_p0;
logic [15:0] memory_0_data_out_p1;
logic [4:0] memory_0_read_addr_p0;
logic [4:0] memory_0_read_addr_p1;
logic memory_0_read_enable_p0;
logic memory_0_read_enable_p1;
logic [4:0] memory_0_write_addr_p0;
logic memory_0_write_enable_p0;
logic mode;
assign mode = 1'h0;
assign gclk = clk & tile_en;
assign mem_ctrl_strg_ub_thin_PondTop_flat_clk = gclk;
always_comb begin
  PondTop_output_width_17_num_0 = 17'h0;
  if (1'h0 == mode) begin
    PondTop_output_width_17_num_0 = mem_ctrl_strg_ub_thin_PondTop_flat_data_out_f_0;
  end
  else PondTop_output_width_17_num_0 = 17'h0;
end
always_comb begin
  PondTop_output_width_17_num_1 = 17'h0;
  if (1'h0 == mode) begin
    PondTop_output_width_17_num_1 = mem_ctrl_strg_ub_thin_PondTop_flat_data_out_f_1;
  end
  else PondTop_output_width_17_num_1 = 17'h0;
end
always_comb begin
  PondTop_output_width_1_num_0 = 1'h0;
  if (1'h0 == mode) begin
    PondTop_output_width_1_num_0 = mem_ctrl_strg_ub_thin_PondTop_flat_valid_out_f_b_0;
  end
  else PondTop_output_width_1_num_0 = 1'h0;
end
always_comb begin
  PondTop_output_width_1_num_1 = 1'h0;
  if (1'h0 == mode) begin
    PondTop_output_width_1_num_1 = mem_ctrl_strg_ub_thin_PondTop_flat_valid_out_f_b_1;
  end
  else PondTop_output_width_1_num_1 = 1'h0;
end
always_comb begin
  memory_0_data_in_p0 = 16'h0;
  memory_0_write_addr_p0 = 5'h0;
  memory_0_write_enable_p0 = 1'h0;
  memory_0_read_addr_p0 = 5'h0;
  memory_0_read_enable_p0 = 1'h0;
  if (|config_en) begin
    memory_0_data_in_p0 = config_seq_wr_data;
    memory_0_write_addr_p0 = config_seq_addr_out;
    memory_0_write_enable_p0 = config_seq_wen_out;
    memory_0_read_addr_p0 = config_seq_addr_out;
    memory_0_read_enable_p0 = config_seq_ren_out;
  end
  else if (1'h0 == mode) begin
    memory_0_data_in_p0 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_data_to_strg_lifted;
    memory_0_write_addr_p0 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_wr_addr_out_lifted;
    memory_0_write_enable_p0 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_wen_to_strg_lifted;
    memory_0_read_addr_p0 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_tmp0_rdaddr_lifted;
    memory_0_read_enable_p0 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_tmp0_rden_lifted;
  end
end
always_comb begin
  config_seq_rd_data_stg = memory_0_data_out_p0;
end
always_comb begin
  memory_0_read_addr_p1 = 5'h0;
  memory_0_read_enable_p1 = 1'h0;
  if (1'h0 == mode) begin
    memory_0_read_addr_p1 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_rd_addr_out_lifted;
    memory_0_read_enable_p1 = mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_ren_to_strg_lifted;
  end
end
always_comb begin
  mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_data_from_strg_lifted = memory_0_data_out_p1;
end
assign config_data_in_shrt = config_data_in[15:0];
assign config_data_out[0] = 32'(config_data_out_shrt[0]);
assign config_seq_clk_en = clk_en | (|config_en);
assign memory_0_clk_en = clk_en | (|config_en);
assign {mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_3, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_3, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_3, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_3, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_3, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_0, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_1, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_2, mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_3} = CONFIG_SPACE[541:0];
assign CONFIG_SPACE[31:0] = CONFIG_SPACE_0;
assign CONFIG_SPACE[63:32] = CONFIG_SPACE_1;
assign CONFIG_SPACE[95:64] = CONFIG_SPACE_2;
assign CONFIG_SPACE[127:96] = CONFIG_SPACE_3;
assign CONFIG_SPACE[159:128] = CONFIG_SPACE_4;
assign CONFIG_SPACE[191:160] = CONFIG_SPACE_5;
assign CONFIG_SPACE[223:192] = CONFIG_SPACE_6;
assign CONFIG_SPACE[255:224] = CONFIG_SPACE_7;
assign CONFIG_SPACE[287:256] = CONFIG_SPACE_8;
assign CONFIG_SPACE[319:288] = CONFIG_SPACE_9;
assign CONFIG_SPACE[351:320] = CONFIG_SPACE_10;
assign CONFIG_SPACE[383:352] = CONFIG_SPACE_11;
assign CONFIG_SPACE[415:384] = CONFIG_SPACE_12;
assign CONFIG_SPACE[447:416] = CONFIG_SPACE_13;
assign CONFIG_SPACE[479:448] = CONFIG_SPACE_14;
assign CONFIG_SPACE[511:480] = CONFIG_SPACE_15;
assign CONFIG_SPACE[541:512] = CONFIG_SPACE_16;
strg_ub_thin_PondTop_flat mem_ctrl_strg_ub_thin_PondTop_flat (
  .clk(mem_ctrl_strg_ub_thin_PondTop_flat_clk),
  .clk_en(clk_en),
  .data_in_f_0(PondTop_input_width_17_num_0),
  .data_in_f_1(PondTop_input_width_17_num_1),
  .flush(flush),
  .rst_n(rst_n),
  .strg_ub_thin_PondTop_inst_data_from_strg_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_data_from_strg_lifted),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_0),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_1),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_0),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_1),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_3(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_3),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_0),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_1),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_0),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_1),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_3(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_3),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_0),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_1),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_0),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_1),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_2),
  .strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_3(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_3),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_0),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_1),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_0),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_1),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_3(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_3),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_0),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_1),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_0),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_1),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_3(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_3),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_0),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_1),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_0(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_0),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_1(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_1),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_2(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_2),
  .strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_3(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_3),
  .data_out_f_0(mem_ctrl_strg_ub_thin_PondTop_flat_data_out_f_0),
  .data_out_f_1(mem_ctrl_strg_ub_thin_PondTop_flat_data_out_f_1),
  .strg_ub_thin_PondTop_inst_data_to_strg_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_data_to_strg_lifted),
  .strg_ub_thin_PondTop_inst_rd_addr_out_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_rd_addr_out_lifted),
  .strg_ub_thin_PondTop_inst_ren_to_strg_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_ren_to_strg_lifted),
  .strg_ub_thin_PondTop_inst_tmp0_rdaddr_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_tmp0_rdaddr_lifted),
  .strg_ub_thin_PondTop_inst_tmp0_rden_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_tmp0_rden_lifted),
  .strg_ub_thin_PondTop_inst_wen_to_strg_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_wen_to_strg_lifted),
  .strg_ub_thin_PondTop_inst_wr_addr_out_lifted(mem_ctrl_strg_ub_thin_PondTop_flat_strg_ub_thin_PondTop_inst_wr_addr_out_lifted),
  .valid_out_f_b_0(mem_ctrl_strg_ub_thin_PondTop_flat_valid_out_f_b_0),
  .valid_out_f_b_1(mem_ctrl_strg_ub_thin_PondTop_flat_valid_out_f_b_1)
);

sram_dp__0 memory_0 (
  .clk(gclk),
  .clk_en(memory_0_clk_en),
  .data_in_p0(memory_0_data_in_p0),
  .flush(flush),
  .read_addr_p0(memory_0_read_addr_p0),
  .read_addr_p1(memory_0_read_addr_p1),
  .read_enable_p0(memory_0_read_enable_p0),
  .read_enable_p1(memory_0_read_enable_p1),
  .write_addr_p0(memory_0_write_addr_p0),
  .write_enable_p0(memory_0_write_enable_p0),
  .data_out_p0(memory_0_data_out_p0),
  .data_out_p1(memory_0_data_out_p1)
);

storage_config_seq_1_16_16 config_seq (
  .clk(gclk),
  .clk_en(config_seq_clk_en),
  .config_addr_in(config_addr_in),
  .config_data_in(config_data_in_shrt),
  .config_en(config_en),
  .config_rd(config_read),
  .config_wr(config_write),
  .flush(flush),
  .rd_data_stg(config_seq_rd_data_stg),
  .rst_n(rst_n),
  .addr_out(config_seq_addr_out),
  .rd_data_out(config_data_out_shrt),
  .ren_out(config_seq_ren_out),
  .wen_out(config_seq_wen_out),
  .wr_data(config_seq_wr_data)
);

endmodule   // PondTop

module PondTop_W (
  input logic [31:0] CONFIG_SPACE_0,
  input logic [31:0] CONFIG_SPACE_1,
  input logic [31:0] CONFIG_SPACE_10,
  input logic [31:0] CONFIG_SPACE_11,
  input logic [31:0] CONFIG_SPACE_12,
  input logic [31:0] CONFIG_SPACE_13,
  input logic [31:0] CONFIG_SPACE_14,
  input logic [31:0] CONFIG_SPACE_15,
  input logic [29:0] CONFIG_SPACE_16,
  input logic [31:0] CONFIG_SPACE_2,
  input logic [31:0] CONFIG_SPACE_3,
  input logic [31:0] CONFIG_SPACE_4,
  input logic [31:0] CONFIG_SPACE_5,
  input logic [31:0] CONFIG_SPACE_6,
  input logic [31:0] CONFIG_SPACE_7,
  input logic [31:0] CONFIG_SPACE_8,
  input logic [31:0] CONFIG_SPACE_9,
  input logic [0:0] [16:0] PondTop_input_width_17_num_0,
  input logic [0:0] [16:0] PondTop_input_width_17_num_1,
  input logic clk,
  input logic clk_en,
  input logic [7:0] config_addr_in,
  input logic [31:0] config_data_in,
  input logic config_en,
  input logic config_read,
  input logic config_write,
  input logic flush,
  input logic rst_n,
  input logic tile_en,
  output logic [0:0] [16:0] PondTop_output_width_17_num_0,
  output logic [0:0] [16:0] PondTop_output_width_17_num_1,
  output logic PondTop_output_width_1_num_0,
  output logic PondTop_output_width_1_num_1,
  output logic [0:0] [31:0] config_data_out
);

PondTop PondTop (
  .CONFIG_SPACE_0(CONFIG_SPACE_0),
  .CONFIG_SPACE_1(CONFIG_SPACE_1),
  .CONFIG_SPACE_10(CONFIG_SPACE_10),
  .CONFIG_SPACE_11(CONFIG_SPACE_11),
  .CONFIG_SPACE_12(CONFIG_SPACE_12),
  .CONFIG_SPACE_13(CONFIG_SPACE_13),
  .CONFIG_SPACE_14(CONFIG_SPACE_14),
  .CONFIG_SPACE_15(CONFIG_SPACE_15),
  .CONFIG_SPACE_16(CONFIG_SPACE_16),
  .CONFIG_SPACE_2(CONFIG_SPACE_2),
  .CONFIG_SPACE_3(CONFIG_SPACE_3),
  .CONFIG_SPACE_4(CONFIG_SPACE_4),
  .CONFIG_SPACE_5(CONFIG_SPACE_5),
  .CONFIG_SPACE_6(CONFIG_SPACE_6),
  .CONFIG_SPACE_7(CONFIG_SPACE_7),
  .CONFIG_SPACE_8(CONFIG_SPACE_8),
  .CONFIG_SPACE_9(CONFIG_SPACE_9),
  .PondTop_input_width_17_num_0(PondTop_input_width_17_num_0),
  .PondTop_input_width_17_num_1(PondTop_input_width_17_num_1),
  .clk(clk),
  .clk_en(clk_en),
  .config_addr_in(config_addr_in),
  .config_data_in(config_data_in),
  .config_en(config_en),
  .config_read(config_read),
  .config_write(config_write),
  .flush(flush),
  .rst_n(rst_n),
  .tile_en(tile_en),
  .PondTop_output_width_17_num_0(PondTop_output_width_17_num_0),
  .PondTop_output_width_17_num_1(PondTop_output_width_17_num_1),
  .PondTop_output_width_1_num_0(PondTop_output_width_1_num_0),
  .PondTop_output_width_1_num_1(PondTop_output_width_1_num_1),
  .config_data_out(config_data_out)
);

endmodule   // PondTop_W

module addr_gen_4_16_dual_config_2 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic mux_sel_msb_init,
  input logic restart,
  input logic rst_n,
  input logic [15:0] starting_addr,
  input logic [15:0] starting_addr2,
  input logic step,
  input logic [3:0] [15:0] strides,
  input logic [1:0] [15:0] strides2,
  output logic [15:0] addr_out,
  output logic starting_addr_comp
);

logic [15:0] calc_addr;
logic [15:0] cur_stride;
logic [15:0] current_addr;
logic [15:0] flush_addr;
logic [1:0] mux_sel_iter1;
logic mux_sel_iter2;
logic mux_sel_msb;
logic [15:0] restart_addr;
logic [15:0] strt_addr;
assign starting_addr_comp = starting_addr2 < starting_addr;
assign mux_sel_iter1 = mux_sel[1:0];
assign mux_sel_iter2 = mux_sel[0];
assign mux_sel_msb = mux_sel[2];
assign flush_addr = mux_sel_msb_init ? starting_addr2: starting_addr;
assign strt_addr = mux_sel_msb ? starting_addr2: starting_addr;
assign restart_addr = (~mux_sel_msb) ? starting_addr2: starting_addr;
assign cur_stride = mux_sel_msb ? strides2[mux_sel_iter2]: strides[mux_sel_iter1];
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= flush_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= restart_addr;
      end
      else current_addr <= current_addr + cur_stride;
    end
  end
end
endmodule   // addr_gen_4_16_dual_config_2

module addr_gen_4_5_dual_config_2 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic mux_sel_msb_init,
  input logic restart,
  input logic rst_n,
  input logic [4:0] starting_addr,
  input logic [4:0] starting_addr2,
  input logic step,
  input logic [3:0] [4:0] strides,
  input logic [1:0] [4:0] strides2,
  output logic [4:0] addr_out,
  output logic starting_addr_comp
);

logic [4:0] calc_addr;
logic [4:0] cur_stride;
logic [4:0] current_addr;
logic [4:0] flush_addr;
logic [1:0] mux_sel_iter1;
logic mux_sel_iter2;
logic mux_sel_msb;
logic [4:0] restart_addr;
logic [4:0] strt_addr;
assign starting_addr_comp = starting_addr2 < starting_addr;
assign mux_sel_iter1 = mux_sel[1:0];
assign mux_sel_iter2 = mux_sel[0];
assign mux_sel_msb = mux_sel[2];
assign flush_addr = mux_sel_msb_init ? starting_addr2: starting_addr;
assign strt_addr = mux_sel_msb ? starting_addr2: starting_addr;
assign restart_addr = (~mux_sel_msb) ? starting_addr2: starting_addr;
assign cur_stride = mux_sel_msb ? strides2[mux_sel_iter2]: strides[mux_sel_iter1];
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 5'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= flush_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= restart_addr;
      end
      else current_addr <= current_addr + cur_stride;
    end
  end
end
endmodule   // addr_gen_4_5_dual_config_2

module for_loop_dual_config_4_2_16 #(
  parameter CONFIG_WIDTH = 5'h10,
  parameter ITERATOR_SUPPORT = 3'h4,
  parameter ITERATOR_SUPPORT2 = 2'h2
)
(
  input logic clk,
  input logic clk_en,
  input logic [2:0] dimensionality,
  input logic [1:0] dimensionality2,
  input logic flush,
  input logic mux_sel_msb_init,
  input logic [3:0] [15:0] ranges,
  input logic [1:0] [15:0] ranges2,
  input logic rst_n,
  input logic step,
  output logic [2:0] mux_sel_out,
  output logic restart
);

logic [3:0] clear;
logic [2:0] cur_dimensionality;
logic [15:0] cur_range;
logic [3:0][15:0] dim_counter;
logic done;
logic [3:0] inc;
logic [15:0] inced_cnt;
logic [3:0] max_value;
logic maxed_value;
logic [1:0] mux_sel;
logic [1:0] mux_sel_iter1;
logic mux_sel_iter2;
logic mux_sel_msb;
logic mux_sel_msb_r;
assign mux_sel_msb = mux_sel_msb_r;
assign cur_dimensionality = mux_sel_msb ? 3'(dimensionality2): dimensionality;
assign mux_sel_iter1 = mux_sel[1:0];
assign mux_sel_iter2 = mux_sel[0];
assign mux_sel_out = {mux_sel_msb, mux_sel};
assign inced_cnt = dim_counter[mux_sel] + 16'h1;
assign cur_range = mux_sel_msb ? ranges2[mux_sel_iter2]: ranges[mux_sel_iter1];
assign maxed_value = (dim_counter[mux_sel] == cur_range) & inc[mux_sel];
always_comb begin
  mux_sel = 2'h0;
  done = 1'h0;
  if (~done) begin
    if ((~max_value[0]) & (cur_dimensionality > 3'h0)) begin
      mux_sel = 2'h0;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[1]) & (cur_dimensionality > 3'h1)) begin
      mux_sel = 2'h1;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[2]) & (cur_dimensionality > 3'h2)) begin
      mux_sel = 2'h2;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[3]) & (cur_dimensionality > 3'h3)) begin
      mux_sel = 2'h3;
      done = 1'h1;
    end
  end
end
always_comb begin
  clear[0] = 1'h0;
  if (((mux_sel > 2'h0) | (~done)) & step) begin
    clear[0] = 1'h1;
  end
end
always_comb begin
  inc[0] = 1'h0;
  if ((5'h0 == 5'h0) & step & (cur_dimensionality > 3'h0)) begin
    inc[0] = 1'h1;
  end
  else if ((mux_sel == 2'h0) & step & (cur_dimensionality > 3'h0)) begin
    inc[0] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[0] <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[0] <= 16'h0;
    end
    else if (clear[0]) begin
      dim_counter[0] <= 16'h0;
    end
    else if (inc[0]) begin
      dim_counter[0] <= inced_cnt;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    max_value[0] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      max_value[0] <= 1'h0;
    end
    else if (clear[0]) begin
      max_value[0] <= 1'h0;
    end
    else if (inc[0]) begin
      max_value[0] <= maxed_value;
    end
  end
end
always_comb begin
  clear[1] = 1'h0;
  if (((mux_sel > 2'h1) | (~done)) & step) begin
    clear[1] = 1'h1;
  end
end
always_comb begin
  inc[1] = 1'h0;
  if ((5'h1 == 5'h0) & step & (cur_dimensionality > 3'h1)) begin
    inc[1] = 1'h1;
  end
  else if ((mux_sel == 2'h1) & step & (cur_dimensionality > 3'h1)) begin
    inc[1] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[1] <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[1] <= 16'h0;
    end
    else if (clear[1]) begin
      dim_counter[1] <= 16'h0;
    end
    else if (inc[1]) begin
      dim_counter[1] <= inced_cnt;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    max_value[1] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      max_value[1] <= 1'h0;
    end
    else if (clear[1]) begin
      max_value[1] <= 1'h0;
    end
    else if (inc[1]) begin
      max_value[1] <= maxed_value;
    end
  end
end
always_comb begin
  clear[2] = 1'h0;
  if (((mux_sel > 2'h2) | (~done)) & step) begin
    clear[2] = 1'h1;
  end
end
always_comb begin
  inc[2] = 1'h0;
  if ((5'h2 == 5'h0) & step & (cur_dimensionality > 3'h2)) begin
    inc[2] = 1'h1;
  end
  else if ((mux_sel == 2'h2) & step & (cur_dimensionality > 3'h2)) begin
    inc[2] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[2] <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[2] <= 16'h0;
    end
    else if (clear[2]) begin
      dim_counter[2] <= 16'h0;
    end
    else if (inc[2]) begin
      dim_counter[2] <= inced_cnt;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    max_value[2] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      max_value[2] <= 1'h0;
    end
    else if (clear[2]) begin
      max_value[2] <= 1'h0;
    end
    else if (inc[2]) begin
      max_value[2] <= maxed_value;
    end
  end
end
always_comb begin
  clear[3] = 1'h0;
  if (((mux_sel > 2'h3) | (~done)) & step) begin
    clear[3] = 1'h1;
  end
end
always_comb begin
  inc[3] = 1'h0;
  if ((5'h3 == 5'h0) & step & (cur_dimensionality > 3'h3)) begin
    inc[3] = 1'h1;
  end
  else if ((mux_sel == 2'h3) & step & (cur_dimensionality > 3'h3)) begin
    inc[3] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[3] <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[3] <= 16'h0;
    end
    else if (clear[3]) begin
      dim_counter[3] <= 16'h0;
    end
    else if (inc[3]) begin
      dim_counter[3] <= inced_cnt;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    max_value[3] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      max_value[3] <= 1'h0;
    end
    else if (clear[3]) begin
      max_value[3] <= 1'h0;
    end
    else if (inc[3]) begin
      max_value[3] <= maxed_value;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    mux_sel_msb_r <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      mux_sel_msb_r <= mux_sel_msb_init;
    end
    else if (restart) begin
      mux_sel_msb_r <= ~mux_sel_msb_r;
    end
  end
end
assign restart = step & (~done);
endmodule   // for_loop_dual_config_4_2_16

module sched_gen_4_16_dual_config_2 (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic enable,
  input logic enable2,
  input logic finished,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic rst_n,
  input logic [15:0] sched_addr_gen_starting_addr,
  input logic [15:0] sched_addr_gen_starting_addr2,
  input logic [15:0] sched_addr_gen_strides2_0,
  input logic [15:0] sched_addr_gen_strides2_1,
  input logic [15:0] sched_addr_gen_strides_0,
  input logic [15:0] sched_addr_gen_strides_1,
  input logic [15:0] sched_addr_gen_strides_2,
  input logic [15:0] sched_addr_gen_strides_3,
  output logic mux_sel_msb_init,
  output logic valid_output
);

logic [15:0] addr_out;
logic cur_enable;
logic cur_valid_gate;
logic mux_sel_msb_init_w;
logic sched_addr_gen_starting_addr_comp;
logic [3:0][15:0] sched_addr_gen_strides;
logic [1:0][15:0] sched_addr_gen_strides2;
logic [1:0] valid_gate;
logic [1:0] valid_gate_inv;
logic valid_out;
assign cur_valid_gate = valid_gate[mux_sel[2]];
assign valid_gate = ~valid_gate_inv;
assign cur_enable = mux_sel[2] ? enable2: enable;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_gate_inv <= 2'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_gate_inv <= 2'h0;
    end
    else if (finished) begin
      valid_gate_inv[mux_sel[2]] <= 1'h1;
    end
  end
end
always_comb begin
  if (enable & enable2) begin
    mux_sel_msb_init_w = sched_addr_gen_starting_addr_comp;
  end
  else if (enable & (~enable2)) begin
    mux_sel_msb_init_w = 1'h0;
  end
  else if ((~enable) & enable2) begin
    mux_sel_msb_init_w = 1'h1;
  end
  else mux_sel_msb_init_w = 1'h0;
end
assign mux_sel_msb_init = mux_sel_msb_init_w;
always_comb begin
  valid_out = (cycle_count == addr_out) & cur_valid_gate & cur_enable;
end
always_comb begin
  valid_output = valid_out;
end
assign sched_addr_gen_strides2[0] = sched_addr_gen_strides2_0;
assign sched_addr_gen_strides2[1] = sched_addr_gen_strides2_1;
assign sched_addr_gen_strides[0] = sched_addr_gen_strides_0;
assign sched_addr_gen_strides[1] = sched_addr_gen_strides_1;
assign sched_addr_gen_strides[2] = sched_addr_gen_strides_2;
assign sched_addr_gen_strides[3] = sched_addr_gen_strides_3;
addr_gen_4_16_dual_config_2 sched_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(mux_sel),
  .mux_sel_msb_init(mux_sel_msb_init_w),
  .restart(finished),
  .rst_n(rst_n),
  .starting_addr(sched_addr_gen_starting_addr),
  .starting_addr2(sched_addr_gen_starting_addr2),
  .step(valid_out),
  .strides(sched_addr_gen_strides),
  .strides2(sched_addr_gen_strides2),
  .addr_out(addr_out),
  .starting_addr_comp(sched_addr_gen_starting_addr_comp)
);

endmodule   // sched_gen_4_16_dual_config_2

module sram_dp__0 (
  input logic clk,
  input logic clk_en,
  input logic [15:0] data_in_p0,
  input logic flush,
  input logic [4:0] read_addr_p0,
  input logic [4:0] read_addr_p1,
  input logic read_enable_p0,
  input logic read_enable_p1,
  input logic [4:0] write_addr_p0,
  input logic write_enable_p0,
  output logic [15:0] data_out_p0,
  output logic [15:0] data_out_p1
);

logic [15:0] data_array [31:0];

always_ff @(posedge clk) begin
  if (clk_en) begin
    if (write_enable_p0 == 1'h1) begin
      data_array[write_addr_p0] <= data_in_p0;
    end
  end
end
assign data_out_p0 = data_array[read_addr_p0];
always_comb begin
  data_out_p1 = data_array[read_addr_p1];
end
endmodule   // sram_dp__0

module storage_config_seq_1_16_16 (
  input logic clk,
  input logic clk_en,
  input logic [7:0] config_addr_in,
  input logic [15:0] config_data_in,
  input logic config_en,
  input logic config_rd,
  input logic config_wr,
  input logic flush,
  input logic [0:0][0:0] [15:0] rd_data_stg,
  input logic rst_n,
  output logic [4:0] addr_out,
  output logic [0:0] [15:0] rd_data_out,
  output logic ren_out,
  output logic wen_out,
  output logic [0:0] [15:0] wr_data
);

assign addr_out = config_addr_in[4:0];
assign wr_data[0] = config_data_in;
assign rd_data_out[0] = rd_data_stg[0];
assign wen_out = config_wr;
assign ren_out = config_rd;
endmodule   // storage_config_seq_1_16_16

module strg_ub_thin_PondTop (
  input logic clk,
  input logic clk_en,
  input logic [15:0] data_from_strg,
  input logic [1:0] [16:0] data_in,
  input logic flush,
  input logic [4:0] in2regfile_0_addr_gen_starting_addr,
  input logic [4:0] in2regfile_0_addr_gen_starting_addr2,
  input logic [4:0] in2regfile_0_addr_gen_strides2_0,
  input logic [4:0] in2regfile_0_addr_gen_strides2_1,
  input logic [4:0] in2regfile_0_addr_gen_strides_0,
  input logic [4:0] in2regfile_0_addr_gen_strides_1,
  input logic [4:0] in2regfile_0_addr_gen_strides_2,
  input logic [4:0] in2regfile_0_addr_gen_strides_3,
  input logic [2:0] in2regfile_0_for_loop_dimensionality,
  input logic [1:0] in2regfile_0_for_loop_dimensionality2,
  input logic [15:0] in2regfile_0_for_loop_ranges2_0,
  input logic [15:0] in2regfile_0_for_loop_ranges2_1,
  input logic [15:0] in2regfile_0_for_loop_ranges_0,
  input logic [15:0] in2regfile_0_for_loop_ranges_1,
  input logic [15:0] in2regfile_0_for_loop_ranges_2,
  input logic [15:0] in2regfile_0_for_loop_ranges_3,
  input logic in2regfile_0_sched_gen_enable,
  input logic in2regfile_0_sched_gen_enable2,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_starting_addr,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_starting_addr2,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_strides2_0,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_strides2_1,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_strides_0,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_strides_1,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_strides_2,
  input logic [15:0] in2regfile_0_sched_gen_sched_addr_gen_strides_3,
  input logic [4:0] regfile2out_0_addr_gen_starting_addr,
  input logic [4:0] regfile2out_0_addr_gen_starting_addr2,
  input logic [4:0] regfile2out_0_addr_gen_strides2_0,
  input logic [4:0] regfile2out_0_addr_gen_strides2_1,
  input logic [4:0] regfile2out_0_addr_gen_strides_0,
  input logic [4:0] regfile2out_0_addr_gen_strides_1,
  input logic [4:0] regfile2out_0_addr_gen_strides_2,
  input logic [4:0] regfile2out_0_addr_gen_strides_3,
  input logic [2:0] regfile2out_0_for_loop_dimensionality,
  input logic [1:0] regfile2out_0_for_loop_dimensionality2,
  input logic [15:0] regfile2out_0_for_loop_ranges2_0,
  input logic [15:0] regfile2out_0_for_loop_ranges2_1,
  input logic [15:0] regfile2out_0_for_loop_ranges_0,
  input logic [15:0] regfile2out_0_for_loop_ranges_1,
  input logic [15:0] regfile2out_0_for_loop_ranges_2,
  input logic [15:0] regfile2out_0_for_loop_ranges_3,
  input logic regfile2out_0_sched_gen_enable,
  input logic regfile2out_0_sched_gen_enable2,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_starting_addr,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_starting_addr2,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_strides2_0,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_strides2_1,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_strides_0,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_strides_1,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_strides_2,
  input logic [15:0] regfile2out_0_sched_gen_sched_addr_gen_strides_3,
  input logic rst_n,
  output logic [1:0] [16:0] data_out,
  output logic [15:0] data_to_strg,
  output logic [4:0] rd_addr_out,
  output logic ren_to_strg,
  output logic [4:0] tmp0_rdaddr,
  output logic tmp0_rden,
  output logic [1:0] valid_out,
  output logic wen_to_strg,
  output logic [4:0] wr_addr_out
);

logic [15:0] cycle_count;
logic [1:0][15:0] data_in_thin;
logic [1:0][15:0] data_out_int;
logic [1:0][15:0] data_out_int_thin;
logic in2regfile_0_addr_gen_mux_sel_msb_init;
logic [3:0][4:0] in2regfile_0_addr_gen_strides;
logic [1:0][4:0] in2regfile_0_addr_gen_strides2;
logic in2regfile_0_for_loop_mux_sel_msb_init;
logic [2:0] in2regfile_0_for_loop_mux_sel_out;
logic [3:0][15:0] in2regfile_0_for_loop_ranges;
logic [1:0][15:0] in2regfile_0_for_loop_ranges2;
logic in2regfile_0_for_loop_restart;
logic in2regfile_0_sched_gen_mux_sel_msb_init;
logic in2regfile_0_sched_gen_valid_output;
logic read;
logic [4:0] read_addr;
logic read_mux_sel_msb;
logic regfile2out_0_addr_gen_mux_sel_msb_init;
logic [3:0][4:0] regfile2out_0_addr_gen_strides;
logic [1:0][4:0] regfile2out_0_addr_gen_strides2;
logic regfile2out_0_for_loop_mux_sel_msb_init;
logic [2:0] regfile2out_0_for_loop_mux_sel_out;
logic [3:0][15:0] regfile2out_0_for_loop_ranges;
logic [1:0][15:0] regfile2out_0_for_loop_ranges2;
logic regfile2out_0_for_loop_restart;
logic regfile2out_0_sched_gen_mux_sel_msb_init;
logic regfile2out_0_sched_gen_valid_output;
logic [1:0] valid_out_int;
logic write;
logic [4:0] write_addr;
logic write_mux_sel_msb;
assign data_in_thin[0] = data_in[0][15:0];
assign data_in_thin[1] = data_in[1][15:0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    cycle_count <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      cycle_count <= 16'h0;
    end
    else cycle_count <= cycle_count + 16'h1;
  end
end
assign valid_out_int[0] = read & (~read_mux_sel_msb);
assign valid_out_int[1] = read & read_mux_sel_msb;
assign data_out_int_thin = data_out_int;
assign data_out[0][15:0] = data_out_int_thin[0];
assign data_out[0][16] = 1'h0;
assign data_out[1][15:0] = data_out_int_thin[1];
assign data_out[1][16] = 1'h0;
assign valid_out = valid_out_int;
assign write = in2regfile_0_sched_gen_valid_output;
assign in2regfile_0_for_loop_mux_sel_msb_init = in2regfile_0_sched_gen_mux_sel_msb_init;
assign in2regfile_0_addr_gen_mux_sel_msb_init = in2regfile_0_sched_gen_mux_sel_msb_init;
assign write_mux_sel_msb = in2regfile_0_for_loop_mux_sel_out[2];
assign read = regfile2out_0_sched_gen_valid_output;
assign regfile2out_0_for_loop_mux_sel_msb_init = regfile2out_0_sched_gen_mux_sel_msb_init;
assign regfile2out_0_addr_gen_mux_sel_msb_init = regfile2out_0_sched_gen_mux_sel_msb_init;
assign read_mux_sel_msb = regfile2out_0_for_loop_mux_sel_out[2];
assign wen_to_strg = |write;
assign ren_to_strg = |read;
assign data_out_int[0] = data_from_strg;
assign data_out_int[1] = data_from_strg;
assign wr_addr_out = write_addr[4:0];
assign data_to_strg = data_in_thin[write_mux_sel_msb];
assign rd_addr_out = read_addr[4:0];
assign tmp0_rdaddr = 5'h0;
assign tmp0_rden = 1'h0;
assign in2regfile_0_for_loop_ranges[0] = in2regfile_0_for_loop_ranges_0;
assign in2regfile_0_for_loop_ranges[1] = in2regfile_0_for_loop_ranges_1;
assign in2regfile_0_for_loop_ranges[2] = in2regfile_0_for_loop_ranges_2;
assign in2regfile_0_for_loop_ranges[3] = in2regfile_0_for_loop_ranges_3;
assign in2regfile_0_for_loop_ranges2[0] = in2regfile_0_for_loop_ranges2_0;
assign in2regfile_0_for_loop_ranges2[1] = in2regfile_0_for_loop_ranges2_1;
assign in2regfile_0_addr_gen_strides2[0] = in2regfile_0_addr_gen_strides2_0;
assign in2regfile_0_addr_gen_strides2[1] = in2regfile_0_addr_gen_strides2_1;
assign in2regfile_0_addr_gen_strides[0] = in2regfile_0_addr_gen_strides_0;
assign in2regfile_0_addr_gen_strides[1] = in2regfile_0_addr_gen_strides_1;
assign in2regfile_0_addr_gen_strides[2] = in2regfile_0_addr_gen_strides_2;
assign in2regfile_0_addr_gen_strides[3] = in2regfile_0_addr_gen_strides_3;
assign regfile2out_0_for_loop_ranges[0] = regfile2out_0_for_loop_ranges_0;
assign regfile2out_0_for_loop_ranges[1] = regfile2out_0_for_loop_ranges_1;
assign regfile2out_0_for_loop_ranges[2] = regfile2out_0_for_loop_ranges_2;
assign regfile2out_0_for_loop_ranges[3] = regfile2out_0_for_loop_ranges_3;
assign regfile2out_0_for_loop_ranges2[0] = regfile2out_0_for_loop_ranges2_0;
assign regfile2out_0_for_loop_ranges2[1] = regfile2out_0_for_loop_ranges2_1;
assign regfile2out_0_addr_gen_strides2[0] = regfile2out_0_addr_gen_strides2_0;
assign regfile2out_0_addr_gen_strides2[1] = regfile2out_0_addr_gen_strides2_1;
assign regfile2out_0_addr_gen_strides[0] = regfile2out_0_addr_gen_strides_0;
assign regfile2out_0_addr_gen_strides[1] = regfile2out_0_addr_gen_strides_1;
assign regfile2out_0_addr_gen_strides[2] = regfile2out_0_addr_gen_strides_2;
assign regfile2out_0_addr_gen_strides[3] = regfile2out_0_addr_gen_strides_3;
for_loop_dual_config_4_2_16 in2regfile_0_for_loop (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(in2regfile_0_for_loop_dimensionality),
  .dimensionality2(in2regfile_0_for_loop_dimensionality2),
  .flush(flush),
  .mux_sel_msb_init(in2regfile_0_for_loop_mux_sel_msb_init),
  .ranges(in2regfile_0_for_loop_ranges),
  .ranges2(in2regfile_0_for_loop_ranges2),
  .rst_n(rst_n),
  .step(write),
  .mux_sel_out(in2regfile_0_for_loop_mux_sel_out),
  .restart(in2regfile_0_for_loop_restart)
);

addr_gen_4_5_dual_config_2 in2regfile_0_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(in2regfile_0_for_loop_mux_sel_out),
  .mux_sel_msb_init(in2regfile_0_addr_gen_mux_sel_msb_init),
  .restart(in2regfile_0_for_loop_restart),
  .rst_n(rst_n),
  .starting_addr(in2regfile_0_addr_gen_starting_addr),
  .starting_addr2(in2regfile_0_addr_gen_starting_addr2),
  .step(write),
  .strides(in2regfile_0_addr_gen_strides),
  .strides2(in2regfile_0_addr_gen_strides2),
  .addr_out(write_addr)
);

sched_gen_4_16_dual_config_2 in2regfile_0_sched_gen (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(in2regfile_0_sched_gen_enable),
  .enable2(in2regfile_0_sched_gen_enable2),
  .finished(in2regfile_0_for_loop_restart),
  .flush(flush),
  .mux_sel(in2regfile_0_for_loop_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_starting_addr(in2regfile_0_sched_gen_sched_addr_gen_starting_addr),
  .sched_addr_gen_starting_addr2(in2regfile_0_sched_gen_sched_addr_gen_starting_addr2),
  .sched_addr_gen_strides2_0(in2regfile_0_sched_gen_sched_addr_gen_strides2_0),
  .sched_addr_gen_strides2_1(in2regfile_0_sched_gen_sched_addr_gen_strides2_1),
  .sched_addr_gen_strides_0(in2regfile_0_sched_gen_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(in2regfile_0_sched_gen_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(in2regfile_0_sched_gen_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(in2regfile_0_sched_gen_sched_addr_gen_strides_3),
  .mux_sel_msb_init(in2regfile_0_sched_gen_mux_sel_msb_init),
  .valid_output(in2regfile_0_sched_gen_valid_output)
);

for_loop_dual_config_4_2_16 regfile2out_0_for_loop (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(regfile2out_0_for_loop_dimensionality),
  .dimensionality2(regfile2out_0_for_loop_dimensionality2),
  .flush(flush),
  .mux_sel_msb_init(regfile2out_0_for_loop_mux_sel_msb_init),
  .ranges(regfile2out_0_for_loop_ranges),
  .ranges2(regfile2out_0_for_loop_ranges2),
  .rst_n(rst_n),
  .step(read),
  .mux_sel_out(regfile2out_0_for_loop_mux_sel_out),
  .restart(regfile2out_0_for_loop_restart)
);

addr_gen_4_5_dual_config_2 regfile2out_0_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(regfile2out_0_for_loop_mux_sel_out),
  .mux_sel_msb_init(regfile2out_0_addr_gen_mux_sel_msb_init),
  .restart(regfile2out_0_for_loop_restart),
  .rst_n(rst_n),
  .starting_addr(regfile2out_0_addr_gen_starting_addr),
  .starting_addr2(regfile2out_0_addr_gen_starting_addr2),
  .step(read),
  .strides(regfile2out_0_addr_gen_strides),
  .strides2(regfile2out_0_addr_gen_strides2),
  .addr_out(read_addr)
);

sched_gen_4_16_dual_config_2 regfile2out_0_sched_gen (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(regfile2out_0_sched_gen_enable),
  .enable2(regfile2out_0_sched_gen_enable2),
  .finished(regfile2out_0_for_loop_restart),
  .flush(flush),
  .mux_sel(regfile2out_0_for_loop_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_starting_addr(regfile2out_0_sched_gen_sched_addr_gen_starting_addr),
  .sched_addr_gen_starting_addr2(regfile2out_0_sched_gen_sched_addr_gen_starting_addr2),
  .sched_addr_gen_strides2_0(regfile2out_0_sched_gen_sched_addr_gen_strides2_0),
  .sched_addr_gen_strides2_1(regfile2out_0_sched_gen_sched_addr_gen_strides2_1),
  .sched_addr_gen_strides_0(regfile2out_0_sched_gen_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(regfile2out_0_sched_gen_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(regfile2out_0_sched_gen_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(regfile2out_0_sched_gen_sched_addr_gen_strides_3),
  .mux_sel_msb_init(regfile2out_0_sched_gen_mux_sel_msb_init),
  .valid_output(regfile2out_0_sched_gen_valid_output)
);

endmodule   // strg_ub_thin_PondTop

module strg_ub_thin_PondTop_flat (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] data_in_f_0,
  input logic [0:0] [16:0] data_in_f_1,
  input logic flush,
  input logic rst_n,
  input logic [15:0] strg_ub_thin_PondTop_inst_data_from_strg_lifted,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr2,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_0,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_1,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_0,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_1,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_2,
  input logic [4:0] strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_3,
  input logic [2:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality,
  input logic [1:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality2,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_2,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_3,
  input logic strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable,
  input logic strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable2,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr2,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_2,
  input logic [15:0] strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_3,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr2,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_0,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_1,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_0,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_1,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_2,
  input logic [4:0] strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_3,
  input logic [2:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality,
  input logic [1:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality2,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_2,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_3,
  input logic strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable,
  input logic strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable2,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr2,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_2,
  input logic [15:0] strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_3,
  output logic [0:0] [16:0] data_out_f_0,
  output logic [0:0] [16:0] data_out_f_1,
  output logic [15:0] strg_ub_thin_PondTop_inst_data_to_strg_lifted,
  output logic [4:0] strg_ub_thin_PondTop_inst_rd_addr_out_lifted,
  output logic strg_ub_thin_PondTop_inst_ren_to_strg_lifted,
  output logic [4:0] strg_ub_thin_PondTop_inst_tmp0_rdaddr_lifted,
  output logic strg_ub_thin_PondTop_inst_tmp0_rden_lifted,
  output logic strg_ub_thin_PondTop_inst_wen_to_strg_lifted,
  output logic [4:0] strg_ub_thin_PondTop_inst_wr_addr_out_lifted,
  output logic valid_out_f_b_0,
  output logic valid_out_f_b_1
);

logic [1:0][16:0] strg_ub_thin_PondTop_inst_data_in;
logic [1:0][16:0] strg_ub_thin_PondTop_inst_data_out;
logic [1:0] strg_ub_thin_PondTop_inst_valid_out;
assign strg_ub_thin_PondTop_inst_data_in[0] = data_in_f_0;
assign strg_ub_thin_PondTop_inst_data_in[1] = data_in_f_1;
assign valid_out_f_b_0 = strg_ub_thin_PondTop_inst_valid_out[0];
assign valid_out_f_b_1 = strg_ub_thin_PondTop_inst_valid_out[1];
assign data_out_f_0 = strg_ub_thin_PondTop_inst_data_out[0];
assign data_out_f_1 = strg_ub_thin_PondTop_inst_data_out[1];
strg_ub_thin_PondTop strg_ub_thin_PondTop_inst (
  .clk(clk),
  .clk_en(clk_en),
  .data_from_strg(strg_ub_thin_PondTop_inst_data_from_strg_lifted),
  .data_in(strg_ub_thin_PondTop_inst_data_in),
  .flush(flush),
  .in2regfile_0_addr_gen_starting_addr(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr),
  .in2regfile_0_addr_gen_starting_addr2(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_starting_addr2),
  .in2regfile_0_addr_gen_strides2_0(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_0),
  .in2regfile_0_addr_gen_strides2_1(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides2_1),
  .in2regfile_0_addr_gen_strides_0(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_0),
  .in2regfile_0_addr_gen_strides_1(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_1),
  .in2regfile_0_addr_gen_strides_2(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_2),
  .in2regfile_0_addr_gen_strides_3(strg_ub_thin_PondTop_inst_in2regfile_0_addr_gen_strides_3),
  .in2regfile_0_for_loop_dimensionality(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality),
  .in2regfile_0_for_loop_dimensionality2(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_dimensionality2),
  .in2regfile_0_for_loop_ranges2_0(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_0),
  .in2regfile_0_for_loop_ranges2_1(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges2_1),
  .in2regfile_0_for_loop_ranges_0(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_0),
  .in2regfile_0_for_loop_ranges_1(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_1),
  .in2regfile_0_for_loop_ranges_2(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_2),
  .in2regfile_0_for_loop_ranges_3(strg_ub_thin_PondTop_inst_in2regfile_0_for_loop_ranges_3),
  .in2regfile_0_sched_gen_enable(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable),
  .in2regfile_0_sched_gen_enable2(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_enable2),
  .in2regfile_0_sched_gen_sched_addr_gen_starting_addr(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr),
  .in2regfile_0_sched_gen_sched_addr_gen_starting_addr2(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_starting_addr2),
  .in2regfile_0_sched_gen_sched_addr_gen_strides2_0(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_0),
  .in2regfile_0_sched_gen_sched_addr_gen_strides2_1(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides2_1),
  .in2regfile_0_sched_gen_sched_addr_gen_strides_0(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_0),
  .in2regfile_0_sched_gen_sched_addr_gen_strides_1(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_1),
  .in2regfile_0_sched_gen_sched_addr_gen_strides_2(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_2),
  .in2regfile_0_sched_gen_sched_addr_gen_strides_3(strg_ub_thin_PondTop_inst_in2regfile_0_sched_gen_sched_addr_gen_strides_3),
  .regfile2out_0_addr_gen_starting_addr(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr),
  .regfile2out_0_addr_gen_starting_addr2(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_starting_addr2),
  .regfile2out_0_addr_gen_strides2_0(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_0),
  .regfile2out_0_addr_gen_strides2_1(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides2_1),
  .regfile2out_0_addr_gen_strides_0(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_0),
  .regfile2out_0_addr_gen_strides_1(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_1),
  .regfile2out_0_addr_gen_strides_2(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_2),
  .regfile2out_0_addr_gen_strides_3(strg_ub_thin_PondTop_inst_regfile2out_0_addr_gen_strides_3),
  .regfile2out_0_for_loop_dimensionality(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality),
  .regfile2out_0_for_loop_dimensionality2(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_dimensionality2),
  .regfile2out_0_for_loop_ranges2_0(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_0),
  .regfile2out_0_for_loop_ranges2_1(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges2_1),
  .regfile2out_0_for_loop_ranges_0(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_0),
  .regfile2out_0_for_loop_ranges_1(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_1),
  .regfile2out_0_for_loop_ranges_2(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_2),
  .regfile2out_0_for_loop_ranges_3(strg_ub_thin_PondTop_inst_regfile2out_0_for_loop_ranges_3),
  .regfile2out_0_sched_gen_enable(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable),
  .regfile2out_0_sched_gen_enable2(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_enable2),
  .regfile2out_0_sched_gen_sched_addr_gen_starting_addr(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr),
  .regfile2out_0_sched_gen_sched_addr_gen_starting_addr2(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_starting_addr2),
  .regfile2out_0_sched_gen_sched_addr_gen_strides2_0(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_0),
  .regfile2out_0_sched_gen_sched_addr_gen_strides2_1(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides2_1),
  .regfile2out_0_sched_gen_sched_addr_gen_strides_0(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_0),
  .regfile2out_0_sched_gen_sched_addr_gen_strides_1(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_1),
  .regfile2out_0_sched_gen_sched_addr_gen_strides_2(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_2),
  .regfile2out_0_sched_gen_sched_addr_gen_strides_3(strg_ub_thin_PondTop_inst_regfile2out_0_sched_gen_sched_addr_gen_strides_3),
  .rst_n(rst_n),
  .data_out(strg_ub_thin_PondTop_inst_data_out),
  .data_to_strg(strg_ub_thin_PondTop_inst_data_to_strg_lifted),
  .rd_addr_out(strg_ub_thin_PondTop_inst_rd_addr_out_lifted),
  .ren_to_strg(strg_ub_thin_PondTop_inst_ren_to_strg_lifted),
  .tmp0_rdaddr(strg_ub_thin_PondTop_inst_tmp0_rdaddr_lifted),
  .tmp0_rden(strg_ub_thin_PondTop_inst_tmp0_rden_lifted),
  .valid_out(strg_ub_thin_PondTop_inst_valid_out),
  .wen_to_strg(strg_ub_thin_PondTop_inst_wen_to_strg_lifted),
  .wr_addr_out(strg_ub_thin_PondTop_inst_wr_addr_out_lifted)
);

endmodule   // strg_ub_thin_PondTop_flat

