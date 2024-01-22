module Chain_2_16 (
  input logic [1:0] accessor_output,
  input logic [1:0] [15:0] chain_data_in,
  input logic chain_en,
  input logic clk_en,
  input logic [1:0] [15:0] curr_tile_data_out,
  input logic flush,
  output logic [1:0] [15:0] data_out_tile
);

always_comb begin
  if (accessor_output[0]) begin
    data_out_tile[0] = curr_tile_data_out[0];
  end
  else if (chain_en) begin
    data_out_tile[0] = chain_data_in[0];
  end
  else data_out_tile[0] = 16'h0;
  if (accessor_output[1]) begin
    data_out_tile[1] = curr_tile_data_out[1];
  end
  else if (chain_en) begin
    data_out_tile[1] = chain_data_in[1];
  end
  else data_out_tile[1] = 16'h0;
end
endmodule   // Chain_2_16

module MemCore_inner (
  input logic [31:0] CONFIG_SPACE_0,
  input logic [31:0] CONFIG_SPACE_1,
  input logic [31:0] CONFIG_SPACE_10,
  input logic [31:0] CONFIG_SPACE_11,
  input logic [31:0] CONFIG_SPACE_12,
  input logic [31:0] CONFIG_SPACE_13,
  input logic [31:0] CONFIG_SPACE_14,
  input logic [31:0] CONFIG_SPACE_15,
  input logic [31:0] CONFIG_SPACE_16,
  input logic [31:0] CONFIG_SPACE_17,
  input logic [31:0] CONFIG_SPACE_18,
  input logic [31:0] CONFIG_SPACE_19,
  input logic [31:0] CONFIG_SPACE_2,
  input logic [31:0] CONFIG_SPACE_20,
  input logic [31:0] CONFIG_SPACE_21,
  input logic [31:0] CONFIG_SPACE_22,
  input logic [31:0] CONFIG_SPACE_23,
  input logic [31:0] CONFIG_SPACE_24,
  input logic [31:0] CONFIG_SPACE_25,
  input logic [31:0] CONFIG_SPACE_26,
  input logic [31:0] CONFIG_SPACE_27,
  input logic [31:0] CONFIG_SPACE_28,
  input logic [31:0] CONFIG_SPACE_29,
  input logic [31:0] CONFIG_SPACE_3,
  input logic [31:0] CONFIG_SPACE_30,
  input logic [31:0] CONFIG_SPACE_31,
  input logic [31:0] CONFIG_SPACE_32,
  input logic [31:0] CONFIG_SPACE_33,
  input logic [31:0] CONFIG_SPACE_34,
  input logic [31:0] CONFIG_SPACE_35,
  input logic [31:0] CONFIG_SPACE_36,
  input logic [31:0] CONFIG_SPACE_37,
  input logic [31:0] CONFIG_SPACE_38,
  input logic [31:0] CONFIG_SPACE_39,
  input logic [31:0] CONFIG_SPACE_4,
  input logic [31:0] CONFIG_SPACE_40,
  input logic [31:0] CONFIG_SPACE_41,
  input logic [31:0] CONFIG_SPACE_42,
  input logic [31:0] CONFIG_SPACE_43,
  input logic [31:0] CONFIG_SPACE_44,
  input logic [18:0] CONFIG_SPACE_45,
  input logic [31:0] CONFIG_SPACE_5,
  input logic [31:0] CONFIG_SPACE_6,
  input logic [31:0] CONFIG_SPACE_7,
  input logic [31:0] CONFIG_SPACE_8,
  input logic [31:0] CONFIG_SPACE_9,
  input logic [0:0] [16:0] MEM_input_width_17_num_0,
  input logic MEM_input_width_17_num_0_valid,
  input logic [0:0] [16:0] MEM_input_width_17_num_1,
  input logic MEM_input_width_17_num_1_valid,
  input logic [0:0] [16:0] MEM_input_width_17_num_2,
  input logic MEM_input_width_17_num_2_valid,
  input logic [0:0] [16:0] MEM_input_width_17_num_3,
  input logic MEM_input_width_17_num_3_valid,
  input logic MEM_input_width_1_num_0,
  input logic MEM_input_width_1_num_1,
  input logic MEM_output_width_17_num_0_ready,
  input logic MEM_output_width_17_num_1_ready,
  input logic MEM_output_width_17_num_2_ready,
  input logic clk,
  input logic clk_en,
  input logic [7:0] config_addr_in,
  input logic [31:0] config_data_in,
  input logic [1:0] config_en,
  input logic config_read,
  input logic config_write,
  input logic flush,
  input logic [1:0] mode,
  input logic mode_excl,
  input logic rst_n,
  input logic tile_en,
  output logic MEM_input_width_17_num_0_ready,
  output logic MEM_input_width_17_num_1_ready,
  output logic MEM_input_width_17_num_2_ready,
  output logic MEM_input_width_17_num_3_ready,
  output logic [0:0] [16:0] MEM_output_width_17_num_0,
  output logic MEM_output_width_17_num_0_valid,
  output logic [0:0] [16:0] MEM_output_width_17_num_1,
  output logic MEM_output_width_17_num_1_valid,
  output logic [0:0] [16:0] MEM_output_width_17_num_2,
  output logic MEM_output_width_17_num_2_valid,
  output logic MEM_output_width_1_num_0,
  output logic MEM_output_width_1_num_1,
  output logic MEM_output_width_1_num_2,
  output logic [1:0] [31:0] config_data_out
);

logic [1458:0] CONFIG_SPACE;
logic [15:0] config_data_in_shrt;
logic [1:0][15:0] config_data_out_shrt;
logic [8:0] config_seq_addr_out;
logic config_seq_clk_en;
logic [0:0][3:0][15:0] config_seq_rd_data_stg;
logic config_seq_ren_out;
logic config_seq_wen_out;
logic [3:0][15:0] config_seq_wr_data;
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
logic mem_ctrl_fiber_access_16_flat_clk;
logic [8:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted;
logic [3:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_buffet_capacity_log_0;
logic [3:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_buffet_capacity_log_1;
logic [3:0][15:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_data_from_mem_lifted_lifted;
logic [3:0][15:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_data_to_mem_lifted_lifted;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_ren_to_mem_lifted_lifted;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_tile_en;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_wen_to_mem_lifted_lifted;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_block_mode;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_dense;
logic [15:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_dim_size;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_do_repeat;
logic [15:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_inner_dim_offset;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_lookup;
logic [15:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_repeat_factor;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_repeat_outer_inner_n;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_root;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_tile_en;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_tile_en;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_vector_reduce_mode;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_block_mode;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_compressed;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_init_blank;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_lowest_level;
logic [15:0] mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_stop_lvl;
logic mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_tile_en;
logic [0:0][16:0] mem_ctrl_fiber_access_16_flat_read_scanner_block_rd_out_f_;
logic mem_ctrl_fiber_access_16_flat_read_scanner_block_rd_out_valid_f_;
logic [0:0][16:0] mem_ctrl_fiber_access_16_flat_read_scanner_coord_out_f_;
logic mem_ctrl_fiber_access_16_flat_read_scanner_coord_out_valid_f_;
logic [0:0][16:0] mem_ctrl_fiber_access_16_flat_read_scanner_pos_out_f_;
logic mem_ctrl_fiber_access_16_flat_read_scanner_pos_out_valid_f_;
logic mem_ctrl_fiber_access_16_flat_read_scanner_us_pos_in_ready_f_;
logic mem_ctrl_fiber_access_16_flat_write_scanner_addr_in_ready_f_;
logic mem_ctrl_fiber_access_16_flat_write_scanner_block_wr_in_ready_f_;
logic mem_ctrl_fiber_access_16_flat_write_scanner_data_in_ready_f_;
logic mem_ctrl_stencil_valid_flat_clk;
logic mem_ctrl_stencil_valid_flat_stencil_valid_f_;
logic [3:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_dimensionality;
logic [10:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_0;
logic [10:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_1;
logic [10:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_2;
logic [10:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_3;
logic [10:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_4;
logic [10:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_5;
logic mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_enable;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_3;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_4;
logic [15:0] mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_5;
logic mem_ctrl_strg_ram_64_512_delay1_flat_clk;
logic [0:0][16:0] mem_ctrl_strg_ram_64_512_delay1_flat_data_out_f_;
logic mem_ctrl_strg_ram_64_512_delay1_flat_ready_f_;
logic [0:0][8:0] mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_addr_out_lifted;
logic [0:0][3:0][15:0] mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_data_from_strg_lifted;
logic [0:0][3:0][15:0] mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_data_to_strg_lifted;
logic mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_ren_to_strg_lifted;
logic mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_wen_to_strg_lifted;
logic mem_ctrl_strg_ram_64_512_delay1_flat_valid_out_f_;
logic mem_ctrl_strg_ub_vec_flat_accessor_output_f_b_0;
logic mem_ctrl_strg_ub_vec_flat_accessor_output_f_b_1;
logic mem_ctrl_strg_ub_vec_flat_clk;
logic [0:0][16:0] mem_ctrl_strg_ub_vec_flat_data_out_f_0;
logic [0:0][16:0] mem_ctrl_strg_ub_vec_flat_data_out_f_1;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_addr_out_lifted;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_starting_addr;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_0;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_1;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_2;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_starting_addr;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_0;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_1;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_2;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_enable;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_enable;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_dimensionality;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_0;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_1;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_2;
logic [2:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_dimensionality;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_0;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_1;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_2;
logic [7:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_0_agg_read_padding;
logic [7:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_1_agg_read_padding;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr;
logic [1:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_mode_0;
logic [1:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_mode_1;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_chain_chain_en;
logic [3:0][15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_data_from_strg_lifted;
logic [3:0][15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_data_to_strg_lifted;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_ren_to_strg_lifted;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_starting_addr;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_0;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_1;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_2;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_3;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_4;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_5;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_starting_addr;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_0;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_1;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_2;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_3;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_4;
logic [8:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_enable;
logic [9:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_enable;
logic [9:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_dimensionality;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_0;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_1;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_2;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_3;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_4;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_dimensionality;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_0;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_1;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_2;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_3;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_4;
logic [10:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_5;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_shared_tb_0;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_starting_addr;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_0;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_1;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_2;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_3;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_4;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_starting_addr;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_0;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_1;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_2;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_3;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_4;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_5;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_enable;
logic [9:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_delay;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_enable;
logic [9:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_delay;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4;
logic [15:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_starting_addr;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_0;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_1;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_2;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_3;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_4;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_5;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_starting_addr;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_0;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_1;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_2;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_3;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_4;
logic [3:0] mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_5;
logic mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_wen_to_strg_lifted;
logic memory_0_clk_en;
logic [63:0] memory_0_data_in_p0;
logic [63:0] memory_0_data_out_p0;
logic [8:0] memory_0_read_addr_p0;
logic memory_0_read_enable_p0;
logic [8:0] memory_0_write_addr_p0;
logic memory_0_write_enable_p0;
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
assign mem_ctrl_fiber_access_16_flat_clk = gclk & (mode == 2'h0);
assign mem_ctrl_strg_ub_vec_flat_clk = gclk & (mode == 2'h1);
assign mem_ctrl_strg_ram_64_512_delay1_flat_clk = gclk & (mode == 2'h2);
assign mem_ctrl_stencil_valid_flat_clk = gclk;
assign input_width_17_num_0_fifo_out_valid = ~input_width_17_num_0_input_fifo_empty;
always_comb begin
  input_width_17_num_0_fifo_out_ready = 1'h1;
  if (mode == 2'h0) begin
    input_width_17_num_0_fifo_out_ready = mem_ctrl_fiber_access_16_flat_read_scanner_us_pos_in_ready_f_;
  end
  else input_width_17_num_0_fifo_out_ready = 1'h1;
end
always_comb begin
  MEM_input_width_17_num_0_ready = 1'h1;
  if (mode == 2'h0) begin
    MEM_input_width_17_num_0_ready = ~input_width_17_num_0_input_fifo_full;
  end
  else if (mode == 2'h1) begin
    MEM_input_width_17_num_0_ready = 1'h1;
  end
  else if (mode == 2'h2) begin
    MEM_input_width_17_num_0_ready = 1'h1;
  end
end
assign input_width_17_num_1_fifo_out_valid = ~input_width_17_num_1_input_fifo_empty;
always_comb begin
  input_width_17_num_1_fifo_out_ready = 1'h1;
  if (mode == 2'h0) begin
    input_width_17_num_1_fifo_out_ready = mem_ctrl_fiber_access_16_flat_write_scanner_addr_in_ready_f_;
  end
  else input_width_17_num_1_fifo_out_ready = 1'h1;
end
always_comb begin
  MEM_input_width_17_num_1_ready = 1'h1;
  if (mode == 2'h0) begin
    MEM_input_width_17_num_1_ready = ~input_width_17_num_1_input_fifo_full;
  end
  else if (mode == 2'h1) begin
    MEM_input_width_17_num_1_ready = 1'h1;
  end
  else if (mode == 2'h2) begin
    MEM_input_width_17_num_1_ready = 1'h1;
  end
end
assign input_width_17_num_2_fifo_out_valid = ~input_width_17_num_2_input_fifo_empty;
always_comb begin
  input_width_17_num_2_fifo_out_ready = 1'h1;
  if (mode == 2'h0) begin
    input_width_17_num_2_fifo_out_ready = mem_ctrl_fiber_access_16_flat_write_scanner_block_wr_in_ready_f_;
  end
  else input_width_17_num_2_fifo_out_ready = 1'h1;
end
always_comb begin
  MEM_input_width_17_num_2_ready = 1'h1;
  if (mode == 2'h0) begin
    MEM_input_width_17_num_2_ready = ~input_width_17_num_2_input_fifo_full;
  end
  else if (mode == 2'h1) begin
    MEM_input_width_17_num_2_ready = 1'h1;
  end
  else if (mode == 2'h2) begin
    MEM_input_width_17_num_2_ready = 1'h1;
  end
end
assign input_width_17_num_3_fifo_out_valid = ~input_width_17_num_3_input_fifo_empty;
always_comb begin
  input_width_17_num_3_fifo_out_ready = 1'h1;
  if (mode == 2'h0) begin
    input_width_17_num_3_fifo_out_ready = mem_ctrl_fiber_access_16_flat_write_scanner_data_in_ready_f_;
  end
  else input_width_17_num_3_fifo_out_ready = 1'h1;
end
always_comb begin
  MEM_input_width_17_num_3_ready = 1'h1;
  if (mode == 2'h0) begin
    MEM_input_width_17_num_3_ready = ~input_width_17_num_3_input_fifo_full;
  end
  else if (mode == 2'h1) begin
    MEM_input_width_17_num_3_ready = 1'h1;
  end
end
assign output_width_17_num_0_fifo_in_ready = ~output_width_17_num_0_output_fifo_full;
always_comb begin
  output_width_17_num_0_fifo_in = 17'h0;
  output_width_17_num_0_fifo_in_valid = 1'h0;
  output_width_17_num_0_fifo_in = mem_ctrl_fiber_access_16_flat_read_scanner_block_rd_out_f_;
  output_width_17_num_0_fifo_in_valid = mem_ctrl_fiber_access_16_flat_read_scanner_block_rd_out_valid_f_;
end
always_comb begin
  MEM_output_width_17_num_0 = 17'h0;
  if (mode == 2'h0) begin
    MEM_output_width_17_num_0 = output_width_17_num_0_output_fifo_data_out;
  end
  else if (mode == 2'h1) begin
    MEM_output_width_17_num_0 = mem_ctrl_strg_ub_vec_flat_data_out_f_0;
  end
  else if (mode == 2'h2) begin
    MEM_output_width_17_num_0 = mem_ctrl_strg_ram_64_512_delay1_flat_data_out_f_;
  end
end
always_comb begin
  MEM_output_width_17_num_0_valid = 1'h0;
  if (mode == 2'h0) begin
    MEM_output_width_17_num_0_valid = ~output_width_17_num_0_output_fifo_empty;
  end
  else if (mode == 2'h1) begin
    MEM_output_width_17_num_0_valid = 1'h1;
  end
  else if (mode == 2'h2) begin
    MEM_output_width_17_num_0_valid = 1'h1;
  end
end
assign output_width_17_num_1_fifo_in_ready = ~output_width_17_num_1_output_fifo_full;
always_comb begin
  output_width_17_num_1_fifo_in = 17'h0;
  output_width_17_num_1_fifo_in_valid = 1'h0;
  output_width_17_num_1_fifo_in = mem_ctrl_fiber_access_16_flat_read_scanner_coord_out_f_;
  output_width_17_num_1_fifo_in_valid = mem_ctrl_fiber_access_16_flat_read_scanner_coord_out_valid_f_;
end
always_comb begin
  MEM_output_width_17_num_1 = 17'h0;
  if (mode == 2'h0) begin
    MEM_output_width_17_num_1 = output_width_17_num_1_output_fifo_data_out;
  end
  else if (mode == 2'h1) begin
    MEM_output_width_17_num_1 = mem_ctrl_strg_ub_vec_flat_data_out_f_1;
  end
end
always_comb begin
  MEM_output_width_17_num_1_valid = 1'h0;
  if (mode == 2'h0) begin
    MEM_output_width_17_num_1_valid = ~output_width_17_num_1_output_fifo_empty;
  end
  else if (mode == 2'h1) begin
    MEM_output_width_17_num_1_valid = 1'h1;
  end
end
assign output_width_17_num_2_fifo_in_ready = ~output_width_17_num_2_output_fifo_full;
always_comb begin
  output_width_17_num_2_fifo_in = 17'h0;
  output_width_17_num_2_fifo_in_valid = 1'h0;
  output_width_17_num_2_fifo_in = mem_ctrl_fiber_access_16_flat_read_scanner_pos_out_f_;
  output_width_17_num_2_fifo_in_valid = mem_ctrl_fiber_access_16_flat_read_scanner_pos_out_valid_f_;
end
always_comb begin
  MEM_output_width_17_num_2 = 17'h0;
  if (mode == 2'h0) begin
    MEM_output_width_17_num_2 = output_width_17_num_2_output_fifo_data_out;
  end
  else MEM_output_width_17_num_2 = 17'h0;
end
always_comb begin
  MEM_output_width_17_num_2_valid = 1'h0;
  if (mode == 2'h0) begin
    MEM_output_width_17_num_2_valid = ~output_width_17_num_2_output_fifo_empty;
  end
  else MEM_output_width_17_num_2_valid = 1'h0;
end
always_comb begin
  MEM_output_width_1_num_0 = 1'h0;
  if (mode == 2'h1) begin
    MEM_output_width_1_num_0 = mem_ctrl_strg_ub_vec_flat_accessor_output_f_b_0;
  end
  else if (mode == 2'h2) begin
    MEM_output_width_1_num_0 = mem_ctrl_strg_ram_64_512_delay1_flat_ready_f_;
  end
end
always_comb begin
  MEM_output_width_1_num_1 = 1'h0;
  if (mode == 2'h1) begin
    MEM_output_width_1_num_1 = mem_ctrl_strg_ub_vec_flat_accessor_output_f_b_1;
  end
  else if (mode == 2'h2) begin
    MEM_output_width_1_num_1 = mem_ctrl_strg_ram_64_512_delay1_flat_valid_out_f_;
  end
end
always_comb begin
  MEM_output_width_1_num_2 = 1'h0;
  if (mode_excl == 1'h1) begin
    MEM_output_width_1_num_2 = mem_ctrl_stencil_valid_flat_stencil_valid_f_;
  end
  else MEM_output_width_1_num_2 = 1'h0;
end
always_comb begin
  memory_0_data_in_p0 = 64'h0;
  memory_0_write_addr_p0 = 9'h0;
  memory_0_write_enable_p0 = 1'h0;
  memory_0_read_addr_p0 = 9'h0;
  memory_0_read_enable_p0 = 1'h0;
  if (|config_en) begin
    memory_0_data_in_p0 = config_seq_wr_data;
    memory_0_write_addr_p0 = config_seq_addr_out;
    memory_0_write_enable_p0 = config_seq_wen_out;
    memory_0_read_addr_p0 = config_seq_addr_out;
    memory_0_read_enable_p0 = config_seq_ren_out;
  end
  else if (mode == 2'h0) begin
    memory_0_data_in_p0 = mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_data_to_mem_lifted_lifted;
    memory_0_write_addr_p0 = mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted;
    memory_0_write_enable_p0 = mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_wen_to_mem_lifted_lifted;
    memory_0_read_addr_p0 = mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted;
    memory_0_read_enable_p0 = mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_ren_to_mem_lifted_lifted;
  end
  else if (mode == 2'h1) begin
    memory_0_data_in_p0 = mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_data_to_strg_lifted;
    memory_0_write_addr_p0 = mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_addr_out_lifted;
    memory_0_write_enable_p0 = mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_wen_to_strg_lifted;
    memory_0_read_addr_p0 = mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_addr_out_lifted;
    memory_0_read_enable_p0 = mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_ren_to_strg_lifted;
  end
  else if (mode == 2'h2) begin
    memory_0_data_in_p0 = mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_data_to_strg_lifted;
    memory_0_write_addr_p0 = mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_addr_out_lifted;
    memory_0_write_enable_p0 = mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_wen_to_strg_lifted;
    memory_0_read_addr_p0 = mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_addr_out_lifted;
    memory_0_read_enable_p0 = mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_ren_to_strg_lifted;
  end
end
always_comb begin
  mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_data_from_mem_lifted_lifted = memory_0_data_out_p0;
  mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_data_from_strg_lifted = memory_0_data_out_p0;
  mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_data_from_strg_lifted = memory_0_data_out_p0;
  config_seq_rd_data_stg = memory_0_data_out_p0;
end
assign config_data_in_shrt = config_data_in[15:0];
assign config_data_out[0] = 32'(config_data_out_shrt[0]);
assign config_data_out[1] = 32'(config_data_out_shrt[1]);
assign config_seq_clk_en = clk_en | (|config_en);
assign memory_0_clk_en = clk_en | (|config_en);
assign {mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_buffet_capacity_log_0, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_buffet_capacity_log_1, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_tile_en, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_block_mode, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_dense, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_dim_size, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_do_repeat, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_inner_dim_offset, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_lookup, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_repeat_factor, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_repeat_outer_inner_n, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_root, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_tile_en, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_tile_en, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_vector_reduce_mode, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_block_mode, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_compressed, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_init_blank, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_lowest_level, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_stop_lvl, mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_tile_en} = CONFIG_SPACE[86:0];
assign {mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_enable, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_enable, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_dimensionality, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_dimensionality, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_0_agg_read_padding, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_1_agg_read_padding, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_mode_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_mode_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_chain_chain_en, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_enable, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_enable, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_dimensionality, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_dimensionality, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_shared_tb_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_enable, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_delay, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_enable, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_delay, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_5, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_starting_addr, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_0, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_1, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_2, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_3, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_4, mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_5} = CONFIG_SPACE[1275:0];
assign {mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_dimensionality, mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_0, mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_1, mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_2, mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_3, mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_4, mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_5, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_enable, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_starting_addr, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_0, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_1, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_2, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_3, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_4, mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_5} = CONFIG_SPACE[1458:1276];
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
assign CONFIG_SPACE[543:512] = CONFIG_SPACE_16;
assign CONFIG_SPACE[575:544] = CONFIG_SPACE_17;
assign CONFIG_SPACE[607:576] = CONFIG_SPACE_18;
assign CONFIG_SPACE[639:608] = CONFIG_SPACE_19;
assign CONFIG_SPACE[671:640] = CONFIG_SPACE_20;
assign CONFIG_SPACE[703:672] = CONFIG_SPACE_21;
assign CONFIG_SPACE[735:704] = CONFIG_SPACE_22;
assign CONFIG_SPACE[767:736] = CONFIG_SPACE_23;
assign CONFIG_SPACE[799:768] = CONFIG_SPACE_24;
assign CONFIG_SPACE[831:800] = CONFIG_SPACE_25;
assign CONFIG_SPACE[863:832] = CONFIG_SPACE_26;
assign CONFIG_SPACE[895:864] = CONFIG_SPACE_27;
assign CONFIG_SPACE[927:896] = CONFIG_SPACE_28;
assign CONFIG_SPACE[959:928] = CONFIG_SPACE_29;
assign CONFIG_SPACE[991:960] = CONFIG_SPACE_30;
assign CONFIG_SPACE[1023:992] = CONFIG_SPACE_31;
assign CONFIG_SPACE[1055:1024] = CONFIG_SPACE_32;
assign CONFIG_SPACE[1087:1056] = CONFIG_SPACE_33;
assign CONFIG_SPACE[1119:1088] = CONFIG_SPACE_34;
assign CONFIG_SPACE[1151:1120] = CONFIG_SPACE_35;
assign CONFIG_SPACE[1183:1152] = CONFIG_SPACE_36;
assign CONFIG_SPACE[1215:1184] = CONFIG_SPACE_37;
assign CONFIG_SPACE[1247:1216] = CONFIG_SPACE_38;
assign CONFIG_SPACE[1279:1248] = CONFIG_SPACE_39;
assign CONFIG_SPACE[1311:1280] = CONFIG_SPACE_40;
assign CONFIG_SPACE[1343:1312] = CONFIG_SPACE_41;
assign CONFIG_SPACE[1375:1344] = CONFIG_SPACE_42;
assign CONFIG_SPACE[1407:1376] = CONFIG_SPACE_43;
assign CONFIG_SPACE[1439:1408] = CONFIG_SPACE_44;
assign CONFIG_SPACE[1458:1440] = CONFIG_SPACE_45;
fiber_access_16_flat mem_ctrl_fiber_access_16_flat (
  .clk(mem_ctrl_fiber_access_16_flat_clk),
  .clk_en(clk_en),
  .fiber_access_16_inst_buffet_buffet_capacity_log_0(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_buffet_capacity_log_0),
  .fiber_access_16_inst_buffet_buffet_capacity_log_1(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_buffet_capacity_log_1),
  .fiber_access_16_inst_buffet_data_from_mem_lifted_lifted(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_data_from_mem_lifted_lifted),
  .fiber_access_16_inst_buffet_tile_en(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_tile_en),
  .fiber_access_16_inst_read_scanner_block_mode(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_block_mode),
  .fiber_access_16_inst_read_scanner_dense(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_dense),
  .fiber_access_16_inst_read_scanner_dim_size(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_dim_size),
  .fiber_access_16_inst_read_scanner_do_repeat(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_do_repeat),
  .fiber_access_16_inst_read_scanner_inner_dim_offset(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_inner_dim_offset),
  .fiber_access_16_inst_read_scanner_lookup(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_lookup),
  .fiber_access_16_inst_read_scanner_repeat_factor(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_repeat_factor),
  .fiber_access_16_inst_read_scanner_repeat_outer_inner_n(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_repeat_outer_inner_n),
  .fiber_access_16_inst_read_scanner_root(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_root),
  .fiber_access_16_inst_read_scanner_tile_en(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_read_scanner_tile_en),
  .fiber_access_16_inst_tile_en(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_tile_en),
  .fiber_access_16_inst_vector_reduce_mode(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_vector_reduce_mode),
  .fiber_access_16_inst_write_scanner_block_mode(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_block_mode),
  .fiber_access_16_inst_write_scanner_compressed(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_compressed),
  .fiber_access_16_inst_write_scanner_init_blank(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_init_blank),
  .fiber_access_16_inst_write_scanner_lowest_level(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_lowest_level),
  .fiber_access_16_inst_write_scanner_stop_lvl(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_stop_lvl),
  .fiber_access_16_inst_write_scanner_tile_en(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_write_scanner_tile_en),
  .flush(flush),
  .read_scanner_block_rd_out_ready_f_(output_width_17_num_0_fifo_in_ready),
  .read_scanner_coord_out_ready_f_(output_width_17_num_1_fifo_in_ready),
  .read_scanner_pos_out_ready_f_(output_width_17_num_2_fifo_in_ready),
  .read_scanner_us_pos_in_f_(input_width_17_num_0_fifo_out),
  .read_scanner_us_pos_in_valid_f_(input_width_17_num_0_fifo_out_valid),
  .rst_n(rst_n),
  .write_scanner_addr_in_f_(input_width_17_num_1_fifo_out),
  .write_scanner_addr_in_valid_f_(input_width_17_num_1_fifo_out_valid),
  .write_scanner_block_wr_in_f_(input_width_17_num_2_fifo_out),
  .write_scanner_block_wr_in_valid_f_(input_width_17_num_2_fifo_out_valid),
  .write_scanner_data_in_f_(input_width_17_num_3_fifo_out),
  .write_scanner_data_in_valid_f_(input_width_17_num_3_fifo_out_valid),
  .fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted),
  .fiber_access_16_inst_buffet_data_to_mem_lifted_lifted(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_data_to_mem_lifted_lifted),
  .fiber_access_16_inst_buffet_ren_to_mem_lifted_lifted(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_ren_to_mem_lifted_lifted),
  .fiber_access_16_inst_buffet_wen_to_mem_lifted_lifted(mem_ctrl_fiber_access_16_flat_fiber_access_16_inst_buffet_wen_to_mem_lifted_lifted),
  .read_scanner_block_rd_out_f_(mem_ctrl_fiber_access_16_flat_read_scanner_block_rd_out_f_),
  .read_scanner_block_rd_out_valid_f_(mem_ctrl_fiber_access_16_flat_read_scanner_block_rd_out_valid_f_),
  .read_scanner_coord_out_f_(mem_ctrl_fiber_access_16_flat_read_scanner_coord_out_f_),
  .read_scanner_coord_out_valid_f_(mem_ctrl_fiber_access_16_flat_read_scanner_coord_out_valid_f_),
  .read_scanner_pos_out_f_(mem_ctrl_fiber_access_16_flat_read_scanner_pos_out_f_),
  .read_scanner_pos_out_valid_f_(mem_ctrl_fiber_access_16_flat_read_scanner_pos_out_valid_f_),
  .read_scanner_us_pos_in_ready_f_(mem_ctrl_fiber_access_16_flat_read_scanner_us_pos_in_ready_f_),
  .write_scanner_addr_in_ready_f_(mem_ctrl_fiber_access_16_flat_write_scanner_addr_in_ready_f_),
  .write_scanner_block_wr_in_ready_f_(mem_ctrl_fiber_access_16_flat_write_scanner_block_wr_in_ready_f_),
  .write_scanner_data_in_ready_f_(mem_ctrl_fiber_access_16_flat_write_scanner_data_in_ready_f_)
);

strg_ub_vec_flat mem_ctrl_strg_ub_vec_flat (
  .chain_data_in_f_0(MEM_input_width_17_num_0),
  .chain_data_in_f_1(MEM_input_width_17_num_1),
  .clk(mem_ctrl_strg_ub_vec_flat_clk),
  .clk_en(clk_en),
  .data_in_f_0(MEM_input_width_17_num_2),
  .data_in_f_1(MEM_input_width_17_num_3),
  .flush(flush),
  .rst_n(rst_n),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_starting_addr),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_0),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_1),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_2),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_starting_addr),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_0),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_1),
  .strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_2),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_enable(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_enable),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_enable(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_enable),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1),
  .strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2),
  .strg_ub_vec_inst_agg_only_loops_in2buf_0_dimensionality(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_dimensionality),
  .strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_0),
  .strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_1),
  .strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_2),
  .strg_ub_vec_inst_agg_only_loops_in2buf_1_dimensionality(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_dimensionality),
  .strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_0),
  .strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_1),
  .strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_2),
  .strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_0_agg_read_padding(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_0_agg_read_padding),
  .strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_1_agg_read_padding(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_1_agg_read_padding),
  .strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr),
  .strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr),
  .strg_ub_vec_inst_agg_sram_shared_mode_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_mode_0),
  .strg_ub_vec_inst_agg_sram_shared_mode_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_agg_sram_shared_mode_1),
  .strg_ub_vec_inst_chain_chain_en(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_chain_chain_en),
  .strg_ub_vec_inst_data_from_strg_lifted(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_data_from_strg_lifted),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_starting_addr),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_0),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_1),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_2),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_3),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_4),
  .strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_5),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_starting_addr),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_0),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_1),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_2),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_3),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_4),
  .strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_5),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4),
  .strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_enable(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_enable),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_enable(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_enable),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4),
  .strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_dimensionality(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_dimensionality),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_0),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_1),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_2),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_3),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_4),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_5),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_dimensionality(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_dimensionality),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_0),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_1),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_2),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_3),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_4),
  .strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_5),
  .strg_ub_vec_inst_tb_only_shared_tb_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_shared_tb_0),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_starting_addr),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_0),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_1),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_2),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_3),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_4),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_5),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_starting_addr),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_0),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_1),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_2),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_3),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_4),
  .strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_5),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_enable(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_enable),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_delay(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_delay),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_enable(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_enable),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_delay(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_delay),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4),
  .strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_starting_addr),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_0),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_1),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_2),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_3),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_4),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_5),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_starting_addr(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_starting_addr),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_0(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_0),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_1(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_1),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_2(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_2),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_3(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_3),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_4(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_4),
  .strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_5(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_5),
  .accessor_output_f_b_0(mem_ctrl_strg_ub_vec_flat_accessor_output_f_b_0),
  .accessor_output_f_b_1(mem_ctrl_strg_ub_vec_flat_accessor_output_f_b_1),
  .data_out_f_0(mem_ctrl_strg_ub_vec_flat_data_out_f_0),
  .data_out_f_1(mem_ctrl_strg_ub_vec_flat_data_out_f_1),
  .strg_ub_vec_inst_addr_out_lifted(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_addr_out_lifted),
  .strg_ub_vec_inst_data_to_strg_lifted(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_data_to_strg_lifted),
  .strg_ub_vec_inst_ren_to_strg_lifted(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_ren_to_strg_lifted),
  .strg_ub_vec_inst_wen_to_strg_lifted(mem_ctrl_strg_ub_vec_flat_strg_ub_vec_inst_wen_to_strg_lifted)
);

strg_ram_64_512_delay1_flat mem_ctrl_strg_ram_64_512_delay1_flat (
  .clk(mem_ctrl_strg_ram_64_512_delay1_flat_clk),
  .clk_en(clk_en),
  .data_in_f_(MEM_input_width_17_num_0),
  .flush(flush),
  .rd_addr_in_f_(MEM_input_width_17_num_1),
  .ren_f_(MEM_input_width_1_num_0),
  .rst_n(rst_n),
  .strg_ram_64_512_delay1_inst_data_from_strg_lifted(mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_data_from_strg_lifted),
  .wen_f_(MEM_input_width_1_num_1),
  .wr_addr_in_f_(MEM_input_width_17_num_2),
  .data_out_f_(mem_ctrl_strg_ram_64_512_delay1_flat_data_out_f_),
  .ready_f_(mem_ctrl_strg_ram_64_512_delay1_flat_ready_f_),
  .strg_ram_64_512_delay1_inst_addr_out_lifted(mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_addr_out_lifted),
  .strg_ram_64_512_delay1_inst_data_to_strg_lifted(mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_data_to_strg_lifted),
  .strg_ram_64_512_delay1_inst_ren_to_strg_lifted(mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_ren_to_strg_lifted),
  .strg_ram_64_512_delay1_inst_wen_to_strg_lifted(mem_ctrl_strg_ram_64_512_delay1_flat_strg_ram_64_512_delay1_inst_wen_to_strg_lifted),
  .valid_out_f_(mem_ctrl_strg_ram_64_512_delay1_flat_valid_out_f_)
);

stencil_valid_flat mem_ctrl_stencil_valid_flat (
  .clk(mem_ctrl_stencil_valid_flat_clk),
  .clk_en(clk_en),
  .flush(flush),
  .rst_n(rst_n),
  .stencil_valid_inst_loops_stencil_valid_dimensionality(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_dimensionality),
  .stencil_valid_inst_loops_stencil_valid_ranges_0(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_0),
  .stencil_valid_inst_loops_stencil_valid_ranges_1(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_1),
  .stencil_valid_inst_loops_stencil_valid_ranges_2(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_2),
  .stencil_valid_inst_loops_stencil_valid_ranges_3(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_3),
  .stencil_valid_inst_loops_stencil_valid_ranges_4(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_4),
  .stencil_valid_inst_loops_stencil_valid_ranges_5(mem_ctrl_stencil_valid_flat_stencil_valid_inst_loops_stencil_valid_ranges_5),
  .stencil_valid_inst_stencil_valid_sched_gen_enable(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_enable),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_starting_addr(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_starting_addr),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_0(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_0),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_1(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_1),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_2(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_2),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_3(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_3),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_4(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_4),
  .stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_5(mem_ctrl_stencil_valid_flat_stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_5),
  .stencil_valid_f_(mem_ctrl_stencil_valid_flat_stencil_valid_f_)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_0_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(MEM_input_width_17_num_0),
  .flush(flush),
  .pop(input_width_17_num_0_fifo_out_ready),
  .push(MEM_input_width_17_num_0_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_0_fifo_out),
  .empty(input_width_17_num_0_input_fifo_empty),
  .full(input_width_17_num_0_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_1_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(MEM_input_width_17_num_1),
  .flush(flush),
  .pop(input_width_17_num_1_fifo_out_ready),
  .push(MEM_input_width_17_num_1_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_1_fifo_out),
  .empty(input_width_17_num_1_input_fifo_empty),
  .full(input_width_17_num_1_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_2_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(MEM_input_width_17_num_2),
  .flush(flush),
  .pop(input_width_17_num_2_fifo_out_ready),
  .push(MEM_input_width_17_num_2_valid),
  .rst_n(rst_n),
  .data_out(input_width_17_num_2_fifo_out),
  .empty(input_width_17_num_2_input_fifo_empty),
  .full(input_width_17_num_2_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 input_width_17_num_3_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(MEM_input_width_17_num_3),
  .flush(flush),
  .pop(input_width_17_num_3_fifo_out_ready),
  .push(MEM_input_width_17_num_3_valid),
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
  .pop(MEM_output_width_17_num_0_ready),
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
  .pop(MEM_output_width_17_num_1_ready),
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
  .pop(MEM_output_width_17_num_2_ready),
  .push(output_width_17_num_2_fifo_in_valid),
  .rst_n(rst_n),
  .data_out(output_width_17_num_2_output_fifo_data_out),
  .empty(output_width_17_num_2_output_fifo_empty),
  .full(output_width_17_num_2_output_fifo_full)
);

sram_sp__0 memory_0 (
  .clk(gclk),
  .clk_en(memory_0_clk_en),
  .data_in_p0(memory_0_data_in_p0),
  .flush(flush),
  .read_addr_p0(memory_0_read_addr_p0),
  .read_enable_p0(memory_0_read_enable_p0),
  .write_addr_p0(memory_0_write_addr_p0),
  .write_enable_p0(memory_0_write_enable_p0),
  .data_out_p0(memory_0_data_out_p0)
);

storage_config_seq_2_64_16 config_seq (
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

endmodule   // MemCore_inner

module MemCore_inner_W (
  input logic [31:0] CONFIG_SPACE_0,
  input logic [31:0] CONFIG_SPACE_1,
  input logic [31:0] CONFIG_SPACE_10,
  input logic [31:0] CONFIG_SPACE_11,
  input logic [31:0] CONFIG_SPACE_12,
  input logic [31:0] CONFIG_SPACE_13,
  input logic [31:0] CONFIG_SPACE_14,
  input logic [31:0] CONFIG_SPACE_15,
  input logic [31:0] CONFIG_SPACE_16,
  input logic [31:0] CONFIG_SPACE_17,
  input logic [31:0] CONFIG_SPACE_18,
  input logic [31:0] CONFIG_SPACE_19,
  input logic [31:0] CONFIG_SPACE_2,
  input logic [31:0] CONFIG_SPACE_20,
  input logic [31:0] CONFIG_SPACE_21,
  input logic [31:0] CONFIG_SPACE_22,
  input logic [31:0] CONFIG_SPACE_23,
  input logic [31:0] CONFIG_SPACE_24,
  input logic [31:0] CONFIG_SPACE_25,
  input logic [31:0] CONFIG_SPACE_26,
  input logic [31:0] CONFIG_SPACE_27,
  input logic [31:0] CONFIG_SPACE_28,
  input logic [31:0] CONFIG_SPACE_29,
  input logic [31:0] CONFIG_SPACE_3,
  input logic [31:0] CONFIG_SPACE_30,
  input logic [31:0] CONFIG_SPACE_31,
  input logic [31:0] CONFIG_SPACE_32,
  input logic [31:0] CONFIG_SPACE_33,
  input logic [31:0] CONFIG_SPACE_34,
  input logic [31:0] CONFIG_SPACE_35,
  input logic [31:0] CONFIG_SPACE_36,
  input logic [31:0] CONFIG_SPACE_37,
  input logic [31:0] CONFIG_SPACE_38,
  input logic [31:0] CONFIG_SPACE_39,
  input logic [31:0] CONFIG_SPACE_4,
  input logic [31:0] CONFIG_SPACE_40,
  input logic [31:0] CONFIG_SPACE_41,
  input logic [31:0] CONFIG_SPACE_42,
  input logic [31:0] CONFIG_SPACE_43,
  input logic [31:0] CONFIG_SPACE_44,
  input logic [18:0] CONFIG_SPACE_45,
  input logic [31:0] CONFIG_SPACE_5,
  input logic [31:0] CONFIG_SPACE_6,
  input logic [31:0] CONFIG_SPACE_7,
  input logic [31:0] CONFIG_SPACE_8,
  input logic [31:0] CONFIG_SPACE_9,
  input logic [0:0] [16:0] MEM_input_width_17_num_0,
  input logic MEM_input_width_17_num_0_valid,
  input logic [0:0] [16:0] MEM_input_width_17_num_1,
  input logic MEM_input_width_17_num_1_valid,
  input logic [0:0] [16:0] MEM_input_width_17_num_2,
  input logic MEM_input_width_17_num_2_valid,
  input logic [0:0] [16:0] MEM_input_width_17_num_3,
  input logic MEM_input_width_17_num_3_valid,
  input logic MEM_input_width_1_num_0,
  input logic MEM_input_width_1_num_1,
  input logic MEM_output_width_17_num_0_ready,
  input logic MEM_output_width_17_num_1_ready,
  input logic MEM_output_width_17_num_2_ready,
  input logic clk,
  input logic clk_en,
  input logic [7:0] config_addr_in,
  input logic [31:0] config_data_in,
  input logic [1:0] config_en,
  input logic config_read,
  input logic config_write,
  input logic flush,
  input logic [1:0] mode,
  input logic mode_excl,
  input logic rst_n,
  input logic tile_en,
  output logic MEM_input_width_17_num_0_ready,
  output logic MEM_input_width_17_num_1_ready,
  output logic MEM_input_width_17_num_2_ready,
  output logic MEM_input_width_17_num_3_ready,
  output logic [0:0] [16:0] MEM_output_width_17_num_0,
  output logic MEM_output_width_17_num_0_valid,
  output logic [0:0] [16:0] MEM_output_width_17_num_1,
  output logic MEM_output_width_17_num_1_valid,
  output logic [0:0] [16:0] MEM_output_width_17_num_2,
  output logic MEM_output_width_17_num_2_valid,
  output logic MEM_output_width_1_num_0,
  output logic MEM_output_width_1_num_1,
  output logic MEM_output_width_1_num_2,
  output logic [31:0] config_data_out_0,
  output logic [31:0] config_data_out_1
);

logic [1:0][31:0] MemCore_inner_config_data_out;
assign config_data_out_0 = MemCore_inner_config_data_out[0];
assign config_data_out_1 = MemCore_inner_config_data_out[1];
MemCore_inner MemCore_inner (
  .CONFIG_SPACE_0(CONFIG_SPACE_0),
  .CONFIG_SPACE_1(CONFIG_SPACE_1),
  .CONFIG_SPACE_10(CONFIG_SPACE_10),
  .CONFIG_SPACE_11(CONFIG_SPACE_11),
  .CONFIG_SPACE_12(CONFIG_SPACE_12),
  .CONFIG_SPACE_13(CONFIG_SPACE_13),
  .CONFIG_SPACE_14(CONFIG_SPACE_14),
  .CONFIG_SPACE_15(CONFIG_SPACE_15),
  .CONFIG_SPACE_16(CONFIG_SPACE_16),
  .CONFIG_SPACE_17(CONFIG_SPACE_17),
  .CONFIG_SPACE_18(CONFIG_SPACE_18),
  .CONFIG_SPACE_19(CONFIG_SPACE_19),
  .CONFIG_SPACE_2(CONFIG_SPACE_2),
  .CONFIG_SPACE_20(CONFIG_SPACE_20),
  .CONFIG_SPACE_21(CONFIG_SPACE_21),
  .CONFIG_SPACE_22(CONFIG_SPACE_22),
  .CONFIG_SPACE_23(CONFIG_SPACE_23),
  .CONFIG_SPACE_24(CONFIG_SPACE_24),
  .CONFIG_SPACE_25(CONFIG_SPACE_25),
  .CONFIG_SPACE_26(CONFIG_SPACE_26),
  .CONFIG_SPACE_27(CONFIG_SPACE_27),
  .CONFIG_SPACE_28(CONFIG_SPACE_28),
  .CONFIG_SPACE_29(CONFIG_SPACE_29),
  .CONFIG_SPACE_3(CONFIG_SPACE_3),
  .CONFIG_SPACE_30(CONFIG_SPACE_30),
  .CONFIG_SPACE_31(CONFIG_SPACE_31),
  .CONFIG_SPACE_32(CONFIG_SPACE_32),
  .CONFIG_SPACE_33(CONFIG_SPACE_33),
  .CONFIG_SPACE_34(CONFIG_SPACE_34),
  .CONFIG_SPACE_35(CONFIG_SPACE_35),
  .CONFIG_SPACE_36(CONFIG_SPACE_36),
  .CONFIG_SPACE_37(CONFIG_SPACE_37),
  .CONFIG_SPACE_38(CONFIG_SPACE_38),
  .CONFIG_SPACE_39(CONFIG_SPACE_39),
  .CONFIG_SPACE_4(CONFIG_SPACE_4),
  .CONFIG_SPACE_40(CONFIG_SPACE_40),
  .CONFIG_SPACE_41(CONFIG_SPACE_41),
  .CONFIG_SPACE_42(CONFIG_SPACE_42),
  .CONFIG_SPACE_43(CONFIG_SPACE_43),
  .CONFIG_SPACE_44(CONFIG_SPACE_44),
  .CONFIG_SPACE_45(CONFIG_SPACE_45),
  .CONFIG_SPACE_5(CONFIG_SPACE_5),
  .CONFIG_SPACE_6(CONFIG_SPACE_6),
  .CONFIG_SPACE_7(CONFIG_SPACE_7),
  .CONFIG_SPACE_8(CONFIG_SPACE_8),
  .CONFIG_SPACE_9(CONFIG_SPACE_9),
  .MEM_input_width_17_num_0(MEM_input_width_17_num_0),
  .MEM_input_width_17_num_0_valid(MEM_input_width_17_num_0_valid),
  .MEM_input_width_17_num_1(MEM_input_width_17_num_1),
  .MEM_input_width_17_num_1_valid(MEM_input_width_17_num_1_valid),
  .MEM_input_width_17_num_2(MEM_input_width_17_num_2),
  .MEM_input_width_17_num_2_valid(MEM_input_width_17_num_2_valid),
  .MEM_input_width_17_num_3(MEM_input_width_17_num_3),
  .MEM_input_width_17_num_3_valid(MEM_input_width_17_num_3_valid),
  .MEM_input_width_1_num_0(MEM_input_width_1_num_0),
  .MEM_input_width_1_num_1(MEM_input_width_1_num_1),
  .MEM_output_width_17_num_0_ready(MEM_output_width_17_num_0_ready),
  .MEM_output_width_17_num_1_ready(MEM_output_width_17_num_1_ready),
  .MEM_output_width_17_num_2_ready(MEM_output_width_17_num_2_ready),
  .clk(clk),
  .clk_en(clk_en),
  .config_addr_in(config_addr_in),
  .config_data_in(config_data_in),
  .config_en(config_en),
  .config_read(config_read),
  .config_write(config_write),
  .flush(flush),
  .mode(mode),
  .mode_excl(mode_excl),
  .rst_n(rst_n),
  .tile_en(tile_en),
  .MEM_input_width_17_num_0_ready(MEM_input_width_17_num_0_ready),
  .MEM_input_width_17_num_1_ready(MEM_input_width_17_num_1_ready),
  .MEM_input_width_17_num_2_ready(MEM_input_width_17_num_2_ready),
  .MEM_input_width_17_num_3_ready(MEM_input_width_17_num_3_ready),
  .MEM_output_width_17_num_0(MEM_output_width_17_num_0),
  .MEM_output_width_17_num_0_valid(MEM_output_width_17_num_0_valid),
  .MEM_output_width_17_num_1(MEM_output_width_17_num_1),
  .MEM_output_width_17_num_1_valid(MEM_output_width_17_num_1_valid),
  .MEM_output_width_17_num_2(MEM_output_width_17_num_2),
  .MEM_output_width_17_num_2_valid(MEM_output_width_17_num_2_valid),
  .MEM_output_width_1_num_0(MEM_output_width_1_num_0),
  .MEM_output_width_1_num_1(MEM_output_width_1_num_1),
  .MEM_output_width_1_num_2(MEM_output_width_1_num_2),
  .config_data_out(MemCore_inner_config_data_out)
);

endmodule   // MemCore_inner_W

module addr_gen_3_16 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [1:0] mux_sel,
  input logic restart,
  input logic rst_n,
  input logic [15:0] starting_addr,
  input logic step,
  input logic [2:0] [15:0] strides,
  output logic [15:0] addr_out
);

logic [15:0] calc_addr;
logic [15:0] current_addr;
logic [15:0] strt_addr;
assign strt_addr = starting_addr;
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= strt_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= strt_addr;
      end
      else current_addr <= current_addr + strides[mux_sel];
    end
  end
end
endmodule   // addr_gen_3_16

module addr_gen_3_3 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [1:0] mux_sel,
  input logic restart,
  input logic rst_n,
  input logic [2:0] starting_addr,
  input logic step,
  input logic [2:0] [2:0] strides,
  output logic [2:0] addr_out
);

logic [2:0] calc_addr;
logic [2:0] current_addr;
logic [2:0] strt_addr;
assign strt_addr = starting_addr;
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 3'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= strt_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= strt_addr;
      end
      else current_addr <= current_addr + strides[mux_sel];
    end
  end
end
endmodule   // addr_gen_3_3

module addr_gen_6_16 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic restart,
  input logic rst_n,
  input logic [15:0] starting_addr,
  input logic step,
  input logic [5:0] [15:0] strides,
  output logic [15:0] addr_out
);

logic [15:0] calc_addr;
logic [15:0] current_addr;
logic [15:0] strt_addr;
assign strt_addr = starting_addr;
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= strt_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= strt_addr;
      end
      else current_addr <= current_addr + strides[mux_sel];
    end
  end
end
endmodule   // addr_gen_6_16

module addr_gen_6_16_delay_addr_10 (
  input logic clk,
  input logic clk_en,
  input logic [9:0] delay,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic restart,
  input logic rst_n,
  input logic [15:0] starting_addr,
  input logic step,
  input logic [5:0] [15:0] strides,
  output logic [15:0] addr_out,
  output logic [9:0] delay_out,
  output logic [15:0] delayed_addr_out
);

logic [15:0] calc_addr;
logic [15:0] current_addr;
logic [15:0] strt_addr;
assign delay_out = delay;
assign strt_addr = starting_addr;
assign addr_out = calc_addr;
assign calc_addr = current_addr;
assign delayed_addr_out = current_addr + 16'(delay);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= strt_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= strt_addr;
      end
      else current_addr <= current_addr + strides[mux_sel];
    end
  end
end
endmodule   // addr_gen_6_16_delay_addr_10

module addr_gen_6_4 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic restart,
  input logic rst_n,
  input logic [3:0] starting_addr,
  input logic step,
  input logic [5:0] [3:0] strides,
  output logic [3:0] addr_out
);

logic [3:0] calc_addr;
logic [3:0] current_addr;
logic [3:0] strt_addr;
assign strt_addr = starting_addr;
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 4'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= strt_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= strt_addr;
      end
      else current_addr <= current_addr + strides[mux_sel];
    end
  end
end
endmodule   // addr_gen_6_4

module addr_gen_6_9 (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic restart,
  input logic rst_n,
  input logic [8:0] starting_addr,
  input logic step,
  input logic [5:0] [8:0] strides,
  output logic [8:0] addr_out
);

logic [8:0] calc_addr;
logic [8:0] current_addr;
logic [8:0] strt_addr;
assign strt_addr = starting_addr;
assign addr_out = calc_addr;
assign calc_addr = current_addr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    current_addr <= 9'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      current_addr <= strt_addr;
    end
    else if (step) begin
      if (restart) begin
        current_addr <= strt_addr;
      end
      else current_addr <= current_addr + strides[mux_sel];
    end
  end
end
endmodule   // addr_gen_6_9

module agg_sram_shared_addr_gen (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [1:0] mode,
  input logic rst_n,
  input logic [1:0] sram_read,
  input logic [1:0] [8:0] sram_read_addr,
  input logic [8:0] starting_addr,
  input logic step,
  output logic [8:0] addr_out
);

logic [3:0][8:0] addr_fifo;
logic [8:0] addr_fifo_in;
logic [8:0] addr_fifo_out;
logic addr_fifo_wr_en;
logic [8:0] lin_addr_cnter;
logic [1:0] rd_ptr;
logic [1:0] wr_ptr;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    lin_addr_cnter <= 9'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      lin_addr_cnter <= 9'h0;
    end
    else if (mode[1] == 1'h0) begin
      if (step) begin
        if (lin_addr_cnter == 9'h1FF) begin
          lin_addr_cnter <= 9'h0;
        end
        else lin_addr_cnter <= lin_addr_cnter + 9'h1;
      end
    end
  end
end
assign addr_fifo_wr_en = mode[0] ? sram_read[1]: sram_read[0];
assign addr_fifo_in = mode[0] ? sram_read_addr[1]: sram_read_addr[0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr <= 2'h0;
    rd_ptr <= 2'h0;
    addr_fifo <= 36'h0;
    addr_fifo_out <= 9'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      wr_ptr <= 2'h0;
      rd_ptr <= 2'h0;
      addr_fifo <= 36'h0;
      addr_fifo_out <= 9'h0;
    end
    else if (mode[1] == 1'h1) begin
      if (addr_fifo_wr_en) begin
        wr_ptr <= wr_ptr + 2'h1;
        addr_fifo[wr_ptr] <= addr_fifo_in;
      end
      if (step) begin
        rd_ptr <= rd_ptr + 2'h1;
      end
      addr_fifo_out <= addr_fifo[rd_ptr];
    end
  end
end
assign addr_out = mode[1] ? addr_fifo_out: lin_addr_cnter + starting_addr;
endmodule   // agg_sram_shared_addr_gen

module agg_sram_shared_sched_gen (
  input logic [7:0] agg_read_padding,
  input logic agg_write,
  input logic [1:0] agg_write_addr_l2b,
  input logic [2:0] agg_write_mux_sel,
  input logic agg_write_restart,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [1:0] mode,
  input logic rst_n,
  input logic [1:0] sram_read_d,
  output logic valid_output
);

logic agg_write_4_r;
logic [7:0] pad_cnt;
logic pad_cnt_en;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    agg_write_4_r <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      agg_write_4_r <= 1'h0;
    end
    else if (mode[1] == 1'h0) begin
      agg_write_4_r <= agg_write & (&agg_write_addr_l2b);
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pad_cnt_en <= 1'h0;
    pad_cnt <= 8'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pad_cnt_en <= 1'h0;
      pad_cnt <= 8'h0;
    end
    else if ((mode[1] == 1'h0) & (agg_read_padding != 8'h0)) begin
      if (agg_write & ((agg_write_mux_sel != 3'h0) | agg_write_restart)) begin
        pad_cnt_en <= 1'h1;
      end
      else if (pad_cnt == agg_read_padding) begin
        pad_cnt_en <= 1'h0;
      end
      if (pad_cnt == agg_read_padding) begin
        pad_cnt <= 8'h0;
      end
      else if (pad_cnt_en | (agg_write & ((agg_write_mux_sel != 3'h0) | agg_write_restart))) begin
        pad_cnt <= pad_cnt + 8'h1;
      end
    end
  end
end
always_comb begin
  if (mode[1] == 1'h0) begin
    if (agg_read_padding != 8'h0) begin
      valid_output = (agg_read_padding == pad_cnt) | agg_write_4_r;
    end
    else valid_output = agg_write_4_r;
  end
  else valid_output = mode[0] ? sram_read_d[1]: sram_read_d[0];
end
endmodule   // agg_sram_shared_sched_gen

module arbiter_2_in_PRIO_algo (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [1:0] request_in,
  input logic resource_ready,
  input logic rst_n,
  output logic [1:0] grant_out
);

logic [1:0] grant_line;
logic [1:0] grant_line_ready;
logic [1:0] grant_out_consolation;
logic [1:0] grant_out_priority;
logic tmp_done;
logic tmp_out_first;
always_comb begin
  grant_line = request_in[1] ? 2'h2: 2'h1;
end
assign grant_line_ready[0] = grant_line[0] & resource_ready;
assign grant_out_priority[0] = grant_line_ready[0] & request_in[0];
assign grant_line_ready[1] = grant_line[1] & resource_ready;
assign grant_out_priority[1] = grant_line_ready[1] & request_in[1];
always_comb begin
  tmp_done = 1'h0;
  tmp_out_first = 1'h0;
  if (~tmp_done) begin
    if (request_in[0]) begin
      tmp_out_first = 1'h0;
      tmp_done = 1'h1;
    end
  end
  if (~tmp_done) begin
    if (request_in[1]) begin
      tmp_out_first = 1'h1;
      tmp_done = 1'h1;
    end
  end
end
assign grant_out_consolation[0] = resource_ready & request_in[0] & (tmp_out_first == 1'h0);
assign grant_out[0] = (|grant_out_priority) ? grant_out_priority[0]: grant_out_consolation[0];
assign grant_out_consolation[1] = resource_ready & request_in[1] & (tmp_out_first == 1'h1);
assign grant_out[1] = (|grant_out_priority) ? grant_out_priority[1]: grant_out_consolation[1];
endmodule   // arbiter_2_in_PRIO_algo

module arbiter_4_in_RR_algo (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [3:0] request_in,
  input logic resource_ready,
  input logic rst_n,
  output logic [3:0] grant_out
);

logic [3:0] grant_line;
logic [3:0] grant_line_ready;
logic [3:0] grant_out_consolation;
logic [3:0] grant_out_priority;
logic tmp_done;
logic [1:0] tmp_out_first;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    grant_line <= 4'h1;
  end
  else if (clk_en) begin
    if (flush) begin
      grant_line <= 4'h1;
    end
    else grant_line <= {grant_line[2:0], grant_line[3]};
  end
end
assign grant_line_ready[0] = grant_line[0] & resource_ready;
assign grant_out_priority[0] = grant_line_ready[0] & request_in[0];
assign grant_line_ready[1] = grant_line[1] & resource_ready;
assign grant_out_priority[1] = grant_line_ready[1] & request_in[1];
assign grant_line_ready[2] = grant_line[2] & resource_ready;
assign grant_out_priority[2] = grant_line_ready[2] & request_in[2];
assign grant_line_ready[3] = grant_line[3] & resource_ready;
assign grant_out_priority[3] = grant_line_ready[3] & request_in[3];
always_comb begin
  tmp_done = 1'h0;
  tmp_out_first = 2'h0;
  if (~tmp_done) begin
    if (request_in[0]) begin
      tmp_out_first = 2'h0;
      tmp_done = 1'h1;
    end
  end
  if (~tmp_done) begin
    if (request_in[1]) begin
      tmp_out_first = 2'h1;
      tmp_done = 1'h1;
    end
  end
  if (~tmp_done) begin
    if (request_in[2]) begin
      tmp_out_first = 2'h2;
      tmp_done = 1'h1;
    end
  end
  if (~tmp_done) begin
    if (request_in[3]) begin
      tmp_out_first = 2'h3;
      tmp_done = 1'h1;
    end
  end
end
assign grant_out_consolation[0] = resource_ready & request_in[0] & (tmp_out_first == 2'h0);
assign grant_out[0] = (|grant_out_priority) ? grant_out_priority[0]: grant_out_consolation[0];
assign grant_out_consolation[1] = resource_ready & request_in[1] & (tmp_out_first == 2'h1);
assign grant_out[1] = (|grant_out_priority) ? grant_out_priority[1]: grant_out_consolation[1];
assign grant_out_consolation[2] = resource_ready & request_in[2] & (tmp_out_first == 2'h2);
assign grant_out[2] = (|grant_out_priority) ? grant_out_priority[2]: grant_out_consolation[2];
assign grant_out_consolation[3] = resource_ready & request_in[3] & (tmp_out_first == 2'h3);
assign grant_out[3] = (|grant_out_priority) ? grant_out_priority[3]: grant_out_consolation[3];
endmodule   // arbiter_4_in_RR_algo

module buffet_like_16 (
  input logic [1:0] [3:0] buffet_capacity_log,
  input logic clk,
  input logic clk_en,
  input logic [3:0] [15:0] data_from_mem,
  input logic flush,
  input logic [0:0] [16:0] rd_addr_0,
  input logic rd_addr_0_valid,
  input logic [0:0] [16:0] rd_addr_1,
  input logic rd_addr_1_valid,
  input logic [0:0] [16:0] rd_op_0,
  input logic rd_op_0_valid,
  input logic [0:0] [16:0] rd_op_1,
  input logic rd_op_1_valid,
  input logic rd_rsp_data_0_ready,
  input logic rd_rsp_data_1_ready,
  input logic rst_n,
  input logic tile_en,
  input logic [0:0] [16:0] wr_ID,
  input logic wr_ID_valid,
  input logic [0:0] [16:0] wr_addr,
  input logic wr_addr_valid,
  input logic [0:0] [16:0] wr_data,
  input logic wr_data_valid,
  output logic [8:0] addr_to_mem,
  output logic [3:0] [15:0] data_to_mem,
  output logic rd_addr_0_ready,
  output logic rd_addr_1_ready,
  output logic rd_op_0_ready,
  output logic rd_op_1_ready,
  output logic [0:0] [16:0] rd_rsp_data_0,
  output logic rd_rsp_data_0_valid,
  output logic [0:0] [16:0] rd_rsp_data_1,
  output logic rd_rsp_data_1_valid,
  output logic ren_to_mem,
  output logic wen_to_mem,
  output logic wr_ID_ready,
  output logic wr_addr_ready,
  output logic wr_data_ready
);

typedef enum logic[1:0] {
  RD_PAUSE_0 = 2'h0,
  RD_PAUSE_T_0 = 2'h1,
  RD_START_0 = 2'h2
} read_fsm_0_state;
typedef enum logic[1:0] {
  RD_PAUSE_1 = 2'h0,
  RD_PAUSE_T_1 = 2'h1,
  RD_START_1 = 2'h2
} read_fsm_1_state;
typedef enum logic[1:0] {
  MODIFY_0 = 2'h0,
  WRITING_0 = 2'h1,
  WR_START_0 = 2'h2
} write_fsm_0_state;
typedef enum logic[1:0] {
  MODIFY_1 = 2'h0,
  WRITING_1 = 2'h1,
  WR_START_1 = 2'h2
} write_fsm_1_state;
logic PREVIOUS_WR_OP;
logic [15:0] addr_to_mem_local;
logic any_sram_lock;
logic [3:0] base_rr;
logic [1:0][15:0] blk_base;
logic [1:0][15:0] blk_bounds;
logic [7:0] blk_count_0;
logic [7:0] blk_count_1;
logic [0:0][31:0] blk_fifo_0_data_in;
logic [0:0][31:0] blk_fifo_0_data_out;
logic blk_fifo_0_empty;
logic blk_fifo_0_full;
logic [0:0][31:0] blk_fifo_1_data_in;
logic [0:0][31:0] blk_fifo_1_data_out;
logic blk_fifo_1_empty;
logic blk_fifo_1_full;
logic [1:0] blk_full;
logic [1:0] blk_valid;
logic [1:0][15:0] buffet_base;
logic [1:0][15:0] buffet_capacity;
logic [1:0][15:0] buffet_capacity_mask;
logic [15:0] cached_read_word_addr_0;
logic [15:0] cached_read_word_addr_1;
logic [15:0] chosen_read_0;
logic [15:0] chosen_read_1;
logic clr_cached_read_0;
logic clr_cached_read_1;
logic clr_write_wide_word_0;
logic clr_write_wide_word_1;
logic [15:0] curr_base_0;
logic [15:0] curr_base_1;
logic [15:0] curr_base_pre_0;
logic [15:0] curr_base_pre_1;
logic [15:0] curr_bounds_0;
logic [15:0] curr_bounds_1;
logic [1:0][15:0] curr_capacity_pre;
logic [15:0] decode_ret_size_request_full_blk_bounds;
logic decode_sel_done_size_request_full_blk_bounds;
logic [1:0] en_curr_base;
logic [1:0] en_curr_bounds;
logic first_base_set_0_sticky;
logic first_base_set_0_was_high;
logic first_base_set_1_sticky;
logic first_base_set_1_was_high;
logic from_cached_read_0;
logic from_cached_read_1;
logic gclk;
logic joined_in_fifo;
logic [1:0] last_read_addr_0;
logic [1:0] last_read_addr_1;
logic [15:0] last_read_addr_wide_0;
logic [15:0] last_read_addr_wide_1;
logic [3:0] mem_acq;
logic [2:0] num_bits_valid_mask_0_sum;
logic [2:0] num_bits_valid_mask_1_sum;
logic [1:0] pop_blk;
logic pop_in_fifos;
logic [1:0] pop_in_full;
logic [1:0] push_blk;
logic rd_acq_0;
logic rd_acq_1;
logic rd_addr_fifo_0_empty;
logic rd_addr_fifo_0_full;
logic rd_addr_fifo_1_empty;
logic rd_addr_fifo_1_full;
logic [15:0] rd_addr_fifo_out_addr_0;
logic [15:0] rd_addr_fifo_out_addr_1;
logic rd_addr_fifo_pop_0;
logic rd_addr_fifo_pop_1;
logic rd_addr_fifo_valid_0;
logic rd_addr_fifo_valid_1;
logic rd_op_fifo_0_empty;
logic rd_op_fifo_0_full;
logic rd_op_fifo_1_empty;
logic rd_op_fifo_1_full;
logic [15:0] rd_op_fifo_out_op_0;
logic [15:0] rd_op_fifo_out_op_1;
logic rd_op_fifo_pop_0;
logic rd_op_fifo_pop_1;
logic rd_op_fifo_valid_0;
logic rd_op_fifo_valid_1;
logic rd_rsp_fifo_0_almost_full;
logic rd_rsp_fifo_0_empty;
logic rd_rsp_fifo_0_full;
logic [16:0] rd_rsp_fifo_0_in_data;
logic rd_rsp_fifo_0_push;
logic rd_rsp_fifo_1_almost_full;
logic rd_rsp_fifo_1_empty;
logic rd_rsp_fifo_1_full;
logic [16:0] rd_rsp_fifo_1_in_data;
logic rd_rsp_fifo_1_push;
logic [15:0] read_addr_delayed_0;
logic [15:0] read_addr_delayed_1;
logic read_d1;
logic read_from_sram_write_side_0;
logic read_from_sram_write_side_1;
read_fsm_0_state read_fsm_0_current_state;
read_fsm_0_state read_fsm_0_next_state;
read_fsm_1_state read_fsm_1_current_state;
read_fsm_1_state read_fsm_1_next_state;
logic read_joined_0;
logic read_joined_1;
logic read_joined_d1_0;
logic read_joined_d1_1;
logic read_pop_0;
logic read_pop_1;
logic [1:0] read_pop_full;
logic [3:0][15:0] read_wide_word_0;
logic [3:0][15:0] read_wide_word_1;
logic read_wide_word_valid_sticky_0_sticky;
logic read_wide_word_valid_sticky_0_was_high;
logic read_wide_word_valid_sticky_1_sticky;
logic read_wide_word_valid_sticky_1_was_high;
logic [1:0] ren_full;
logic ren_full_delayed_0;
logic ren_full_delayed_1;
logic rr_arbiter_resource_ready;
logic set_cached_read_0;
logic set_cached_read_1;
logic set_read_word_addr_0;
logic set_read_word_addr_1;
logic set_wide_word_addr_0;
logic set_wide_word_addr_1;
logic set_write_wide_word_0;
logic set_write_wide_word_1;
logic [1:0] size_request_full;
logic sram_lock_0;
logic sram_lock_1;
logic [15:0] tmp_addr_0;
logic [15:0] tmp_addr_1;
logic [15:0] tmp_rd_base;
logic [15:0] tmp_wr_base;
logic use_cached_read_0;
logic use_cached_read_1;
logic valid_from_mem;
logic [1:0] wen_full;
logic wr_ID_fifo_empty;
logic wr_ID_fifo_full;
logic [15:0] wr_ID_fifo_out_data;
logic wr_ID_fifo_pop;
logic wr_ID_fifo_valid;
logic wr_acq_0;
logic wr_acq_1;
logic wr_addr_fifo_empty;
logic wr_addr_fifo_full;
logic [15:0] wr_addr_fifo_out_data;
logic wr_addr_fifo_pop;
logic wr_addr_fifo_valid;
logic [0:0][16:0] wr_data_fifo_data_out;
logic wr_data_fifo_empty;
logic wr_data_fifo_full;
logic [15:0] wr_data_fifo_out_data;
logic wr_data_fifo_out_op;
logic wr_data_fifo_pop;
logic wr_data_fifo_valid;
write_fsm_0_state write_fsm_0_current_state;
write_fsm_0_state write_fsm_0_next_state;
write_fsm_1_state write_fsm_1_current_state;
write_fsm_1_state write_fsm_1_next_state;
logic write_full_word_0;
logic write_full_word_1;
logic write_to_sram_0;
logic write_to_sram_1;
logic [3:0][15:0] write_wide_word_comb_in_0;
logic [3:0][15:0] write_wide_word_comb_in_1;
logic [3:0][15:0] write_wide_word_comb_out_0;
logic [3:0][15:0] write_wide_word_comb_out_1;
logic [3:0] write_wide_word_mask_comb_0;
logic [3:0] write_wide_word_mask_comb_1;
logic [3:0] write_wide_word_mask_reg_in_0;
logic [3:0] write_wide_word_mask_reg_in_1;
logic [3:0] write_wide_word_mask_reg_out_0;
logic [3:0] write_wide_word_mask_reg_out_1;
logic [3:0] write_wide_word_mask_reg_strg_0;
logic [3:0] write_wide_word_mask_reg_strg_1;
logic [3:0][15:0] write_wide_word_modified_0;
logic [3:0][15:0] write_wide_word_modified_1;
logic [3:0][15:0] write_wide_word_reg_0;
logic [3:0][15:0] write_wide_word_reg_1;
logic [15:0] write_word_addr_reg_0;
logic [15:0] write_word_addr_reg_1;
logic write_word_addr_valid_sticky_0_sticky;
logic write_word_addr_valid_sticky_0_was_high;
logic write_word_addr_valid_sticky_1_sticky;
logic write_word_addr_valid_sticky_1_was_high;
assign gclk = clk & tile_en;
assign buffet_capacity_mask[0][0] = (buffet_capacity_log[0] > 4'h0) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][1] = (buffet_capacity_log[0] > 4'h1) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][2] = (buffet_capacity_log[0] > 4'h2) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][3] = (buffet_capacity_log[0] > 4'h3) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][4] = (buffet_capacity_log[0] > 4'h4) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][5] = (buffet_capacity_log[0] > 4'h5) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][6] = (buffet_capacity_log[0] > 4'h6) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][7] = (buffet_capacity_log[0] > 4'h7) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][8] = (buffet_capacity_log[0] > 4'h8) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][9] = (buffet_capacity_log[0] > 4'h9) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][10] = (buffet_capacity_log[0] > 4'hA) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][11] = (buffet_capacity_log[0] > 4'hB) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][12] = (buffet_capacity_log[0] > 4'hC) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][13] = (buffet_capacity_log[0] > 4'hD) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][14] = (buffet_capacity_log[0] > 4'hE) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[0][15] = (buffet_capacity_log[0] > 4'hF) & (buffet_capacity_log[0] != 4'h0);
assign buffet_capacity_mask[1][0] = (buffet_capacity_log[1] > 4'h0) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][1] = (buffet_capacity_log[1] > 4'h1) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][2] = (buffet_capacity_log[1] > 4'h2) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][3] = (buffet_capacity_log[1] > 4'h3) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][4] = (buffet_capacity_log[1] > 4'h4) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][5] = (buffet_capacity_log[1] > 4'h5) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][6] = (buffet_capacity_log[1] > 4'h6) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][7] = (buffet_capacity_log[1] > 4'h7) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][8] = (buffet_capacity_log[1] > 4'h8) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][9] = (buffet_capacity_log[1] > 4'h9) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][10] = (buffet_capacity_log[1] > 4'hA) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][11] = (buffet_capacity_log[1] > 4'hB) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][12] = (buffet_capacity_log[1] > 4'hC) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][13] = (buffet_capacity_log[1] > 4'hD) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][14] = (buffet_capacity_log[1] > 4'hE) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity_mask[1][15] = (buffet_capacity_log[1] > 4'hF) & (buffet_capacity_log[1] != 4'h0);
assign buffet_capacity[0] = (buffet_capacity_log[0] == 4'h0) ? 16'h0: 16'h1 << 16'(buffet_capacity_log[0] +
    4'h2);
assign buffet_capacity[1] = (buffet_capacity_log[1] == 4'h0) ? 16'h0: 16'h1 << 16'(buffet_capacity_log[1] +
    4'h2);
assign buffet_base[0] = 16'h0;
assign buffet_base[1] = 16'h100;
assign {wr_data_fifo_out_op, wr_data_fifo_out_data} = wr_data_fifo_data_out;
assign wr_data_ready = ~wr_data_fifo_full;
assign wr_data_fifo_valid = ~wr_data_fifo_empty;
assign wr_addr_ready = ~wr_addr_fifo_full;
assign wr_addr_fifo_valid = ~wr_addr_fifo_empty;
assign wr_ID_ready = ~wr_ID_fifo_full;
assign wr_ID_fifo_valid = ~wr_ID_fifo_empty;
assign rd_op_0_ready = ~rd_op_fifo_0_full;
assign rd_op_1_ready = ~rd_op_fifo_1_full;
assign rd_op_fifo_valid_0 = ~rd_op_fifo_0_empty;
assign rd_op_fifo_valid_1 = ~rd_op_fifo_1_empty;
assign rd_addr_0_ready = ~rd_addr_fifo_0_full;
assign rd_addr_1_ready = ~rd_addr_fifo_1_full;
assign rd_addr_fifo_valid_0 = ~rd_addr_fifo_0_empty;
assign rd_addr_fifo_valid_1 = ~rd_addr_fifo_1_empty;
assign read_joined_0 = rd_op_fifo_valid_0 & rd_addr_fifo_valid_0;
assign read_joined_1 = rd_op_fifo_valid_1 & rd_addr_fifo_valid_1;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    curr_bounds_0 <= 16'hFFFF;
  end
  else if (clk_en) begin
    if (flush) begin
      curr_bounds_0 <= 16'hFFFF;
    end
    else if (1'h0) begin
      curr_bounds_0 <= 16'h0;
    end
    else if (en_curr_bounds[0]) begin
      curr_bounds_0 <= wr_addr_fifo_out_data;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    curr_bounds_1 <= 16'hFFFF;
  end
  else if (clk_en) begin
    if (flush) begin
      curr_bounds_1 <= 16'hFFFF;
    end
    else if (1'h0) begin
      curr_bounds_1 <= 16'h0;
    end
    else if (en_curr_bounds[1]) begin
      curr_bounds_1 <= wr_addr_fifo_out_data;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    first_base_set_0_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      first_base_set_0_was_high <= 1'h0;
    end
    else if (1'h0) begin
      first_base_set_0_was_high <= 1'h0;
    end
    else if (en_curr_base[0]) begin
      first_base_set_0_was_high <= 1'h1;
    end
  end
end
assign first_base_set_0_sticky = first_base_set_0_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    first_base_set_1_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      first_base_set_1_was_high <= 1'h0;
    end
    else if (1'h0) begin
      first_base_set_1_was_high <= 1'h0;
    end
    else if (en_curr_base[1]) begin
      first_base_set_1_was_high <= 1'h1;
    end
  end
end
assign first_base_set_1_sticky = first_base_set_1_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    curr_base_0 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      curr_base_0 <= 16'h0;
    end
    else if (1'h0) begin
      curr_base_0 <= 16'h0;
    end
    else if (en_curr_base[0]) begin
      curr_base_0 <= curr_base_pre_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    curr_base_1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      curr_base_1 <= 16'h0;
    end
    else if (1'h0) begin
      curr_base_1 <= 16'h0;
    end
    else if (en_curr_base[1]) begin
      curr_base_1 <= curr_base_pre_1;
    end
  end
end
assign curr_base_pre_0 = 1'h1 ? (curr_bounds_0 >> 16'h2) + 16'h1 + curr_base_0: 16'h0;
assign curr_base_pre_1 = 1'h1 ? (curr_bounds_1 >> 16'h2) + 16'h1 + curr_base_1: 16'h0;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_joined_d1_0 <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_joined_d1_0 <= 1'h0;
    end
    else if (1'h0) begin
      read_joined_d1_0 <= 1'h0;
    end
    else if (1'h1) begin
      read_joined_d1_0 <= read_joined_0 & read_pop_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_joined_d1_1 <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_joined_d1_1 <= 1'h0;
    end
    else if (1'h0) begin
      read_joined_d1_1 <= 1'h0;
    end
    else if (1'h1) begin
      read_joined_d1_1 <= read_joined_1 & read_pop_1;
    end
  end
end
assign rd_acq_0 = mem_acq[1] & ren_full[0];
assign rd_acq_1 = mem_acq[3] & ren_full[1];
assign wr_acq_0 = mem_acq[0] & write_to_sram_0;
assign wr_acq_1 = mem_acq[2] & write_to_sram_1;
assign addr_to_mem = addr_to_mem_local[8:0];
assign tmp_addr_0 = ((16'(wr_addr_fifo_out_data[15:2]) + curr_base_0) & buffet_capacity_mask[0]) +
    buffet_base[0];
assign tmp_addr_1 = ((16'(wr_addr_fifo_out_data[15:2]) + curr_base_1) & buffet_capacity_mask[1]) +
    buffet_base[1];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_word_addr_reg_0 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_word_addr_reg_0 <= 16'h0;
    end
    else if (1'h0) begin
      write_word_addr_reg_0 <= 16'h0;
    end
    else if (set_wide_word_addr_0) begin
      write_word_addr_reg_0 <= tmp_addr_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_word_addr_reg_1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_word_addr_reg_1 <= 16'h0;
    end
    else if (1'h0) begin
      write_word_addr_reg_1 <= 16'h0;
    end
    else if (set_wide_word_addr_1) begin
      write_word_addr_reg_1 <= tmp_addr_1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_word_addr_valid_sticky_0_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_word_addr_valid_sticky_0_was_high <= 1'h0;
    end
    else if (1'h0) begin
      write_word_addr_valid_sticky_0_was_high <= 1'h0;
    end
    else if (set_wide_word_addr_0) begin
      write_word_addr_valid_sticky_0_was_high <= 1'h1;
    end
  end
end
assign write_word_addr_valid_sticky_0_sticky = write_word_addr_valid_sticky_0_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_word_addr_valid_sticky_1_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_word_addr_valid_sticky_1_was_high <= 1'h0;
    end
    else if (1'h0) begin
      write_word_addr_valid_sticky_1_was_high <= 1'h0;
    end
    else if (set_wide_word_addr_1) begin
      write_word_addr_valid_sticky_1_was_high <= 1'h1;
    end
  end
end
assign write_word_addr_valid_sticky_1_sticky = write_word_addr_valid_sticky_1_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_wide_word_mask_reg_strg_0 <= 4'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_wide_word_mask_reg_strg_0 <= 4'h0;
    end
    else if (1'h0) begin
      write_wide_word_mask_reg_strg_0 <= 4'h0;
    end
    else if (set_write_wide_word_0 | clr_write_wide_word_0) begin
      write_wide_word_mask_reg_strg_0 <= write_wide_word_mask_reg_in_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_wide_word_mask_reg_strg_1 <= 4'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_wide_word_mask_reg_strg_1 <= 4'h0;
    end
    else if (1'h0) begin
      write_wide_word_mask_reg_strg_1 <= 4'h0;
    end
    else if (set_write_wide_word_1 | clr_write_wide_word_1) begin
      write_wide_word_mask_reg_strg_1 <= write_wide_word_mask_reg_in_1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_wide_word_reg_0 <= 64'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_wide_word_reg_0 <= 64'h0;
    end
    else if (1'h0) begin
      write_wide_word_reg_0 <= 64'h0;
    end
    else if (set_write_wide_word_0 | clr_write_wide_word_0) begin
      write_wide_word_reg_0 <= write_wide_word_comb_in_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_wide_word_reg_1 <= 64'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_wide_word_reg_1 <= 64'h0;
    end
    else if (1'h0) begin
      write_wide_word_reg_1 <= 64'h0;
    end
    else if (set_write_wide_word_1 | clr_write_wide_word_1) begin
      write_wide_word_reg_1 <= write_wide_word_comb_in_1;
    end
  end
end
assign write_wide_word_comb_out_0[0] = write_wide_word_mask_reg_out_0[0] ? write_wide_word_reg_0[0]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_0[1] = write_wide_word_mask_reg_out_0[1] ? write_wide_word_reg_0[1]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_0[2] = write_wide_word_mask_reg_out_0[2] ? write_wide_word_reg_0[2]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_0[3] = write_wide_word_mask_reg_out_0[3] ? write_wide_word_reg_0[3]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_1[0] = write_wide_word_mask_reg_out_1[0] ? write_wide_word_reg_1[0]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_1[1] = write_wide_word_mask_reg_out_1[1] ? write_wide_word_reg_1[1]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_1[2] = write_wide_word_mask_reg_out_1[2] ? write_wide_word_reg_1[2]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_out_1[3] = write_wide_word_mask_reg_out_1[3] ? write_wide_word_reg_1[3]:
    wr_data_fifo_out_data;
assign write_wide_word_comb_in_0[0] = (write_wide_word_mask_reg_out_0[0] & (~clr_write_wide_word_0)) ?
    write_wide_word_reg_0[0]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_0[1] = (write_wide_word_mask_reg_out_0[1] & (~clr_write_wide_word_0)) ?
    write_wide_word_reg_0[1]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_0[2] = (write_wide_word_mask_reg_out_0[2] & (~clr_write_wide_word_0)) ?
    write_wide_word_reg_0[2]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_0[3] = (write_wide_word_mask_reg_out_0[3] & (~clr_write_wide_word_0)) ?
    write_wide_word_reg_0[3]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_1[0] = (write_wide_word_mask_reg_out_1[0] & (~clr_write_wide_word_1)) ?
    write_wide_word_reg_1[0]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_1[1] = (write_wide_word_mask_reg_out_1[1] & (~clr_write_wide_word_1)) ?
    write_wide_word_reg_1[1]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_1[2] = (write_wide_word_mask_reg_out_1[2] & (~clr_write_wide_word_1)) ?
    write_wide_word_reg_1[2]: wr_data_fifo_out_data;
assign write_wide_word_comb_in_1[3] = (write_wide_word_mask_reg_out_1[3] & (~clr_write_wide_word_1)) ?
    write_wide_word_reg_1[3]: wr_data_fifo_out_data;
assign write_wide_word_modified_0[0] = write_wide_word_mask_reg_out_0[0] ? write_wide_word_reg_0[0]: data_from_mem[0];
assign write_wide_word_modified_0[1] = write_wide_word_mask_reg_out_0[1] ? write_wide_word_reg_0[1]: data_from_mem[1];
assign write_wide_word_modified_0[2] = write_wide_word_mask_reg_out_0[2] ? write_wide_word_reg_0[2]: data_from_mem[2];
assign write_wide_word_modified_0[3] = write_wide_word_mask_reg_out_0[3] ? write_wide_word_reg_0[3]: data_from_mem[3];
assign write_wide_word_modified_1[0] = write_wide_word_mask_reg_out_1[0] ? write_wide_word_reg_1[0]: data_from_mem[0];
assign write_wide_word_modified_1[1] = write_wide_word_mask_reg_out_1[1] ? write_wide_word_reg_1[1]: data_from_mem[1];
assign write_wide_word_modified_1[2] = write_wide_word_mask_reg_out_1[2] ? write_wide_word_reg_1[2]: data_from_mem[2];
assign write_wide_word_modified_1[3] = write_wide_word_mask_reg_out_1[3] ? write_wide_word_reg_1[3]: data_from_mem[3];
assign write_wide_word_mask_reg_out_0 = write_wide_word_mask_reg_strg_0;
assign write_wide_word_mask_reg_out_1 = write_wide_word_mask_reg_strg_1;
assign write_wide_word_mask_comb_0 = write_wide_word_mask_reg_out_0 | 4'(2'(((tmp_addr_0 == write_word_addr_reg_0) &
    joined_in_fifo & (1'h1 == wr_data_fifo_out_op) & (16'h0 == wr_ID_fifo_out_data))
    ? 1'h1: 1'h0) << wr_addr_fifo_out_data[1:0]);
assign write_wide_word_mask_comb_1 = write_wide_word_mask_reg_out_1 | 4'(2'(((tmp_addr_1 == write_word_addr_reg_1) &
    joined_in_fifo & (1'h1 == wr_data_fifo_out_op) & (16'h1 == wr_ID_fifo_out_data))
    ? 1'h1: 1'h0) << wr_addr_fifo_out_data[1:0]);
assign write_wide_word_mask_reg_in_0 = (clr_write_wide_word_0 ? 4'h0: write_wide_word_mask_reg_out_0) |
    (((((clr_write_wide_word_0 & (tmp_addr_0 != write_word_addr_reg_0)) |
    ((tmp_addr_0 == write_word_addr_reg_0) & ((~write_full_word_0) |
    (write_full_word_0 & (~mem_acq[0]))))) & (1'h1 == wr_data_fifo_out_op) & (16'h0
    == wr_ID_fifo_out_data)) ? {3'h0, joined_in_fifo}: 4'h0) <<
    4'(wr_addr_fifo_out_data[1:0]));
assign write_wide_word_mask_reg_in_1 = (clr_write_wide_word_1 ? 4'h0: write_wide_word_mask_reg_out_1) |
    (((((clr_write_wide_word_1 & (tmp_addr_1 != write_word_addr_reg_1)) |
    ((tmp_addr_1 == write_word_addr_reg_1) & ((~write_full_word_1) |
    (write_full_word_1 & (~mem_acq[2]))))) & (1'h1 == wr_data_fifo_out_op) & (16'h1
    == wr_ID_fifo_out_data)) ? {3'h0, joined_in_fifo}: 4'h0) <<
    4'(wr_addr_fifo_out_data[1:0]));
always_comb begin
  num_bits_valid_mask_0_sum = 3'h0;
  num_bits_valid_mask_0_sum = num_bits_valid_mask_0_sum + 3'(write_wide_word_mask_comb_0[0]);
  num_bits_valid_mask_0_sum = num_bits_valid_mask_0_sum + 3'(write_wide_word_mask_comb_0[1]);
  num_bits_valid_mask_0_sum = num_bits_valid_mask_0_sum + 3'(write_wide_word_mask_comb_0[2]);
  num_bits_valid_mask_0_sum = num_bits_valid_mask_0_sum + 3'(write_wide_word_mask_comb_0[3]);
end
always_comb begin
  num_bits_valid_mask_1_sum = 3'h0;
  num_bits_valid_mask_1_sum = num_bits_valid_mask_1_sum + 3'(write_wide_word_mask_comb_1[0]);
  num_bits_valid_mask_1_sum = num_bits_valid_mask_1_sum + 3'(write_wide_word_mask_comb_1[1]);
  num_bits_valid_mask_1_sum = num_bits_valid_mask_1_sum + 3'(write_wide_word_mask_comb_1[2]);
  num_bits_valid_mask_1_sum = num_bits_valid_mask_1_sum + 3'(write_wide_word_mask_comb_1[3]);
end
assign write_full_word_0 = 3'h4 == num_bits_valid_mask_0_sum;
assign write_full_word_1 = 3'h4 == num_bits_valid_mask_1_sum;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_wide_word_0 <= 64'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_wide_word_0 <= 64'h0;
    end
    else if (1'h0) begin
      read_wide_word_0 <= 64'h0;
    end
    else if (set_cached_read_0) begin
      read_wide_word_0 <= data_from_mem;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_wide_word_1 <= 64'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_wide_word_1 <= 64'h0;
    end
    else if (1'h0) begin
      read_wide_word_1 <= 64'h0;
    end
    else if (set_cached_read_1) begin
      read_wide_word_1 <= data_from_mem;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_wide_word_valid_sticky_0_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_wide_word_valid_sticky_0_was_high <= 1'h0;
    end
    else if (clr_cached_read_0) begin
      read_wide_word_valid_sticky_0_was_high <= 1'h0;
    end
    else if (set_cached_read_0) begin
      read_wide_word_valid_sticky_0_was_high <= 1'h1;
    end
  end
end
assign read_wide_word_valid_sticky_0_sticky = set_cached_read_0 | read_wide_word_valid_sticky_0_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_wide_word_valid_sticky_1_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_wide_word_valid_sticky_1_was_high <= 1'h0;
    end
    else if (clr_cached_read_1) begin
      read_wide_word_valid_sticky_1_was_high <= 1'h0;
    end
    else if (set_cached_read_1) begin
      read_wide_word_valid_sticky_1_was_high <= 1'h1;
    end
  end
end
assign read_wide_word_valid_sticky_1_sticky = set_cached_read_1 | read_wide_word_valid_sticky_1_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    last_read_addr_0 <= 2'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      last_read_addr_0 <= 2'h0;
    end
    else if (1'h0) begin
      last_read_addr_0 <= 2'h0;
    end
    else if (rd_acq_0) begin
      last_read_addr_0 <= rd_addr_fifo_out_addr_0[1:0];
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    last_read_addr_1 <= 2'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      last_read_addr_1 <= 2'h0;
    end
    else if (1'h0) begin
      last_read_addr_1 <= 2'h0;
    end
    else if (rd_acq_1) begin
      last_read_addr_1 <= rd_addr_fifo_out_addr_1[1:0];
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    last_read_addr_wide_0 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      last_read_addr_wide_0 <= 16'h0;
    end
    else if (1'h0) begin
      last_read_addr_wide_0 <= 16'h0;
    end
    else if (rd_acq_0) begin
      last_read_addr_wide_0 <= addr_to_mem_local;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    last_read_addr_wide_1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      last_read_addr_wide_1 <= 16'h0;
    end
    else if (1'h0) begin
      last_read_addr_wide_1 <= 16'h0;
    end
    else if (rd_acq_1) begin
      last_read_addr_wide_1 <= addr_to_mem_local;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_addr_delayed_0 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_addr_delayed_0 <= 16'h0;
    end
    else if (1'h0) begin
      read_addr_delayed_0 <= 16'h0;
    end
    else if (rd_addr_fifo_pop_0 & rd_addr_fifo_valid_0) begin
      read_addr_delayed_0 <= rd_addr_fifo_out_addr_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_addr_delayed_1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_addr_delayed_1 <= 16'h0;
    end
    else if (1'h0) begin
      read_addr_delayed_1 <= 16'h0;
    end
    else if (rd_addr_fifo_pop_1 & rd_addr_fifo_valid_1) begin
      read_addr_delayed_1 <= rd_addr_fifo_out_addr_1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    cached_read_word_addr_0 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      cached_read_word_addr_0 <= 16'h0;
    end
    else if (1'h0) begin
      cached_read_word_addr_0 <= 16'h0;
    end
    else if (set_read_word_addr_0) begin
      cached_read_word_addr_0 <= addr_to_mem_local;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    cached_read_word_addr_1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      cached_read_word_addr_1 <= 16'h0;
    end
    else if (1'h0) begin
      cached_read_word_addr_1 <= 16'h0;
    end
    else if (set_read_word_addr_1) begin
      cached_read_word_addr_1 <= addr_to_mem_local;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    ren_full_delayed_0 <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      ren_full_delayed_0 <= 1'h0;
    end
    else if (1'h0) begin
      ren_full_delayed_0 <= 1'h0;
    end
    else if (1'h1) begin
      ren_full_delayed_0 <= ren_full[0] & rd_acq_0;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    ren_full_delayed_1 <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      ren_full_delayed_1 <= 1'h0;
    end
    else if (1'h0) begin
      ren_full_delayed_1 <= 1'h0;
    end
    else if (1'h1) begin
      ren_full_delayed_1 <= ren_full[1] & rd_acq_1;
    end
  end
end
assign use_cached_read_0 = read_wide_word_valid_sticky_0_sticky & ((((16'(rd_addr_fifo_out_addr_0[15:2]) +
    blk_base[0]) & buffet_capacity_mask[0]) + buffet_base[0]) ==
    cached_read_word_addr_0) & (16'h1 == rd_op_fifo_out_op_0) & read_joined_0;
assign use_cached_read_1 = read_wide_word_valid_sticky_1_sticky & ((((16'(rd_addr_fifo_out_addr_1[15:2]) +
    blk_base[1]) & buffet_capacity_mask[1]) + buffet_base[1]) ==
    cached_read_word_addr_1) & (16'h1 == rd_op_fifo_out_op_1) & read_joined_1;
assign from_cached_read_0 = read_wide_word_valid_sticky_0_sticky & ((((16'(read_addr_delayed_0[15:2]) +
    blk_base[0]) & buffet_capacity_mask[0]) + buffet_base[0]) ==
    cached_read_word_addr_0) & read_joined_d1_0;
assign from_cached_read_1 = read_wide_word_valid_sticky_1_sticky & ((((16'(read_addr_delayed_1[15:2]) +
    blk_base[1]) & buffet_capacity_mask[1]) + buffet_base[1]) ==
    cached_read_word_addr_1) & read_joined_d1_1;
assign chosen_read_0 = (~ren_full_delayed_0) ? read_wide_word_0[read_addr_delayed_0[1:0]]:
    data_from_mem[last_read_addr_0[1:0]];
assign chosen_read_1 = (~ren_full_delayed_1) ? read_wide_word_1[read_addr_delayed_1[1:0]]:
    data_from_mem[last_read_addr_1[1:0]];
assign any_sram_lock = |{sram_lock_0, sram_lock_1};

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_d1 <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_d1 <= 1'h0;
    end
    else if (1'h0) begin
      read_d1 <= 1'h0;
    end
    else if (1'h1) begin
      read_d1 <= |{mem_acq[1], mem_acq[3]};
    end
  end
end
assign valid_from_mem = read_d1;
assign rd_rsp_data_0_valid = ~rd_rsp_fifo_0_empty;
assign rd_rsp_data_1_valid = ~rd_rsp_fifo_1_empty;
always_comb begin
  decode_sel_done_size_request_full_blk_bounds = 1'h0;
  decode_ret_size_request_full_blk_bounds = 16'h0;
  if ((~decode_sel_done_size_request_full_blk_bounds) & size_request_full[0]) begin
    decode_ret_size_request_full_blk_bounds = blk_bounds[0];
    decode_sel_done_size_request_full_blk_bounds = 1'h1;
  end
  if ((~decode_sel_done_size_request_full_blk_bounds) & size_request_full[1]) begin
    decode_ret_size_request_full_blk_bounds = blk_bounds[1];
    decode_sel_done_size_request_full_blk_bounds = 1'h1;
  end
end
assign rd_rsp_fifo_0_in_data[15:0] = (from_cached_read_0 & read_wide_word_valid_sticky_0_sticky) ? chosen_read_0:
    decode_ret_size_request_full_blk_bounds + 16'h1;
assign rd_rsp_fifo_1_in_data[15:0] = (from_cached_read_1 & read_wide_word_valid_sticky_1_sticky) ? chosen_read_1:
    decode_ret_size_request_full_blk_bounds + 16'h1;
assign rd_rsp_fifo_0_in_data[16] = from_cached_read_0 ? 1'h0: 1'h1;
assign rd_rsp_fifo_1_in_data[16] = from_cached_read_1 ? 1'h0: 1'h1;
assign rd_rsp_fifo_0_push = from_cached_read_0 | size_request_full[0];
assign rd_rsp_fifo_1_push = from_cached_read_1 | size_request_full[1];
assign joined_in_fifo = wr_data_fifo_valid & wr_addr_fifo_valid & wr_ID_fifo_valid;
assign {wr_addr_fifo_pop, wr_data_fifo_pop, wr_ID_fifo_pop} = {pop_in_fifos, pop_in_fifos, pop_in_fifos};
assign pop_in_fifos = |pop_in_full;
assign {rd_op_fifo_pop_0, rd_addr_fifo_pop_0} = {read_pop_0, read_pop_0};
assign {rd_op_fifo_pop_1, rd_addr_fifo_pop_1} = {read_pop_1, read_pop_1};
assign read_pop_0 = read_pop_full[0];
assign read_pop_1 = read_pop_full[1];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    blk_count_0 <= 8'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      blk_count_0 <= 8'h0;
    end
    else if (push_blk[0]) begin
      blk_count_0 <= blk_count_0 + 8'h1;
    end
    else if (pop_blk[0]) begin
      blk_count_0 <= blk_count_0 - 8'h1;
    end
    else blk_count_0 <= blk_count_0;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    blk_count_1 <= 8'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      blk_count_1 <= 8'h0;
    end
    else if (push_blk[1]) begin
      blk_count_1 <= blk_count_1 + 8'h1;
    end
    else if (pop_blk[1]) begin
      blk_count_1 <= blk_count_1 - 8'h1;
    end
    else blk_count_1 <= blk_count_1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    curr_capacity_pre[0] <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      curr_capacity_pre[0] <= 16'h0;
    end
    else if (push_blk[0] || pop_blk[0]) begin
      curr_capacity_pre[0] <= (curr_capacity_pre[0] + (push_blk[0] ? blk_bounds[0]: 16'h0)) - (pop_blk[0] ?
          blk_bounds[0]: 16'h0);
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    curr_capacity_pre[1] <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      curr_capacity_pre[1] <= 16'h0;
    end
    else if (push_blk[1] || pop_blk[1]) begin
      curr_capacity_pre[1] <= (curr_capacity_pre[1] + (push_blk[1] ? blk_bounds[1]: 16'h0)) - (pop_blk[1] ?
          blk_bounds[1]: 16'h0);
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    PREVIOUS_WR_OP <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      PREVIOUS_WR_OP <= 1'h0;
    end
    else if (1'h0) begin
      PREVIOUS_WR_OP <= 1'h0;
    end
    else if (1'h1) begin
      PREVIOUS_WR_OP <= wr_data_fifo_out_op;
    end
  end
end
assign blk_fifo_0_data_in = {curr_base_0, curr_bounds_0};
assign {blk_base[0], blk_bounds[0]} = blk_fifo_0_data_out;
assign blk_full[0] = blk_fifo_0_full;
assign blk_valid[0] = ~blk_fifo_0_empty;
assign blk_fifo_1_data_in = {curr_base_1, curr_bounds_1};
assign {blk_base[1], blk_bounds[1]} = blk_fifo_1_data_out;
assign blk_full[1] = blk_fifo_1_full;
assign blk_valid[1] = ~blk_fifo_1_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    read_fsm_0_current_state <= RD_START_0;
  end
  else if (clk_en) begin
    if (flush) begin
      read_fsm_0_current_state <= RD_START_0;
    end
    else read_fsm_0_current_state <= read_fsm_0_next_state;
  end
end
always_comb begin
  read_fsm_0_next_state = read_fsm_0_current_state;
  unique case (read_fsm_0_current_state)
    RD_PAUSE_0: begin
        if (push_blk[0]) begin
          read_fsm_0_next_state = RD_PAUSE_T_0;
        end
        else read_fsm_0_next_state = RD_PAUSE_0;
      end
    RD_PAUSE_T_0: read_fsm_0_next_state = RD_START_0;
    RD_START_0: begin
        if ((blk_count_0 == 8'h0) & (rd_op_fifo_out_op_0 == 16'h0) & read_joined_0 & 1'h1) begin
          read_fsm_0_next_state = RD_PAUSE_0;
        end
        else read_fsm_0_next_state = RD_START_0;
      end
    default: read_fsm_0_next_state = read_fsm_0_current_state;
  endcase
end
always_comb begin
  unique case (read_fsm_0_current_state)
    RD_PAUSE_0: begin :read_fsm_0_RD_PAUSE_0_Output
        pop_blk[0] = 1'h0;
        ren_full[0] = 1'h0;
        read_pop_full[0] = 1'h0;
        size_request_full[0] = 1'h0;
        set_cached_read_0 = 1'h0;
        clr_cached_read_0 = 1'h0;
        set_read_word_addr_0 = 1'h0;
      end :read_fsm_0_RD_PAUSE_0_Output
    RD_PAUSE_T_0: begin :read_fsm_0_RD_PAUSE_T_0_Output
        pop_blk[0] = 1'h1;
        ren_full[0] = 1'h0;
        read_pop_full[0] = 1'h0;
        size_request_full[0] = 1'h0;
        set_cached_read_0 = 1'h0;
        clr_cached_read_0 = 1'h0;
        set_read_word_addr_0 = 1'h0;
      end :read_fsm_0_RD_PAUSE_T_0_Output
    RD_START_0: begin :read_fsm_0_RD_START_0_Output
        pop_blk[0] = (rd_op_fifo_out_op_0 == 16'h0) & read_joined_0 & 1'h1 & (blk_count_0 > 8'h0);
        ren_full[0] = (rd_op_fifo_out_op_0 == 16'h1) & (~use_cached_read_0) & read_joined_0 &
            (~rd_rsp_fifo_0_almost_full) & blk_valid[0] &
            (((16'(rd_addr_fifo_out_addr_0[15:2]) + blk_base[0] + buffet_base[0]) !=
            cached_read_word_addr_0) | (~read_wide_word_valid_sticky_0_sticky)) & 1'h1;
        read_pop_full[0] = ((rd_op_fifo_out_op_0 == 16'h2) ? (~ren_full_delayed_0) & blk_valid[0]:
            (rd_op_fifo_out_op_0 == 16'h1) ? (mem_acq[1] | use_cached_read_0) &
            (~rd_rsp_fifo_0_full): 1'h1) & read_joined_0 & 1'h1;
        size_request_full[0] = blk_valid[0] & (rd_op_fifo_out_op_0 == 16'h2) & read_joined_0 & 1'h1;
        set_cached_read_0 = ren_full_delayed_0 & 1'h1;
        clr_cached_read_0 = (rd_op_fifo_out_op_0 == 16'h0) & read_joined_0 & 1'h1;
        set_read_word_addr_0 = ren_full[0] & mem_acq[1] & (addr_to_mem_local != cached_read_word_addr_0) & 1'h1;
      end :read_fsm_0_RD_START_0_Output
    default: begin :read_fsm_0_default_Output
        pop_blk[0] = (rd_op_fifo_out_op_0 == 16'h0) & read_joined_0 & 1'h1 & (blk_count_0 > 8'h0);
        ren_full[0] = (rd_op_fifo_out_op_0 == 16'h1) & (~use_cached_read_0) & read_joined_0 &
            (~rd_rsp_fifo_0_almost_full) & blk_valid[0] &
            (((16'(rd_addr_fifo_out_addr_0[15:2]) + blk_base[0] + buffet_base[0]) !=
            cached_read_word_addr_0) | (~read_wide_word_valid_sticky_0_sticky)) & 1'h1;
        read_pop_full[0] = ((rd_op_fifo_out_op_0 == 16'h2) ? (~ren_full_delayed_0) & blk_valid[0]:
            (rd_op_fifo_out_op_0 == 16'h1) ? (mem_acq[1] | use_cached_read_0) &
            (~rd_rsp_fifo_0_full): 1'h1) & read_joined_0 & 1'h1;
        size_request_full[0] = blk_valid[0] & (rd_op_fifo_out_op_0 == 16'h2) & read_joined_0 & 1'h1;
        set_cached_read_0 = ren_full_delayed_0 & 1'h1;
        clr_cached_read_0 = (rd_op_fifo_out_op_0 == 16'h0) & read_joined_0 & 1'h1;
        set_read_word_addr_0 = ren_full[0] & mem_acq[1] & (addr_to_mem_local != cached_read_word_addr_0) & 1'h1;
      end :read_fsm_0_default_Output
  endcase
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    read_fsm_1_current_state <= RD_START_1;
  end
  else if (clk_en) begin
    if (flush) begin
      read_fsm_1_current_state <= RD_START_1;
    end
    else read_fsm_1_current_state <= read_fsm_1_next_state;
  end
end
always_comb begin
  read_fsm_1_next_state = read_fsm_1_current_state;
  unique case (read_fsm_1_current_state)
    RD_PAUSE_1: begin
        if (push_blk[1]) begin
          read_fsm_1_next_state = RD_PAUSE_T_1;
        end
        else read_fsm_1_next_state = RD_PAUSE_1;
      end
    RD_PAUSE_T_1: read_fsm_1_next_state = RD_START_1;
    RD_START_1: begin
        if ((blk_count_1 == 8'h0) & (rd_op_fifo_out_op_1 == 16'h0) & read_joined_1 & 1'h1) begin
          read_fsm_1_next_state = RD_PAUSE_1;
        end
        else read_fsm_1_next_state = RD_START_1;
      end
    default: read_fsm_1_next_state = read_fsm_1_current_state;
  endcase
end
always_comb begin
  unique case (read_fsm_1_current_state)
    RD_PAUSE_1: begin :read_fsm_1_RD_PAUSE_1_Output
        pop_blk[1] = 1'h0;
        ren_full[1] = 1'h0;
        read_pop_full[1] = 1'h0;
        size_request_full[1] = 1'h0;
        set_cached_read_1 = 1'h0;
        clr_cached_read_1 = 1'h0;
        set_read_word_addr_1 = 1'h0;
      end :read_fsm_1_RD_PAUSE_1_Output
    RD_PAUSE_T_1: begin :read_fsm_1_RD_PAUSE_T_1_Output
        pop_blk[1] = 1'h1;
        ren_full[1] = 1'h0;
        read_pop_full[1] = 1'h0;
        size_request_full[1] = 1'h0;
        set_cached_read_1 = 1'h0;
        clr_cached_read_1 = 1'h0;
        set_read_word_addr_1 = 1'h0;
      end :read_fsm_1_RD_PAUSE_T_1_Output
    RD_START_1: begin :read_fsm_1_RD_START_1_Output
        pop_blk[1] = (rd_op_fifo_out_op_1 == 16'h0) & read_joined_1 & 1'h1 & (blk_count_1 > 8'h0);
        ren_full[1] = (rd_op_fifo_out_op_1 == 16'h1) & (~use_cached_read_1) & read_joined_1 &
            (~rd_rsp_fifo_1_almost_full) & blk_valid[1] &
            (((16'(rd_addr_fifo_out_addr_1[15:2]) + blk_base[1] + buffet_base[1]) !=
            cached_read_word_addr_1) | (~read_wide_word_valid_sticky_1_sticky)) & 1'h1;
        read_pop_full[1] = ((rd_op_fifo_out_op_1 == 16'h2) ? (~ren_full_delayed_1) & blk_valid[1]:
            (rd_op_fifo_out_op_1 == 16'h1) ? (mem_acq[3] | use_cached_read_1) &
            (~rd_rsp_fifo_1_full): 1'h1) & read_joined_1 & 1'h1;
        size_request_full[1] = blk_valid[1] & (rd_op_fifo_out_op_1 == 16'h2) & read_joined_1 & 1'h1;
        set_cached_read_1 = ren_full_delayed_1 & 1'h1;
        clr_cached_read_1 = (rd_op_fifo_out_op_1 == 16'h0) & read_joined_1 & 1'h1;
        set_read_word_addr_1 = ren_full[1] & mem_acq[3] & (addr_to_mem_local != cached_read_word_addr_1) & 1'h1;
      end :read_fsm_1_RD_START_1_Output
    default: begin :read_fsm_1_default_Output
        pop_blk[1] = (rd_op_fifo_out_op_1 == 16'h0) & read_joined_1 & 1'h1 & (blk_count_1 > 8'h0);
        ren_full[1] = (rd_op_fifo_out_op_1 == 16'h1) & (~use_cached_read_1) & read_joined_1 &
            (~rd_rsp_fifo_1_almost_full) & blk_valid[1] &
            (((16'(rd_addr_fifo_out_addr_1[15:2]) + blk_base[1] + buffet_base[1]) !=
            cached_read_word_addr_1) | (~read_wide_word_valid_sticky_1_sticky)) & 1'h1;
        read_pop_full[1] = ((rd_op_fifo_out_op_1 == 16'h2) ? (~ren_full_delayed_1) & blk_valid[1]:
            (rd_op_fifo_out_op_1 == 16'h1) ? (mem_acq[3] | use_cached_read_1) &
            (~rd_rsp_fifo_1_full): 1'h1) & read_joined_1 & 1'h1;
        size_request_full[1] = blk_valid[1] & (rd_op_fifo_out_op_1 == 16'h2) & read_joined_1 & 1'h1;
        set_cached_read_1 = ren_full_delayed_1 & 1'h1;
        clr_cached_read_1 = (rd_op_fifo_out_op_1 == 16'h0) & read_joined_1 & 1'h1;
        set_read_word_addr_1 = ren_full[1] & mem_acq[3] & (addr_to_mem_local != cached_read_word_addr_1) & 1'h1;
      end :read_fsm_1_default_Output
  endcase
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    write_fsm_0_current_state <= WR_START_0;
  end
  else if (clk_en) begin
    if (flush) begin
      write_fsm_0_current_state <= WR_START_0;
    end
    else write_fsm_0_current_state <= write_fsm_0_next_state;
  end
end
always_comb begin
  write_fsm_0_next_state = write_fsm_0_current_state;
  unique case (write_fsm_0_current_state)
    MODIFY_0: begin
        if (1'h1 == PREVIOUS_WR_OP) begin
          write_fsm_0_next_state = WRITING_0;
        end
        else if ((1'h0 == PREVIOUS_WR_OP) & (~blk_full[0])) begin
          write_fsm_0_next_state = WR_START_0;
        end
        else write_fsm_0_next_state = MODIFY_0;
      end
    WRITING_0: begin
        if (joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (~blk_full[0]) & ((write_full_word_0 & mem_acq[0]) | (num_bits_valid_mask_0_sum == 3'h0)) & (16'h0 == wr_ID_fifo_out_data)) begin
          write_fsm_0_next_state = WR_START_0;
        end
        else if (joined_in_fifo & (16'h0 == wr_ID_fifo_out_data) & mem_acq[0] & ((1'h0 == wr_data_fifo_out_op) | ((tmp_addr_0 != write_word_addr_reg_0) & write_word_addr_valid_sticky_0_sticky & (1'h1 == wr_data_fifo_out_op))) & (num_bits_valid_mask_0_sum > 3'h0) & (~write_full_word_0)) begin
          write_fsm_0_next_state = MODIFY_0;
        end
        else write_fsm_0_next_state = WRITING_0;
      end
    WR_START_0: begin
        if (joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (16'h0 == wr_ID_fifo_out_data) & tile_en) begin
          write_fsm_0_next_state = WRITING_0;
        end
        else write_fsm_0_next_state = WR_START_0;
      end
    default: write_fsm_0_next_state = write_fsm_0_current_state;
  endcase
end
always_comb begin
  unique case (write_fsm_0_current_state)
    MODIFY_0: begin :write_fsm_0_MODIFY_0_Output
        push_blk[0] = (1'h0 == PREVIOUS_WR_OP) & (~blk_full[0]);
        en_curr_base[0] = (1'h0 == PREVIOUS_WR_OP) & (~blk_full[0]);
        en_curr_bounds[0] = 1'h0;
        wen_full[0] = ~blk_full[0];
        pop_in_full[0] = ~blk_full[0];
        set_write_wide_word_0 = 1'h0;
        clr_write_wide_word_0 = ~blk_full[0];
        write_to_sram_0 = ~blk_full[0];
        read_from_sram_write_side_0 = 1'h0;
        set_wide_word_addr_0 = 1'h0;
        sram_lock_0 = ~blk_full[0];
      end :write_fsm_0_MODIFY_0_Output
    WRITING_0: begin :write_fsm_0_WRITING_0_Output
        push_blk[0] = joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (~blk_full[0]) &
            ((write_full_word_0 & mem_acq[0]) | (num_bits_valid_mask_0_sum == 3'h0)) &
            (16'h0 == wr_ID_fifo_out_data);
        en_curr_base[0] = joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (~blk_full[0]) &
            ((write_full_word_0 & mem_acq[0]) | (num_bits_valid_mask_0_sum == 3'h0)) &
            (16'h0 == wr_ID_fifo_out_data);
        set_write_wide_word_0 = (tmp_addr_0 == write_word_addr_reg_0) & write_word_addr_valid_sticky_0_sticky &
            joined_in_fifo & (wr_data_fifo_out_op == 1'h1) & (16'h0 == wr_ID_fifo_out_data);
        en_curr_bounds[0] = (mem_acq[0] | set_write_wide_word_0) & joined_in_fifo & (wr_data_fifo_out_op ==
            1'h1) & (16'h0 == wr_ID_fifo_out_data);
        wen_full[0] = joined_in_fifo & (wr_data_fifo_out_op == 1'h1) & ((buffet_capacity[0] -
            curr_capacity_pre[0]) > wr_addr_fifo_out_data) & (16'h0 == wr_ID_fifo_out_data);
        clr_write_wide_word_0 = ((tmp_addr_0 != write_word_addr_reg_0) |
            (~write_word_addr_valid_sticky_0_sticky) | ((tmp_addr_0 ==
            write_word_addr_reg_0) & write_word_addr_valid_sticky_0_sticky &
            write_full_word_0)) & joined_in_fifo & (wr_data_fifo_out_op == 1'h1) &
            mem_acq[0] & (16'h0 == wr_ID_fifo_out_data);
        write_to_sram_0 = write_full_word_0 & joined_in_fifo & ((buffet_capacity[0] -
            curr_capacity_pre[0]) > wr_addr_fifo_out_data) & (16'h0 == wr_ID_fifo_out_data);
        set_wide_word_addr_0 = ((tmp_addr_0 != write_word_addr_reg_0) |
            (~write_word_addr_valid_sticky_0_sticky)) & joined_in_fifo &
            (wr_data_fifo_out_op == 1'h1) & ((|write_wide_word_mask_reg_out_0) ? mem_acq[0]:
            1'h1) & (16'h0 == wr_ID_fifo_out_data);
        sram_lock_0 = 1'h0;
        read_from_sram_write_side_0 = joined_in_fifo & (16'h0 == wr_ID_fifo_out_data) & (~any_sram_lock) &
            ((buffet_capacity[0] - curr_capacity_pre[0]) > wr_addr_fifo_out_data) & ((1'h0
            == wr_data_fifo_out_op) | ((tmp_addr_0 != write_word_addr_reg_0) &
            write_word_addr_valid_sticky_0_sticky & (1'h1 == wr_data_fifo_out_op))) &
            (num_bits_valid_mask_0_sum > 3'h0) & (~write_full_word_0);
        pop_in_full[0] = ((mem_acq[0] | set_write_wide_word_0) & joined_in_fifo & (wr_data_fifo_out_op ==
            1'h1) & ((buffet_capacity[0] - curr_capacity_pre[0]) > wr_addr_fifo_out_data) &
            (16'h0 == wr_ID_fifo_out_data)) | (joined_in_fifo & (wr_data_fifo_out_op ==
            1'h0) & (~blk_full[0]) & ((write_full_word_0 & mem_acq[0]) |
            (num_bits_valid_mask_0_sum == 3'h0)) & (16'h0 == wr_ID_fifo_out_data));
      end :write_fsm_0_WRITING_0_Output
    WR_START_0: begin :write_fsm_0_WR_START_0_Output
        push_blk[0] = 1'h0;
        en_curr_base[0] = 1'h0;
        en_curr_bounds[0] = 1'h0;
        wen_full[0] = 1'h0;
        pop_in_full[0] = (wr_data_fifo_out_op == 1'h0) & (16'h0 == wr_ID_fifo_out_data);
        set_write_wide_word_0 = 1'h0;
        clr_write_wide_word_0 = 1'h0;
        write_to_sram_0 = 1'h0;
        set_wide_word_addr_0 = 1'h0;
        sram_lock_0 = 1'h0;
        read_from_sram_write_side_0 = 1'h0;
      end :write_fsm_0_WR_START_0_Output
    default: begin :write_fsm_0_default_Output
        push_blk[0] = 1'h0;
        en_curr_base[0] = 1'h0;
        en_curr_bounds[0] = 1'h0;
        wen_full[0] = 1'h0;
        pop_in_full[0] = (wr_data_fifo_out_op == 1'h0) & (16'h0 == wr_ID_fifo_out_data);
        set_write_wide_word_0 = 1'h0;
        clr_write_wide_word_0 = 1'h0;
        write_to_sram_0 = 1'h0;
        set_wide_word_addr_0 = 1'h0;
        sram_lock_0 = 1'h0;
        read_from_sram_write_side_0 = 1'h0;
      end :write_fsm_0_default_Output
  endcase
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    write_fsm_1_current_state <= WR_START_1;
  end
  else if (clk_en) begin
    if (flush) begin
      write_fsm_1_current_state <= WR_START_1;
    end
    else write_fsm_1_current_state <= write_fsm_1_next_state;
  end
end
always_comb begin
  write_fsm_1_next_state = write_fsm_1_current_state;
  unique case (write_fsm_1_current_state)
    MODIFY_1: begin
        if (1'h1 == PREVIOUS_WR_OP) begin
          write_fsm_1_next_state = WRITING_1;
        end
        else if ((1'h0 == PREVIOUS_WR_OP) & (~blk_full[1])) begin
          write_fsm_1_next_state = WR_START_1;
        end
        else write_fsm_1_next_state = MODIFY_1;
      end
    WRITING_1: begin
        if (joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (~blk_full[1]) & ((write_full_word_1 & mem_acq[2]) | (num_bits_valid_mask_1_sum == 3'h0)) & (16'h1 == wr_ID_fifo_out_data)) begin
          write_fsm_1_next_state = WR_START_1;
        end
        else if (joined_in_fifo & (16'h1 == wr_ID_fifo_out_data) & mem_acq[2] & ((1'h0 == wr_data_fifo_out_op) | ((tmp_addr_1 != write_word_addr_reg_1) & write_word_addr_valid_sticky_1_sticky & (1'h1 == wr_data_fifo_out_op))) & (num_bits_valid_mask_1_sum > 3'h0) & (~write_full_word_1)) begin
          write_fsm_1_next_state = MODIFY_1;
        end
        else write_fsm_1_next_state = WRITING_1;
      end
    WR_START_1: begin
        if (joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (16'h1 == wr_ID_fifo_out_data) & tile_en) begin
          write_fsm_1_next_state = WRITING_1;
        end
        else write_fsm_1_next_state = WR_START_1;
      end
    default: write_fsm_1_next_state = write_fsm_1_current_state;
  endcase
end
always_comb begin
  unique case (write_fsm_1_current_state)
    MODIFY_1: begin :write_fsm_1_MODIFY_1_Output
        push_blk[1] = (1'h0 == PREVIOUS_WR_OP) & (~blk_full[1]);
        en_curr_base[1] = (1'h0 == PREVIOUS_WR_OP) & (~blk_full[1]);
        en_curr_bounds[1] = 1'h0;
        wen_full[1] = ~blk_full[1];
        pop_in_full[1] = ~blk_full[1];
        set_write_wide_word_1 = 1'h0;
        clr_write_wide_word_1 = ~blk_full[1];
        write_to_sram_1 = ~blk_full[1];
        read_from_sram_write_side_1 = 1'h0;
        set_wide_word_addr_1 = 1'h0;
        sram_lock_1 = ~blk_full[1];
      end :write_fsm_1_MODIFY_1_Output
    WRITING_1: begin :write_fsm_1_WRITING_1_Output
        push_blk[1] = joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (~blk_full[1]) &
            ((write_full_word_1 & mem_acq[2]) | (num_bits_valid_mask_1_sum == 3'h0)) &
            (16'h1 == wr_ID_fifo_out_data);
        en_curr_base[1] = joined_in_fifo & (wr_data_fifo_out_op == 1'h0) & (~blk_full[1]) &
            ((write_full_word_1 & mem_acq[2]) | (num_bits_valid_mask_1_sum == 3'h0)) &
            (16'h1 == wr_ID_fifo_out_data);
        set_write_wide_word_1 = (tmp_addr_1 == write_word_addr_reg_1) & write_word_addr_valid_sticky_1_sticky &
            joined_in_fifo & (wr_data_fifo_out_op == 1'h1) & (16'h1 == wr_ID_fifo_out_data);
        en_curr_bounds[1] = (mem_acq[2] | set_write_wide_word_1) & joined_in_fifo & (wr_data_fifo_out_op ==
            1'h1) & (16'h1 == wr_ID_fifo_out_data);
        wen_full[1] = joined_in_fifo & (wr_data_fifo_out_op == 1'h1) & ((buffet_capacity[1] -
            curr_capacity_pre[1]) > wr_addr_fifo_out_data) & (16'h1 == wr_ID_fifo_out_data);
        clr_write_wide_word_1 = ((tmp_addr_1 != write_word_addr_reg_1) |
            (~write_word_addr_valid_sticky_1_sticky) | ((tmp_addr_1 ==
            write_word_addr_reg_1) & write_word_addr_valid_sticky_1_sticky &
            write_full_word_1)) & joined_in_fifo & (wr_data_fifo_out_op == 1'h1) &
            mem_acq[2] & (16'h1 == wr_ID_fifo_out_data);
        write_to_sram_1 = write_full_word_1 & joined_in_fifo & ((buffet_capacity[1] -
            curr_capacity_pre[1]) > wr_addr_fifo_out_data) & (16'h1 == wr_ID_fifo_out_data);
        set_wide_word_addr_1 = ((tmp_addr_1 != write_word_addr_reg_1) |
            (~write_word_addr_valid_sticky_1_sticky)) & joined_in_fifo &
            (wr_data_fifo_out_op == 1'h1) & ((|write_wide_word_mask_reg_out_1) ? mem_acq[2]:
            1'h1) & (16'h1 == wr_ID_fifo_out_data);
        sram_lock_1 = 1'h0;
        read_from_sram_write_side_1 = joined_in_fifo & (16'h1 == wr_ID_fifo_out_data) & (~any_sram_lock) &
            ((buffet_capacity[1] - curr_capacity_pre[1]) > wr_addr_fifo_out_data) & ((1'h0
            == wr_data_fifo_out_op) | ((tmp_addr_1 != write_word_addr_reg_1) &
            write_word_addr_valid_sticky_1_sticky & (1'h1 == wr_data_fifo_out_op))) &
            (num_bits_valid_mask_1_sum > 3'h0) & (~write_full_word_1);
        pop_in_full[1] = ((mem_acq[2] | set_write_wide_word_1) & joined_in_fifo & (wr_data_fifo_out_op ==
            1'h1) & ((buffet_capacity[1] - curr_capacity_pre[1]) > wr_addr_fifo_out_data) &
            (16'h1 == wr_ID_fifo_out_data)) | (joined_in_fifo & (wr_data_fifo_out_op ==
            1'h0) & (~blk_full[1]) & ((write_full_word_1 & mem_acq[2]) |
            (num_bits_valid_mask_1_sum == 3'h0)) & (16'h1 == wr_ID_fifo_out_data));
      end :write_fsm_1_WRITING_1_Output
    WR_START_1: begin :write_fsm_1_WR_START_1_Output
        push_blk[1] = 1'h0;
        en_curr_base[1] = 1'h0;
        en_curr_bounds[1] = 1'h0;
        wen_full[1] = 1'h0;
        pop_in_full[1] = (wr_data_fifo_out_op == 1'h0) & (16'h1 == wr_ID_fifo_out_data);
        set_write_wide_word_1 = 1'h0;
        clr_write_wide_word_1 = 1'h0;
        write_to_sram_1 = 1'h0;
        set_wide_word_addr_1 = 1'h0;
        sram_lock_1 = 1'h0;
        read_from_sram_write_side_1 = 1'h0;
      end :write_fsm_1_WR_START_1_Output
    default: begin :write_fsm_1_default_Output
        push_blk[1] = 1'h0;
        en_curr_base[1] = 1'h0;
        en_curr_bounds[1] = 1'h0;
        wen_full[1] = 1'h0;
        pop_in_full[1] = (wr_data_fifo_out_op == 1'h0) & (16'h1 == wr_ID_fifo_out_data);
        set_write_wide_word_1 = 1'h0;
        clr_write_wide_word_1 = 1'h0;
        write_to_sram_1 = 1'h0;
        set_wide_word_addr_1 = 1'h0;
        sram_lock_1 = 1'h0;
        read_from_sram_write_side_1 = 1'h0;
      end :write_fsm_1_default_Output
  endcase
end
assign base_rr = {ren_full[1], write_to_sram_1 | read_from_sram_write_side_1, {ren_full[0],
    write_to_sram_0 | read_from_sram_write_side_0}};
assign rr_arbiter_resource_ready = ~any_sram_lock;
assign ren_to_mem = (|{mem_acq[1] & ren_full[0], mem_acq[3] & ren_full[1]}) |
    (|{read_from_sram_write_side_0, read_from_sram_write_side_1});
assign wen_to_mem = |({mem_acq[0] & write_to_sram_0, mem_acq[2] & write_to_sram_1} | {sram_lock_0,
    sram_lock_1});
assign tmp_wr_base = (mem_acq[2] & write_to_sram_1) ? curr_base_1 + buffet_base[1]: (mem_acq[0] &
    write_to_sram_0) ? curr_base_0 + buffet_base[0]: 16'h0;
assign tmp_rd_base = (mem_acq[3] & ren_full[1]) ? blk_base[1] + buffet_base[1]: (mem_acq[1] &
    ren_full[0]) ? blk_base[0] + buffet_base[0]: 16'h0;
assign data_to_mem = (mem_acq[0] & write_to_sram_0) ? write_wide_word_comb_out_0: (mem_acq[2] &
    write_to_sram_1) ? write_wide_word_comb_out_1: sram_lock_0 ?
    write_wide_word_modified_0: sram_lock_1 ? write_wide_word_modified_1: 64'h0;
assign addr_to_mem_local = (wen_to_mem | mem_acq[0] | mem_acq[2]) ? (mem_acq[0] | sram_lock_0) ?
    write_word_addr_reg_0: write_word_addr_reg_1: (mem_acq[1] & ren_full[0]) ?
    ((16'(rd_addr_fifo_out_addr_0[15:2]) + blk_base[0]) & buffet_capacity_mask[0]) +
    buffet_base[0]: ((16'(rd_addr_fifo_out_addr_1[15:2]) + blk_base[1]) &
    buffet_capacity_mask[1]) + buffet_base[1];
reg_fifo_depth_2_w_17_afd_2 wr_data_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(wr_data),
  .flush(flush),
  .pop(wr_data_fifo_pop),
  .push(wr_data_valid),
  .rst_n(rst_n),
  .data_out(wr_data_fifo_data_out),
  .empty(wr_data_fifo_empty),
  .full(wr_data_fifo_full)
);

reg_fifo_depth_2_w_16_afd_2 wr_addr_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(wr_addr[0][15:0]),
  .flush(flush),
  .pop(wr_addr_fifo_pop),
  .push(wr_addr_valid),
  .rst_n(rst_n),
  .data_out(wr_addr_fifo_out_data),
  .empty(wr_addr_fifo_empty),
  .full(wr_addr_fifo_full)
);

reg_fifo_depth_2_w_16_afd_2 wr_ID_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(wr_ID[0][15:0]),
  .flush(flush),
  .pop(wr_ID_fifo_pop),
  .push(wr_ID_valid),
  .rst_n(rst_n),
  .data_out(wr_ID_fifo_out_data),
  .empty(wr_ID_fifo_empty),
  .full(wr_ID_fifo_full)
);

reg_fifo_depth_2_w_16_afd_2 rd_op_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_op_0[0][15:0]),
  .flush(flush),
  .pop(rd_op_fifo_pop_0),
  .push(rd_op_0_valid),
  .rst_n(rst_n),
  .data_out(rd_op_fifo_out_op_0),
  .empty(rd_op_fifo_0_empty),
  .full(rd_op_fifo_0_full)
);

reg_fifo_depth_2_w_16_afd_2 rd_op_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_op_1[0][15:0]),
  .flush(flush),
  .pop(rd_op_fifo_pop_1),
  .push(rd_op_1_valid),
  .rst_n(rst_n),
  .data_out(rd_op_fifo_out_op_1),
  .empty(rd_op_fifo_1_empty),
  .full(rd_op_fifo_1_full)
);

reg_fifo_depth_2_w_16_afd_2 rd_addr_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_addr_0[0][15:0]),
  .flush(flush),
  .pop(rd_addr_fifo_pop_0),
  .push(rd_addr_0_valid),
  .rst_n(rst_n),
  .data_out(rd_addr_fifo_out_addr_0),
  .empty(rd_addr_fifo_0_empty),
  .full(rd_addr_fifo_0_full)
);

reg_fifo_depth_2_w_16_afd_2 rd_addr_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_addr_1[0][15:0]),
  .flush(flush),
  .pop(rd_addr_fifo_pop_1),
  .push(rd_addr_1_valid),
  .rst_n(rst_n),
  .data_out(rd_addr_fifo_out_addr_1),
  .empty(rd_addr_fifo_1_empty),
  .full(rd_addr_fifo_1_full)
);

reg_fifo_depth_2_w_17_afd_0 rd_rsp_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_rsp_fifo_0_in_data),
  .flush(flush),
  .pop(rd_rsp_data_0_ready),
  .push(rd_rsp_fifo_0_push),
  .rst_n(rst_n),
  .almost_full(rd_rsp_fifo_0_almost_full),
  .data_out(rd_rsp_data_0),
  .empty(rd_rsp_fifo_0_empty),
  .full(rd_rsp_fifo_0_full)
);

reg_fifo_depth_2_w_17_afd_0 rd_rsp_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_rsp_fifo_1_in_data),
  .flush(flush),
  .pop(rd_rsp_data_1_ready),
  .push(rd_rsp_fifo_1_push),
  .rst_n(rst_n),
  .almost_full(rd_rsp_fifo_1_almost_full),
  .data_out(rd_rsp_data_1),
  .empty(rd_rsp_fifo_1_empty),
  .full(rd_rsp_fifo_1_full)
);

reg_fifo_depth_2_w_32_afd_2 blk_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(blk_fifo_0_data_in),
  .flush(flush),
  .pop(pop_blk[0]),
  .push(push_blk[0]),
  .rst_n(rst_n),
  .data_out(blk_fifo_0_data_out),
  .empty(blk_fifo_0_empty),
  .full(blk_fifo_0_full)
);

reg_fifo_depth_2_w_32_afd_2 blk_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(blk_fifo_1_data_in),
  .flush(flush),
  .pop(pop_blk[1]),
  .push(push_blk[1]),
  .rst_n(rst_n),
  .data_out(blk_fifo_1_data_out),
  .empty(blk_fifo_1_empty),
  .full(blk_fifo_1_full)
);

arbiter_4_in_RR_algo rr_arbiter (
  .clk(gclk),
  .clk_en(clk_en),
  .flush(flush),
  .request_in(base_rr),
  .resource_ready(rr_arbiter_resource_ready),
  .rst_n(rst_n),
  .grant_out(mem_acq)
);

endmodule   // buffet_like_16

module fiber_access_16 (
  input logic [1:0] [3:0] buffet_buffet_capacity_log,
  input logic [3:0] [15:0] buffet_data_from_mem_lifted,
  input logic buffet_tile_en,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic read_scanner_block_mode,
  input logic read_scanner_block_rd_out_ready,
  input logic read_scanner_coord_out_ready,
  input logic read_scanner_dense,
  input logic [15:0] read_scanner_dim_size,
  input logic read_scanner_do_repeat,
  input logic [15:0] read_scanner_inner_dim_offset,
  input logic read_scanner_lookup,
  input logic read_scanner_pos_out_ready,
  input logic [15:0] read_scanner_repeat_factor,
  input logic read_scanner_repeat_outer_inner_n,
  input logic read_scanner_root,
  input logic read_scanner_tile_en,
  input logic [16:0] read_scanner_us_pos_in,
  input logic read_scanner_us_pos_in_valid,
  input logic rst_n,
  input logic tile_en,
  input logic vector_reduce_mode,
  input logic [16:0] write_scanner_addr_in,
  input logic write_scanner_addr_in_valid,
  input logic write_scanner_block_mode,
  input logic [16:0] write_scanner_block_wr_in,
  input logic write_scanner_block_wr_in_valid,
  input logic write_scanner_compressed,
  input logic [16:0] write_scanner_data_in,
  input logic write_scanner_data_in_valid,
  input logic write_scanner_init_blank,
  input logic write_scanner_lowest_level,
  input logic [15:0] write_scanner_stop_lvl,
  input logic write_scanner_tile_en,
  output logic [8:0] buffet_addr_to_mem_lifted,
  output logic [3:0] [15:0] buffet_data_to_mem_lifted,
  output logic buffet_ren_to_mem_lifted,
  output logic buffet_wen_to_mem_lifted,
  output logic [16:0] read_scanner_block_rd_out,
  output logic read_scanner_block_rd_out_valid,
  output logic [16:0] read_scanner_coord_out,
  output logic read_scanner_coord_out_valid,
  output logic [16:0] read_scanner_pos_out,
  output logic read_scanner_pos_out_valid,
  output logic read_scanner_us_pos_in_ready,
  output logic write_scanner_addr_in_ready,
  output logic write_scanner_block_wr_in_ready,
  output logic write_scanner_data_in_ready
);

typedef enum logic[2:0] {
  DS_READ_ROW = 3'h0,
  INIT_BLANK_SEND_DONE = 3'h1,
  INIT_BLANK_SEND_S0 = 3'h2,
  ISSUE_READ_SEND_DONE = 3'h3,
  ISSUE_READ_SEND_REF_CNT = 3'h4,
  ISSUE_READ_SEND_S0 = 3'h5,
  PROCESS_ROW = 3'h6,
  START = 3'h7
} vr_seq_state;
logic [16:0] S_level_0;
logic [16:0] S_level_1;
logic [16:0] S_level_2;
logic [0:0][16:0] buffet_wr_ID;
logic buffet_wr_ID_ready;
logic buffet_wr_ID_valid;
logic [0:0][16:0] buffet_wr_addr;
logic buffet_wr_addr_ready;
logic buffet_wr_addr_valid;
logic [0:0][16:0] buffet_wr_data;
logic buffet_wr_data_ready;
logic buffet_wr_data_valid;
logic done_sent_to_ds;
logic done_sent_to_ds_d1;
logic [16:0] done_token;
logic gclk;
logic [2:0] highest_seen_stoken;
logic input_row_fully_processed;
logic input_row_fully_processed_sticky_sticky;
logic input_row_fully_processed_sticky_was_high;
logic is_stop_token;
logic new_highest_stoken_seen;
logic output_matrix_fully_accumulated;
logic output_matrix_fully_accumulated_sticky_sticky;
logic output_matrix_fully_accumulated_sticky_was_high;
logic output_row_fully_accumulated;
logic output_row_fully_accumulated_sticky_sticky;
logic output_row_fully_accumulated_sticky_was_high;
logic [0:0][16:0] read_scanner_addr_out_0;
logic read_scanner_addr_out_0_ready;
logic read_scanner_addr_out_0_valid;
logic [0:0][16:0] read_scanner_addr_out_1;
logic read_scanner_addr_out_1_ready;
logic read_scanner_addr_out_1_valid;
logic [0:0][16:0] read_scanner_op_out_0;
logic read_scanner_op_out_0_ready;
logic read_scanner_op_out_0_valid;
logic [0:0][16:0] read_scanner_op_out_1;
logic read_scanner_op_out_1_ready;
logic read_scanner_op_out_1_valid;
logic [16:0] read_scanner_pos_out_0;
logic read_scanner_pos_out_valid_0;
logic [0:0][16:0] read_scanner_rd_rsp_data_in_0;
logic read_scanner_rd_rsp_data_in_0_ready;
logic read_scanner_rd_rsp_data_in_0_valid;
logic [0:0][16:0] read_scanner_rd_rsp_data_in_1;
logic read_scanner_rd_rsp_data_in_1_ready;
logic read_scanner_rd_rsp_data_in_1_valid;
logic [16:0] read_scanner_us_pos_in_0;
logic read_scanner_us_pos_in_valid_0;
logic rs_has_prepped_ds_row;
logic [16:0] semi_done_token;
logic vr_fsm_init_blank_DONE;
logic vr_fsm_init_blank_S0;
logic [16:0] vr_fsm_pos_to_read_scanner;
logic vr_fsm_pos_valid_to_read_scanner;
vr_seq_state vr_seq_current_state;
vr_seq_state vr_seq_next_state;
assign gclk = clk & tile_en;
assign S_level_0 = {1'h1, 16'h0};
assign S_level_1 = {1'h1, 15'h0, 1'h1};
assign S_level_2 = {1'h1, 14'h0, 1'h1, 1'h0};
assign done_token = {1'h1, 7'h0, 1'h1, 8'h0};
assign semi_done_token = {1'h1, 11'h0, 1'h1, 4'h0};
assign done_sent_to_ds = (read_scanner_pos_out_0 == done_token) & read_scanner_pos_out_valid_0 &
    read_scanner_pos_out_ready;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    done_sent_to_ds_d1 <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      done_sent_to_ds_d1 <= 1'h0;
    end
    else done_sent_to_ds_d1 <= done_sent_to_ds;
  end
end
assign is_stop_token = (write_scanner_data_in[16] == 1'h1) & (~(write_scanner_data_in == done_token)) &
    (~(write_scanner_data_in == semi_done_token));
assign new_highest_stoken_seen = is_stop_token & (write_scanner_data_in[2:0] > highest_seen_stoken) &
    write_scanner_data_in_valid & write_scanner_data_in_ready;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    highest_seen_stoken <= S_level_0[2:0];
  end
  else if (clk_en) begin
    if (flush) begin
      highest_seen_stoken <= S_level_0[2:0];
    end
    else if (done_sent_to_ds) begin
      highest_seen_stoken <= S_level_0[2:0];
    end
    else if (new_highest_stoken_seen) begin
      highest_seen_stoken <= write_scanner_data_in[2:0];
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    vr_seq_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      vr_seq_current_state <= START;
    end
    else vr_seq_current_state <= vr_seq_next_state;
  end
end
always_comb begin
  vr_seq_next_state = vr_seq_current_state;
  unique case (vr_seq_current_state)
    DS_READ_ROW: begin
        if (rs_has_prepped_ds_row) begin
          vr_seq_next_state = INIT_BLANK_SEND_S0;
        end
        else vr_seq_next_state = DS_READ_ROW;
      end
    INIT_BLANK_SEND_DONE: begin
        if (read_scanner_coord_out_ready) begin
          vr_seq_next_state = PROCESS_ROW;
        end
        else vr_seq_next_state = INIT_BLANK_SEND_DONE;
      end
    INIT_BLANK_SEND_S0: begin
        if (read_scanner_coord_out_ready) begin
          vr_seq_next_state = INIT_BLANK_SEND_DONE;
        end
        else vr_seq_next_state = INIT_BLANK_SEND_S0;
      end
    ISSUE_READ_SEND_DONE: begin
        if (read_scanner_us_pos_in_ready & (~output_row_fully_accumulated)) begin
          vr_seq_next_state = PROCESS_ROW;
        end
        else if (read_scanner_us_pos_in_ready & output_row_fully_accumulated) begin
          vr_seq_next_state = DS_READ_ROW;
        end
        else vr_seq_next_state = ISSUE_READ_SEND_DONE;
      end
    ISSUE_READ_SEND_REF_CNT: begin
        if (read_scanner_us_pos_in_ready & ((~output_matrix_fully_accumulated) | (output_matrix_fully_accumulated & (~(S_level_1[2:0] < highest_seen_stoken))))) begin
          vr_seq_next_state = ISSUE_READ_SEND_DONE;
        end
        else if (read_scanner_us_pos_in_ready & output_matrix_fully_accumulated & (S_level_1[2:0] < highest_seen_stoken)) begin
          vr_seq_next_state = ISSUE_READ_SEND_S0;
        end
        else vr_seq_next_state = ISSUE_READ_SEND_REF_CNT;
      end
    ISSUE_READ_SEND_S0: begin
        if (read_scanner_us_pos_in_ready) begin
          vr_seq_next_state = ISSUE_READ_SEND_DONE;
        end
        else vr_seq_next_state = ISSUE_READ_SEND_S0;
      end
    PROCESS_ROW: begin
        if (input_row_fully_processed) begin
          vr_seq_next_state = ISSUE_READ_SEND_REF_CNT;
        end
        else vr_seq_next_state = PROCESS_ROW;
      end
    START: begin
        if (vector_reduce_mode) begin
          vr_seq_next_state = INIT_BLANK_SEND_S0;
        end
        else vr_seq_next_state = START;
      end
    default: begin end
  endcase
end
always_comb begin
  unique case (vr_seq_current_state)
    DS_READ_ROW: begin :vr_seq_DS_READ_ROW_Output
        vr_fsm_pos_to_read_scanner = 17'h0;
        vr_fsm_pos_valid_to_read_scanner = 1'h0;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_DS_READ_ROW_Output
    INIT_BLANK_SEND_DONE: begin :vr_seq_INIT_BLANK_SEND_DONE_Output
        vr_fsm_pos_to_read_scanner = 17'h0;
        vr_fsm_pos_valid_to_read_scanner = 1'h0;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h1;
      end :vr_seq_INIT_BLANK_SEND_DONE_Output
    INIT_BLANK_SEND_S0: begin :vr_seq_INIT_BLANK_SEND_S0_Output
        vr_fsm_pos_to_read_scanner = 17'h0;
        vr_fsm_pos_valid_to_read_scanner = 1'h0;
        vr_fsm_init_blank_S0 = 1'h1;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_INIT_BLANK_SEND_S0_Output
    ISSUE_READ_SEND_DONE: begin :vr_seq_ISSUE_READ_SEND_DONE_Output
        vr_fsm_pos_to_read_scanner = done_token;
        vr_fsm_pos_valid_to_read_scanner = 1'h1;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_ISSUE_READ_SEND_DONE_Output
    ISSUE_READ_SEND_REF_CNT: begin :vr_seq_ISSUE_READ_SEND_REF_CNT_Output
        vr_fsm_pos_to_read_scanner = 17'h0;
        vr_fsm_pos_valid_to_read_scanner = 1'h1;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_ISSUE_READ_SEND_REF_CNT_Output
    ISSUE_READ_SEND_S0: begin :vr_seq_ISSUE_READ_SEND_S0_Output
        vr_fsm_pos_to_read_scanner = S_level_0;
        vr_fsm_pos_valid_to_read_scanner = 1'h1;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_ISSUE_READ_SEND_S0_Output
    PROCESS_ROW: begin :vr_seq_PROCESS_ROW_Output
        vr_fsm_pos_to_read_scanner = 17'h0;
        vr_fsm_pos_valid_to_read_scanner = 1'h0;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_PROCESS_ROW_Output
    START: begin :vr_seq_START_Output
        vr_fsm_pos_to_read_scanner = 17'h0;
        vr_fsm_pos_valid_to_read_scanner = 1'h0;
        vr_fsm_init_blank_S0 = 1'h0;
        vr_fsm_init_blank_DONE = 1'h0;
      end :vr_seq_START_Output
    default: begin end
  endcase
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    input_row_fully_processed_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      input_row_fully_processed_sticky_was_high <= 1'h0;
    end
    else if (vr_seq_current_state == ISSUE_READ_SEND_DONE) begin
      input_row_fully_processed_sticky_was_high <= 1'h0;
    end
    else if (((write_scanner_data_in == S_level_0) | (write_scanner_data_in == S_level_1) | (write_scanner_data_in == S_level_2)) & write_scanner_data_in_valid & write_scanner_data_in_ready) begin
      input_row_fully_processed_sticky_was_high <= 1'h1;
    end
  end
end
assign input_row_fully_processed_sticky_sticky = (((write_scanner_data_in == S_level_0) | (write_scanner_data_in == S_level_1) |
    (write_scanner_data_in == S_level_2)) & write_scanner_data_in_valid &
    write_scanner_data_in_ready) | input_row_fully_processed_sticky_was_high;
assign input_row_fully_processed = input_row_fully_processed_sticky_sticky;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    output_row_fully_accumulated_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      output_row_fully_accumulated_sticky_was_high <= 1'h0;
    end
    else if (vr_seq_current_state == INIT_BLANK_SEND_S0) begin
      output_row_fully_accumulated_sticky_was_high <= 1'h0;
    end
    else if (((write_scanner_data_in == S_level_1) | (write_scanner_data_in == S_level_2)) & write_scanner_data_in_valid & write_scanner_data_in_ready) begin
      output_row_fully_accumulated_sticky_was_high <= 1'h1;
    end
  end
end
assign output_row_fully_accumulated_sticky_sticky = (((write_scanner_data_in == S_level_1) | (write_scanner_data_in == S_level_2)) &
    write_scanner_data_in_valid & write_scanner_data_in_ready) |
    output_row_fully_accumulated_sticky_was_high;
assign output_row_fully_accumulated = output_row_fully_accumulated_sticky_sticky;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    output_matrix_fully_accumulated_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      output_matrix_fully_accumulated_sticky_was_high <= 1'h0;
    end
    else if (done_sent_to_ds_d1) begin
      output_matrix_fully_accumulated_sticky_was_high <= 1'h0;
    end
    else if ((write_scanner_data_in == done_token) & write_scanner_data_in_valid & write_scanner_data_in_ready) begin
      output_matrix_fully_accumulated_sticky_was_high <= 1'h1;
    end
  end
end
assign output_matrix_fully_accumulated_sticky_sticky = ((write_scanner_data_in == done_token) & write_scanner_data_in_valid &
    write_scanner_data_in_ready) | output_matrix_fully_accumulated_sticky_was_high;
assign output_matrix_fully_accumulated = output_matrix_fully_accumulated_sticky_sticky;
assign read_scanner_pos_out = read_scanner_pos_out_0;
assign read_scanner_pos_out_valid = read_scanner_pos_out_valid_0;
assign read_scanner_us_pos_in_0 = vector_reduce_mode ? vr_fsm_pos_to_read_scanner: read_scanner_us_pos_in;
assign read_scanner_us_pos_in_valid_0 = vector_reduce_mode ? vr_fsm_pos_valid_to_read_scanner:
    read_scanner_us_pos_in_valid;
scanner_pipe read_scanner (
  .addr_out_0_ready(read_scanner_addr_out_0_ready),
  .addr_out_1_ready(read_scanner_addr_out_1_ready),
  .block_mode(read_scanner_block_mode),
  .block_rd_out_ready(read_scanner_block_rd_out_ready),
  .clk(gclk),
  .clk_en(clk_en),
  .coord_out_ready(read_scanner_coord_out_ready),
  .dense(read_scanner_dense),
  .dim_size(read_scanner_dim_size),
  .do_repeat(read_scanner_do_repeat),
  .flush(flush),
  .inner_dim_offset(read_scanner_inner_dim_offset),
  .lookup(read_scanner_lookup),
  .op_out_0_ready(read_scanner_op_out_0_ready),
  .op_out_1_ready(read_scanner_op_out_1_ready),
  .output_matrix_fully_accumulated(output_matrix_fully_accumulated),
  .output_row_fully_accumulated(output_row_fully_accumulated),
  .pos_out_ready(read_scanner_pos_out_ready),
  .pos_to_read_scanner_from_vr_fsm(vr_fsm_pos_to_read_scanner),
  .rd_rsp_data_in_0(read_scanner_rd_rsp_data_in_0),
  .rd_rsp_data_in_0_valid(read_scanner_rd_rsp_data_in_0_valid),
  .rd_rsp_data_in_1(read_scanner_rd_rsp_data_in_1),
  .rd_rsp_data_in_1_valid(read_scanner_rd_rsp_data_in_1_valid),
  .repeat_factor(read_scanner_repeat_factor),
  .repeat_outer_inner_n(read_scanner_repeat_outer_inner_n),
  .root(read_scanner_root),
  .rst_n(rst_n),
  .tile_en(read_scanner_tile_en),
  .us_pos_in(read_scanner_us_pos_in_0),
  .us_pos_in_valid(read_scanner_us_pos_in_valid_0),
  .vector_reduce_mode(vector_reduce_mode),
  .vr_fsm_state_init_blank_DONE(vr_fsm_init_blank_DONE),
  .vr_fsm_state_init_blank_S0(vr_fsm_init_blank_S0),
  .addr_out_0(read_scanner_addr_out_0),
  .addr_out_0_valid(read_scanner_addr_out_0_valid),
  .addr_out_1(read_scanner_addr_out_1),
  .addr_out_1_valid(read_scanner_addr_out_1_valid),
  .block_rd_out(read_scanner_block_rd_out),
  .block_rd_out_valid(read_scanner_block_rd_out_valid),
  .coord_out(read_scanner_coord_out),
  .coord_out_valid(read_scanner_coord_out_valid),
  .op_out_0(read_scanner_op_out_0),
  .op_out_0_valid(read_scanner_op_out_0_valid),
  .op_out_1(read_scanner_op_out_1),
  .op_out_1_valid(read_scanner_op_out_1_valid),
  .pos_out(read_scanner_pos_out_0),
  .pos_out_valid(read_scanner_pos_out_valid_0),
  .rd_rsp_data_in_0_ready(read_scanner_rd_rsp_data_in_0_ready),
  .rd_rsp_data_in_1_ready(read_scanner_rd_rsp_data_in_1_ready),
  .rs_has_prepped_ds_row(rs_has_prepped_ds_row),
  .us_pos_in_ready(read_scanner_us_pos_in_ready)
);

buffet_like_16 buffet (
  .buffet_capacity_log(buffet_buffet_capacity_log),
  .clk(gclk),
  .clk_en(clk_en),
  .data_from_mem(buffet_data_from_mem_lifted),
  .flush(flush),
  .rd_addr_0(read_scanner_addr_out_0),
  .rd_addr_0_valid(read_scanner_addr_out_0_valid),
  .rd_addr_1(read_scanner_addr_out_1),
  .rd_addr_1_valid(read_scanner_addr_out_1_valid),
  .rd_op_0(read_scanner_op_out_0),
  .rd_op_0_valid(read_scanner_op_out_0_valid),
  .rd_op_1(read_scanner_op_out_1),
  .rd_op_1_valid(read_scanner_op_out_1_valid),
  .rd_rsp_data_0_ready(read_scanner_rd_rsp_data_in_0_ready),
  .rd_rsp_data_1_ready(read_scanner_rd_rsp_data_in_1_ready),
  .rst_n(rst_n),
  .tile_en(buffet_tile_en),
  .wr_ID(buffet_wr_ID),
  .wr_ID_valid(buffet_wr_ID_valid),
  .wr_addr(buffet_wr_addr),
  .wr_addr_valid(buffet_wr_addr_valid),
  .wr_data(buffet_wr_data),
  .wr_data_valid(buffet_wr_data_valid),
  .addr_to_mem(buffet_addr_to_mem_lifted),
  .data_to_mem(buffet_data_to_mem_lifted),
  .rd_addr_0_ready(read_scanner_addr_out_0_ready),
  .rd_addr_1_ready(read_scanner_addr_out_1_ready),
  .rd_op_0_ready(read_scanner_op_out_0_ready),
  .rd_op_1_ready(read_scanner_op_out_1_ready),
  .rd_rsp_data_0(read_scanner_rd_rsp_data_in_0),
  .rd_rsp_data_0_valid(read_scanner_rd_rsp_data_in_0_valid),
  .rd_rsp_data_1(read_scanner_rd_rsp_data_in_1),
  .rd_rsp_data_1_valid(read_scanner_rd_rsp_data_in_1_valid),
  .ren_to_mem(buffet_ren_to_mem_lifted),
  .wen_to_mem(buffet_wen_to_mem_lifted),
  .wr_ID_ready(buffet_wr_ID_ready),
  .wr_addr_ready(buffet_wr_addr_ready),
  .wr_data_ready(buffet_wr_data_ready)
);

write_scanner write_scanner (
  .ID_out_ready(buffet_wr_ID_ready),
  .addr_in(write_scanner_addr_in),
  .addr_in_valid(write_scanner_addr_in_valid),
  .addr_out_ready(buffet_wr_addr_ready),
  .block_mode(write_scanner_block_mode),
  .block_wr_in(write_scanner_block_wr_in),
  .block_wr_in_valid(write_scanner_block_wr_in_valid),
  .clk(gclk),
  .clk_en(clk_en),
  .compressed(write_scanner_compressed),
  .data_in(write_scanner_data_in),
  .data_in_valid(write_scanner_data_in_valid),
  .data_out_ready(buffet_wr_data_ready),
  .flush(flush),
  .init_blank(write_scanner_init_blank),
  .lowest_level(write_scanner_lowest_level),
  .rst_n(rst_n),
  .stop_lvl(write_scanner_stop_lvl),
  .tile_en(write_scanner_tile_en),
  .vector_reduce_mode(vector_reduce_mode),
  .ID_out(buffet_wr_ID),
  .ID_out_valid(buffet_wr_ID_valid),
  .addr_in_ready(write_scanner_addr_in_ready),
  .addr_out(buffet_wr_addr),
  .addr_out_valid(buffet_wr_addr_valid),
  .block_wr_in_ready(write_scanner_block_wr_in_ready),
  .data_in_ready(write_scanner_data_in_ready),
  .data_out(buffet_wr_data),
  .data_out_valid(buffet_wr_data_valid)
);

endmodule   // fiber_access_16

module fiber_access_16_flat (
  input logic clk,
  input logic clk_en,
  input logic [3:0] fiber_access_16_inst_buffet_buffet_capacity_log_0,
  input logic [3:0] fiber_access_16_inst_buffet_buffet_capacity_log_1,
  input logic [3:0] [15:0] fiber_access_16_inst_buffet_data_from_mem_lifted_lifted,
  input logic fiber_access_16_inst_buffet_tile_en,
  input logic fiber_access_16_inst_read_scanner_block_mode,
  input logic fiber_access_16_inst_read_scanner_dense,
  input logic [15:0] fiber_access_16_inst_read_scanner_dim_size,
  input logic fiber_access_16_inst_read_scanner_do_repeat,
  input logic [15:0] fiber_access_16_inst_read_scanner_inner_dim_offset,
  input logic fiber_access_16_inst_read_scanner_lookup,
  input logic [15:0] fiber_access_16_inst_read_scanner_repeat_factor,
  input logic fiber_access_16_inst_read_scanner_repeat_outer_inner_n,
  input logic fiber_access_16_inst_read_scanner_root,
  input logic fiber_access_16_inst_read_scanner_tile_en,
  input logic fiber_access_16_inst_tile_en,
  input logic fiber_access_16_inst_vector_reduce_mode,
  input logic fiber_access_16_inst_write_scanner_block_mode,
  input logic fiber_access_16_inst_write_scanner_compressed,
  input logic fiber_access_16_inst_write_scanner_init_blank,
  input logic fiber_access_16_inst_write_scanner_lowest_level,
  input logic [15:0] fiber_access_16_inst_write_scanner_stop_lvl,
  input logic fiber_access_16_inst_write_scanner_tile_en,
  input logic flush,
  input logic read_scanner_block_rd_out_ready_f_,
  input logic read_scanner_coord_out_ready_f_,
  input logic read_scanner_pos_out_ready_f_,
  input logic [0:0] [16:0] read_scanner_us_pos_in_f_,
  input logic read_scanner_us_pos_in_valid_f_,
  input logic rst_n,
  input logic [0:0] [16:0] write_scanner_addr_in_f_,
  input logic write_scanner_addr_in_valid_f_,
  input logic [0:0] [16:0] write_scanner_block_wr_in_f_,
  input logic write_scanner_block_wr_in_valid_f_,
  input logic [0:0] [16:0] write_scanner_data_in_f_,
  input logic write_scanner_data_in_valid_f_,
  output logic [8:0] fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted,
  output logic [3:0] [15:0] fiber_access_16_inst_buffet_data_to_mem_lifted_lifted,
  output logic fiber_access_16_inst_buffet_ren_to_mem_lifted_lifted,
  output logic fiber_access_16_inst_buffet_wen_to_mem_lifted_lifted,
  output logic [0:0] [16:0] read_scanner_block_rd_out_f_,
  output logic read_scanner_block_rd_out_valid_f_,
  output logic [0:0] [16:0] read_scanner_coord_out_f_,
  output logic read_scanner_coord_out_valid_f_,
  output logic [0:0] [16:0] read_scanner_pos_out_f_,
  output logic read_scanner_pos_out_valid_f_,
  output logic read_scanner_us_pos_in_ready_f_,
  output logic write_scanner_addr_in_ready_f_,
  output logic write_scanner_block_wr_in_ready_f_,
  output logic write_scanner_data_in_ready_f_
);

logic [1:0][3:0] fiber_access_16_inst_buffet_buffet_capacity_log;
assign fiber_access_16_inst_buffet_buffet_capacity_log[0] = fiber_access_16_inst_buffet_buffet_capacity_log_0;
assign fiber_access_16_inst_buffet_buffet_capacity_log[1] = fiber_access_16_inst_buffet_buffet_capacity_log_1;
fiber_access_16 fiber_access_16_inst (
  .buffet_buffet_capacity_log(fiber_access_16_inst_buffet_buffet_capacity_log),
  .buffet_data_from_mem_lifted(fiber_access_16_inst_buffet_data_from_mem_lifted_lifted),
  .buffet_tile_en(fiber_access_16_inst_buffet_tile_en),
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .read_scanner_block_mode(fiber_access_16_inst_read_scanner_block_mode),
  .read_scanner_block_rd_out_ready(read_scanner_block_rd_out_ready_f_),
  .read_scanner_coord_out_ready(read_scanner_coord_out_ready_f_),
  .read_scanner_dense(fiber_access_16_inst_read_scanner_dense),
  .read_scanner_dim_size(fiber_access_16_inst_read_scanner_dim_size),
  .read_scanner_do_repeat(fiber_access_16_inst_read_scanner_do_repeat),
  .read_scanner_inner_dim_offset(fiber_access_16_inst_read_scanner_inner_dim_offset),
  .read_scanner_lookup(fiber_access_16_inst_read_scanner_lookup),
  .read_scanner_pos_out_ready(read_scanner_pos_out_ready_f_),
  .read_scanner_repeat_factor(fiber_access_16_inst_read_scanner_repeat_factor),
  .read_scanner_repeat_outer_inner_n(fiber_access_16_inst_read_scanner_repeat_outer_inner_n),
  .read_scanner_root(fiber_access_16_inst_read_scanner_root),
  .read_scanner_tile_en(fiber_access_16_inst_read_scanner_tile_en),
  .read_scanner_us_pos_in(read_scanner_us_pos_in_f_),
  .read_scanner_us_pos_in_valid(read_scanner_us_pos_in_valid_f_),
  .rst_n(rst_n),
  .tile_en(fiber_access_16_inst_tile_en),
  .vector_reduce_mode(fiber_access_16_inst_vector_reduce_mode),
  .write_scanner_addr_in(write_scanner_addr_in_f_),
  .write_scanner_addr_in_valid(write_scanner_addr_in_valid_f_),
  .write_scanner_block_mode(fiber_access_16_inst_write_scanner_block_mode),
  .write_scanner_block_wr_in(write_scanner_block_wr_in_f_),
  .write_scanner_block_wr_in_valid(write_scanner_block_wr_in_valid_f_),
  .write_scanner_compressed(fiber_access_16_inst_write_scanner_compressed),
  .write_scanner_data_in(write_scanner_data_in_f_),
  .write_scanner_data_in_valid(write_scanner_data_in_valid_f_),
  .write_scanner_init_blank(fiber_access_16_inst_write_scanner_init_blank),
  .write_scanner_lowest_level(fiber_access_16_inst_write_scanner_lowest_level),
  .write_scanner_stop_lvl(fiber_access_16_inst_write_scanner_stop_lvl),
  .write_scanner_tile_en(fiber_access_16_inst_write_scanner_tile_en),
  .buffet_addr_to_mem_lifted(fiber_access_16_inst_buffet_addr_to_mem_lifted_lifted),
  .buffet_data_to_mem_lifted(fiber_access_16_inst_buffet_data_to_mem_lifted_lifted),
  .buffet_ren_to_mem_lifted(fiber_access_16_inst_buffet_ren_to_mem_lifted_lifted),
  .buffet_wen_to_mem_lifted(fiber_access_16_inst_buffet_wen_to_mem_lifted_lifted),
  .read_scanner_block_rd_out(read_scanner_block_rd_out_f_),
  .read_scanner_block_rd_out_valid(read_scanner_block_rd_out_valid_f_),
  .read_scanner_coord_out(read_scanner_coord_out_f_),
  .read_scanner_coord_out_valid(read_scanner_coord_out_valid_f_),
  .read_scanner_pos_out(read_scanner_pos_out_f_),
  .read_scanner_pos_out_valid(read_scanner_pos_out_valid_f_),
  .read_scanner_us_pos_in_ready(read_scanner_us_pos_in_ready_f_),
  .write_scanner_addr_in_ready(write_scanner_addr_in_ready_f_),
  .write_scanner_block_wr_in_ready(write_scanner_block_wr_in_ready_f_),
  .write_scanner_data_in_ready(write_scanner_data_in_ready_f_)
);

endmodule   // fiber_access_16_flat

module for_loop_3_11 #(
  parameter CONFIG_WIDTH = 5'hB,
  parameter ITERATOR_SUPPORT = 3'h3,
  parameter ITERATOR_SUPPORT2 = 2'h2
)
(
  input logic clk,
  input logic clk_en,
  input logic [2:0] dimensionality,
  input logic flush,
  input logic [2:0] [10:0] ranges,
  input logic rst_n,
  input logic step,
  output logic [1:0] mux_sel_out,
  output logic restart
);

logic [2:0] clear;
logic [2:0][10:0] dim_counter;
logic done;
logic [2:0] inc;
logic [10:0] inced_cnt;
logic [2:0] max_value;
logic maxed_value;
logic [1:0] mux_sel;
assign mux_sel_out = mux_sel;
assign inced_cnt = dim_counter[mux_sel] + 11'h1;
assign maxed_value = (dim_counter[mux_sel] == ranges[mux_sel]) & inc[mux_sel];
always_comb begin
  mux_sel = 2'h0;
  done = 1'h0;
  if (~done) begin
    if ((~max_value[0]) & (dimensionality > 3'h0)) begin
      mux_sel = 2'h0;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[1]) & (dimensionality > 3'h1)) begin
      mux_sel = 2'h1;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[2]) & (dimensionality > 3'h2)) begin
      mux_sel = 2'h2;
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
  if ((5'h0 == 5'h0) & step & (dimensionality > 3'h0)) begin
    inc[0] = 1'h1;
  end
  else if ((mux_sel == 2'h0) & step & (dimensionality > 3'h0)) begin
    inc[0] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[0] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[0] <= 11'h0;
    end
    else if (clear[0]) begin
      dim_counter[0] <= 11'h0;
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
  if ((5'h1 == 5'h0) & step & (dimensionality > 3'h1)) begin
    inc[1] = 1'h1;
  end
  else if ((mux_sel == 2'h1) & step & (dimensionality > 3'h1)) begin
    inc[1] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[1] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[1] <= 11'h0;
    end
    else if (clear[1]) begin
      dim_counter[1] <= 11'h0;
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
  if ((5'h2 == 5'h0) & step & (dimensionality > 3'h2)) begin
    inc[2] = 1'h1;
  end
  else if ((mux_sel == 2'h2) & step & (dimensionality > 3'h2)) begin
    inc[2] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[2] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[2] <= 11'h0;
    end
    else if (clear[2]) begin
      dim_counter[2] <= 11'h0;
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
assign restart = step & (~done);
endmodule   // for_loop_3_11

module for_loop_6_11 #(
  parameter CONFIG_WIDTH = 5'hB,
  parameter ITERATOR_SUPPORT = 4'h6,
  parameter ITERATOR_SUPPORT2 = 2'h2
)
(
  input logic clk,
  input logic clk_en,
  input logic [3:0] dimensionality,
  input logic flush,
  input logic [5:0] [10:0] ranges,
  input logic rst_n,
  input logic step,
  output logic [2:0] mux_sel_out,
  output logic restart
);

logic [5:0] clear;
logic [5:0][10:0] dim_counter;
logic done;
logic [5:0] inc;
logic [10:0] inced_cnt;
logic [5:0] max_value;
logic maxed_value;
logic [2:0] mux_sel;
assign mux_sel_out = mux_sel;
assign inced_cnt = dim_counter[mux_sel] + 11'h1;
assign maxed_value = (dim_counter[mux_sel] == ranges[mux_sel]) & inc[mux_sel];
always_comb begin
  mux_sel = 3'h0;
  done = 1'h0;
  if (~done) begin
    if ((~max_value[0]) & (dimensionality > 4'h0)) begin
      mux_sel = 3'h0;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[1]) & (dimensionality > 4'h1)) begin
      mux_sel = 3'h1;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[2]) & (dimensionality > 4'h2)) begin
      mux_sel = 3'h2;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[3]) & (dimensionality > 4'h3)) begin
      mux_sel = 3'h3;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[4]) & (dimensionality > 4'h4)) begin
      mux_sel = 3'h4;
      done = 1'h1;
    end
  end
  if (~done) begin
    if ((~max_value[5]) & (dimensionality > 4'h5)) begin
      mux_sel = 3'h5;
      done = 1'h1;
    end
  end
end
always_comb begin
  clear[0] = 1'h0;
  if (((mux_sel > 3'h0) | (~done)) & step) begin
    clear[0] = 1'h1;
  end
end
always_comb begin
  inc[0] = 1'h0;
  if ((5'h0 == 5'h0) & step & (dimensionality > 4'h0)) begin
    inc[0] = 1'h1;
  end
  else if ((mux_sel == 3'h0) & step & (dimensionality > 4'h0)) begin
    inc[0] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[0] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[0] <= 11'h0;
    end
    else if (clear[0]) begin
      dim_counter[0] <= 11'h0;
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
  if (((mux_sel > 3'h1) | (~done)) & step) begin
    clear[1] = 1'h1;
  end
end
always_comb begin
  inc[1] = 1'h0;
  if ((5'h1 == 5'h0) & step & (dimensionality > 4'h1)) begin
    inc[1] = 1'h1;
  end
  else if ((mux_sel == 3'h1) & step & (dimensionality > 4'h1)) begin
    inc[1] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[1] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[1] <= 11'h0;
    end
    else if (clear[1]) begin
      dim_counter[1] <= 11'h0;
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
  if (((mux_sel > 3'h2) | (~done)) & step) begin
    clear[2] = 1'h1;
  end
end
always_comb begin
  inc[2] = 1'h0;
  if ((5'h2 == 5'h0) & step & (dimensionality > 4'h2)) begin
    inc[2] = 1'h1;
  end
  else if ((mux_sel == 3'h2) & step & (dimensionality > 4'h2)) begin
    inc[2] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[2] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[2] <= 11'h0;
    end
    else if (clear[2]) begin
      dim_counter[2] <= 11'h0;
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
  if (((mux_sel > 3'h3) | (~done)) & step) begin
    clear[3] = 1'h1;
  end
end
always_comb begin
  inc[3] = 1'h0;
  if ((5'h3 == 5'h0) & step & (dimensionality > 4'h3)) begin
    inc[3] = 1'h1;
  end
  else if ((mux_sel == 3'h3) & step & (dimensionality > 4'h3)) begin
    inc[3] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[3] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[3] <= 11'h0;
    end
    else if (clear[3]) begin
      dim_counter[3] <= 11'h0;
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
always_comb begin
  clear[4] = 1'h0;
  if (((mux_sel > 3'h4) | (~done)) & step) begin
    clear[4] = 1'h1;
  end
end
always_comb begin
  inc[4] = 1'h0;
  if ((5'h4 == 5'h0) & step & (dimensionality > 4'h4)) begin
    inc[4] = 1'h1;
  end
  else if ((mux_sel == 3'h4) & step & (dimensionality > 4'h4)) begin
    inc[4] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[4] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[4] <= 11'h0;
    end
    else if (clear[4]) begin
      dim_counter[4] <= 11'h0;
    end
    else if (inc[4]) begin
      dim_counter[4] <= inced_cnt;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    max_value[4] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      max_value[4] <= 1'h0;
    end
    else if (clear[4]) begin
      max_value[4] <= 1'h0;
    end
    else if (inc[4]) begin
      max_value[4] <= maxed_value;
    end
  end
end
always_comb begin
  clear[5] = 1'h0;
  if (((mux_sel > 3'h5) | (~done)) & step) begin
    clear[5] = 1'h1;
  end
end
always_comb begin
  inc[5] = 1'h0;
  if ((5'h5 == 5'h0) & step & (dimensionality > 4'h5)) begin
    inc[5] = 1'h1;
  end
  else if ((mux_sel == 3'h5) & step & (dimensionality > 4'h5)) begin
    inc[5] = 1'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    dim_counter[5] <= 11'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      dim_counter[5] <= 11'h0;
    end
    else if (clear[5]) begin
      dim_counter[5] <= 11'h0;
    end
    else if (inc[5]) begin
      dim_counter[5] <= inced_cnt;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    max_value[5] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      max_value[5] <= 1'h0;
    end
    else if (clear[5]) begin
      max_value[5] <= 1'h0;
    end
    else if (inc[5]) begin
      max_value[5] <= maxed_value;
    end
  end
end
assign restart = step & (~done);
endmodule   // for_loop_6_11

module reg_fifo_depth_0_w_16_afd_2 (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [15:0] data_in,
  input logic flush,
  input logic pop,
  input logic push,
  input logic rst_n,
  output logic almost_full,
  output logic [0:0] [15:0] data_out,
  output logic empty,
  output logic full,
  output logic valid
);

assign data_out = data_in;
assign valid = push;
assign empty = ~push;
assign full = ~pop;
assign almost_full = ~pop;
endmodule   // reg_fifo_depth_0_w_16_afd_2

module reg_fifo_depth_2_w_16_afd_2 (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [15:0] data_in,
  input logic flush,
  input logic pop,
  input logic push,
  input logic rst_n,
  output logic almost_full,
  output logic [0:0] [15:0] data_out,
  output logic empty,
  output logic full,
  output logic valid
);

logic [1:0] num_items;
logic passthru;
logic rd_ptr;
logic read;
logic [1:0][0:0][15:0] reg_array;
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
    reg_array <= 32'h0;
  end
  else if (flush) begin
    reg_array <= 32'h0;
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
endmodule   // reg_fifo_depth_2_w_16_afd_2

module reg_fifo_depth_2_w_17_afd_0 (
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
assign almost_full = num_items >= 2'h2;
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
endmodule   // reg_fifo_depth_2_w_17_afd_0

module reg_fifo_depth_2_w_32_afd_2 (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [31:0] data_in,
  input logic flush,
  input logic pop,
  input logic push,
  input logic rst_n,
  output logic almost_full,
  output logic [0:0] [31:0] data_out,
  output logic empty,
  output logic full,
  output logic valid
);

logic [1:0] num_items;
logic passthru;
logic rd_ptr;
logic read;
logic [1:0][0:0][31:0] reg_array;
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
    if (clk_en) begin
      if (clk_en) begin
        if (write & (~read)) begin
          num_items <= num_items + 2'h1;
        end
        else if ((~write) & read) begin
          num_items <= num_items - 2'h1;
        end
      end
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reg_array <= 64'h0;
  end
  else if (flush) begin
    reg_array <= 64'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (clk_en) begin
        if (write) begin
          reg_array[wr_ptr] <= data_in;
        end
      end
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
    if (clk_en) begin
      if (clk_en) begin
        if (write) begin
          if (wr_ptr == 1'h1) begin
            wr_ptr <= 1'h0;
          end
          else wr_ptr <= wr_ptr + 1'h1;
        end
      end
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
    if (clk_en) begin
      if (clk_en) begin
        if (read) begin
          rd_ptr <= rd_ptr + 1'h1;
        end
      end
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
endmodule   // reg_fifo_depth_2_w_32_afd_2

module reservation_fifo_depth_32_w_17_num_per_1 (
  input logic clk,
  input logic clk_en,
  input logic [16:0] data_in_0,
  input logic [16:0] fill_data_in,
  input logic flush,
  input logic pop,
  input logic push_alloc,
  input logic push_fill,
  input logic push_reserve,
  input logic rst_n,
  output logic [16:0] data_out_0,
  output logic empty,
  output logic full,
  output logic valid
);

logic clr_item_ptr;
logic clr_read_ptr;
logic clr_write_ptr;
logic [0:0][16:0] data_in_packed;
logic [0:0][16:0] data_out;
logic enable_reserve_ptr;
logic inc_item_ptr;
logic inc_read_ptr;
logic inc_reserve_count;
logic inc_write_ptr;
logic item_ptr;
logic jump_next_0;
logic [4:0] next_0_valid;
logic [4:0] next_0_valid_d1;
logic [4:0] next_0_valid_high;
logic next_0_valid_high_done;
logic next_0_valid_high_found;
logic [4:0] next_0_valid_low;
logic next_0_valid_low_done;
logic next_0_valid_low_found;
logic [5:0] num_items;
logic read;
logic [4:0] read_ptr_addr;
logic [31:0][0:0][16:0] reg_array;
logic [15:0] reserve_count;
logic [4:0] reserve_ptr_val;
logic [31:0] valid_mask;
logic write_alloc;
logic write_fill;
logic [4:0] write_ptr_addr;
logic write_reserve;
logic write_reserve_final;
assign data_in_packed[0] = data_in_0;
assign data_out_0 = data_out[0];
assign item_ptr = 1'h0;
assign inc_item_ptr = push_reserve;
assign clr_item_ptr = push_reserve & (item_ptr == 1'h0);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_ptr_addr <= 5'h0;
  end
  else if (flush) begin
    read_ptr_addr <= 5'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (clr_read_ptr) begin
        read_ptr_addr <= 5'h0;
      end
      else if (inc_read_ptr) begin
        read_ptr_addr <= read_ptr_addr + 5'h1;
      end
    end
  end
end
assign inc_read_ptr = read;
assign clr_read_ptr = 1'h0;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_ptr_addr <= 5'h0;
  end
  else if (flush) begin
    write_ptr_addr <= 5'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (clr_write_ptr) begin
        write_ptr_addr <= 5'h0;
      end
      else if (inc_write_ptr) begin
        write_ptr_addr <= write_ptr_addr + 5'h1;
      end
    end
  end
end
assign inc_write_ptr = write_alloc | write_fill;
assign clr_write_ptr = 1'h0;
assign jump_next_0 = next_0_valid_high_found | next_0_valid_low_found;
assign enable_reserve_ptr = write_reserve_final | (write_fill & (reserve_ptr_val == write_ptr_addr));

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    next_0_valid_d1 <= 5'h0;
  end
  else if (flush) begin
    next_0_valid_d1 <= 5'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (1'h0) begin
        next_0_valid_d1 <= 5'h0;
      end
      else if (enable_reserve_ptr) begin
        next_0_valid_d1 <= next_0_valid;
      end
    end
  end
end
assign reserve_ptr_val = next_0_valid_d1;
assign next_0_valid = (write_fill & ((next_0_valid_d1 == write_ptr_addr) | ((~next_0_valid_high_found)
    & (~next_0_valid_low_found)) | (next_0_valid_high_found ? next_0_valid_high ==
    write_ptr_addr: next_0_valid_low == write_ptr_addr))) ? write_ptr_addr + 5'h1:
    ((~next_0_valid_high_found) & (~next_0_valid_low_found)) ? write_ptr_addr:
    next_0_valid_high_found ? next_0_valid_high: next_0_valid_low;
assign full = num_items == 6'h20;
assign empty = num_items == 6'h0;
assign write_fill = push_fill & push_alloc & (~full);
assign write_alloc = push_alloc & (~full);
assign write_reserve = inc_item_ptr;
assign write_reserve_final = clr_item_ptr;
assign read = pop & valid_mask[read_ptr_addr];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_items <= 6'h0;
  end
  else if (flush) begin
    num_items <= 6'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (write_alloc & (~read)) begin
        num_items <= num_items + 6'h1;
      end
      else if ((~write_alloc) & read) begin
        num_items <= num_items - 6'h1;
      end
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reg_array <= 544'h0;
  end
  else if (flush) begin
    reg_array <= 544'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (write_fill) begin
        reg_array[write_ptr_addr] <= fill_data_in;
      end
      if (write_reserve) begin
        reg_array[next_0_valid_d1] <= data_in_packed;
      end
    end
  end
end
always_comb begin
  data_out = reg_array[read_ptr_addr];
end
always_comb begin
  valid = valid_mask[read_ptr_addr];
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_mask <= 32'h0;
  end
  else if (flush) begin
    valid_mask <= 32'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (write_fill) begin
        valid_mask[write_ptr_addr] <= 1'h1;
      end
      if (write_reserve_final) begin
        valid_mask[next_0_valid_d1] <= 1'h1;
      end
      if (read) begin
        valid_mask[read_ptr_addr] <= 1'h0;
      end
    end
  end
end
always_comb begin
  next_0_valid_high_found = 1'h0;
  next_0_valid_high = 5'h0;
  next_0_valid_high_done = 1'h0;
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h0) begin
      if (valid_mask[0] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h0;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1) begin
      if (valid_mask[1] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h2) begin
      if (valid_mask[2] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h2;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h3) begin
      if (valid_mask[3] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h3;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h4) begin
      if (valid_mask[4] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h4;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h5) begin
      if (valid_mask[5] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h5;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h6) begin
      if (valid_mask[6] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h6;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h7) begin
      if (valid_mask[7] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h7;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h8) begin
      if (valid_mask[8] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h8;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h9) begin
      if (valid_mask[9] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h9;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'hA) begin
      if (valid_mask[10] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'hA;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'hB) begin
      if (valid_mask[11] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'hB;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'hC) begin
      if (valid_mask[12] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'hC;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'hD) begin
      if (valid_mask[13] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'hD;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'hE) begin
      if (valid_mask[14] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'hE;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'hF) begin
      if (valid_mask[15] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'hF;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h10) begin
      if (valid_mask[16] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h10;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h11) begin
      if (valid_mask[17] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h11;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h12) begin
      if (valid_mask[18] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h12;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h13) begin
      if (valid_mask[19] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h13;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h14) begin
      if (valid_mask[20] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h14;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h15) begin
      if (valid_mask[21] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h15;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h16) begin
      if (valid_mask[22] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h16;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h17) begin
      if (valid_mask[23] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h17;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h18) begin
      if (valid_mask[24] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h18;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h19) begin
      if (valid_mask[25] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h19;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1A) begin
      if (valid_mask[26] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1A;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1B) begin
      if (valid_mask[27] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1B;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1C) begin
      if (valid_mask[28] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1C;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1D) begin
      if (valid_mask[29] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1D;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1E) begin
      if (valid_mask[30] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1E;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 5'h1F) begin
      if (valid_mask[31] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 5'h1F;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
end
always_comb begin
  next_0_valid_low_found = 1'h0;
  next_0_valid_low = 5'h0;
  next_0_valid_low_done = 1'h0;
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h0) begin
      if (valid_mask[0] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h0;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1) begin
      if (valid_mask[1] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h2) begin
      if (valid_mask[2] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h2;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h3) begin
      if (valid_mask[3] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h3;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h4) begin
      if (valid_mask[4] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h4;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h5) begin
      if (valid_mask[5] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h5;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h6) begin
      if (valid_mask[6] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h6;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h7) begin
      if (valid_mask[7] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h7;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h8) begin
      if (valid_mask[8] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h8;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h9) begin
      if (valid_mask[9] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h9;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'hA) begin
      if (valid_mask[10] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'hA;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'hB) begin
      if (valid_mask[11] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'hB;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'hC) begin
      if (valid_mask[12] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'hC;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'hD) begin
      if (valid_mask[13] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'hD;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'hE) begin
      if (valid_mask[14] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'hE;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'hF) begin
      if (valid_mask[15] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'hF;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h10) begin
      if (valid_mask[16] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h10;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h11) begin
      if (valid_mask[17] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h11;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h12) begin
      if (valid_mask[18] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h12;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h13) begin
      if (valid_mask[19] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h13;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h14) begin
      if (valid_mask[20] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h14;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h15) begin
      if (valid_mask[21] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h15;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h16) begin
      if (valid_mask[22] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h16;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h17) begin
      if (valid_mask[23] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h17;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h18) begin
      if (valid_mask[24] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h18;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h19) begin
      if (valid_mask[25] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h19;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1A) begin
      if (valid_mask[26] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1A;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1B) begin
      if (valid_mask[27] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1B;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1C) begin
      if (valid_mask[28] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1C;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1D) begin
      if (valid_mask[29] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1D;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1E) begin
      if (valid_mask[30] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1E;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 5'h1F) begin
      if (valid_mask[31] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 5'h1F;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
end
assign inc_reserve_count = write_alloc & (~write_fill);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reserve_count <= 16'h0;
  end
  else if (flush) begin
    reserve_count <= 16'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (inc_reserve_count) begin
        reserve_count <= reserve_count + 16'h1;
      end
    end
  end
end
endmodule   // reservation_fifo_depth_32_w_17_num_per_1

module reservation_fifo_depth_8_w_17_num_per_2 (
  input logic clk,
  input logic clk_en,
  input logic [16:0] data_in_0,
  input logic [16:0] data_in_1,
  input logic [16:0] fill_data_in,
  input logic flush,
  input logic pop,
  input logic push_alloc,
  input logic push_fill,
  input logic push_reserve,
  input logic rst_n,
  output logic [16:0] data_out_0,
  output logic [16:0] data_out_1,
  output logic empty,
  output logic full,
  output logic valid
);

logic clr_item_ptr;
logic clr_read_ptr;
logic clr_write_ptr;
logic [1:0][16:0] data_in_packed;
logic [1:0][16:0] data_out;
logic enable_reserve_ptr;
logic inc_item_ptr;
logic inc_read_ptr;
logic inc_reserve_count;
logic inc_write_ptr;
logic item_ptr_addr;
logic jump_next_0;
logic [2:0] next_0_valid;
logic [2:0] next_0_valid_d1;
logic [2:0] next_0_valid_high;
logic next_0_valid_high_done;
logic next_0_valid_high_found;
logic [2:0] next_0_valid_low;
logic next_0_valid_low_done;
logic next_0_valid_low_found;
logic [3:0] num_items;
logic read;
logic [2:0] read_ptr_addr;
logic [7:0][1:0][16:0] reg_array;
logic [15:0] reserve_count;
logic [2:0] reserve_ptr_val;
logic [7:0] valid_mask;
logic write_alloc;
logic write_fill;
logic [2:0] write_ptr_addr;
logic write_reserve;
logic write_reserve_final;
assign data_in_packed[0] = data_in_0;
assign data_in_packed[1] = data_in_1;
assign data_out_0 = data_out[0];
assign data_out_1 = data_out[1];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    item_ptr_addr <= 1'h0;
  end
  else if (flush) begin
    item_ptr_addr <= 1'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (clr_item_ptr) begin
        item_ptr_addr <= 1'h0;
      end
      else if (inc_item_ptr) begin
        item_ptr_addr <= item_ptr_addr + 1'h1;
      end
    end
  end
end
assign inc_item_ptr = push_reserve;
assign clr_item_ptr = push_reserve & (item_ptr_addr == 1'h1);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    read_ptr_addr <= 3'h0;
  end
  else if (flush) begin
    read_ptr_addr <= 3'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (clr_read_ptr) begin
        read_ptr_addr <= 3'h0;
      end
      else if (inc_read_ptr) begin
        read_ptr_addr <= read_ptr_addr + 3'h1;
      end
    end
  end
end
assign inc_read_ptr = read;
assign clr_read_ptr = 1'h0;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    write_ptr_addr <= 3'h0;
  end
  else if (flush) begin
    write_ptr_addr <= 3'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (clr_write_ptr) begin
        write_ptr_addr <= 3'h0;
      end
      else if (inc_write_ptr) begin
        write_ptr_addr <= write_ptr_addr + 3'h1;
      end
    end
  end
end
assign inc_write_ptr = write_alloc | write_fill;
assign clr_write_ptr = 1'h0;
assign jump_next_0 = next_0_valid_high_found | next_0_valid_low_found;
assign enable_reserve_ptr = write_reserve_final | (write_fill & (reserve_ptr_val == write_ptr_addr));

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    next_0_valid_d1 <= 3'h0;
  end
  else if (flush) begin
    next_0_valid_d1 <= 3'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (1'h0) begin
        next_0_valid_d1 <= 3'h0;
      end
      else if (enable_reserve_ptr) begin
        next_0_valid_d1 <= next_0_valid;
      end
    end
  end
end
assign reserve_ptr_val = next_0_valid_d1;
assign next_0_valid = (write_fill & ((next_0_valid_d1 == write_ptr_addr) | ((~next_0_valid_high_found)
    & (~next_0_valid_low_found)) | (next_0_valid_high_found ? next_0_valid_high ==
    write_ptr_addr: next_0_valid_low == write_ptr_addr))) ? write_ptr_addr + 3'h1:
    ((~next_0_valid_high_found) & (~next_0_valid_low_found)) ? write_ptr_addr:
    next_0_valid_high_found ? next_0_valid_high: next_0_valid_low;
assign full = num_items == 4'h8;
assign empty = num_items == 4'h0;
assign write_fill = push_fill & push_alloc & (~full);
assign write_alloc = push_alloc & (~full);
assign write_reserve = inc_item_ptr;
assign write_reserve_final = clr_item_ptr;
assign read = pop & valid_mask[read_ptr_addr];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_items <= 4'h0;
  end
  else if (flush) begin
    num_items <= 4'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (write_alloc & (~read)) begin
        num_items <= num_items + 4'h1;
      end
      else if ((~write_alloc) & read) begin
        num_items <= num_items - 4'h1;
      end
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reg_array <= 272'h0;
  end
  else if (flush) begin
    reg_array <= 272'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (write_fill) begin
        reg_array[write_ptr_addr][0] <= fill_data_in;
      end
      if (write_reserve) begin
        reg_array[next_0_valid_d1][item_ptr_addr] <= data_in_packed[item_ptr_addr];
      end
    end
  end
end
always_comb begin
  data_out = reg_array[read_ptr_addr];
end
always_comb begin
  valid = valid_mask[read_ptr_addr];
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_mask <= 8'h0;
  end
  else if (flush) begin
    valid_mask <= 8'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (write_fill) begin
        valid_mask[write_ptr_addr] <= 1'h1;
      end
      if (write_reserve_final) begin
        valid_mask[next_0_valid_d1] <= 1'h1;
      end
      if (read) begin
        valid_mask[read_ptr_addr] <= 1'h0;
      end
    end
  end
end
always_comb begin
  next_0_valid_high_found = 1'h0;
  next_0_valid_high = 3'h0;
  next_0_valid_high_done = 1'h0;
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h0) begin
      if (valid_mask[0] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h0;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h1) begin
      if (valid_mask[1] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h1;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h2) begin
      if (valid_mask[2] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h2;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h3) begin
      if (valid_mask[3] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h3;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h4) begin
      if (valid_mask[4] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h4;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h5) begin
      if (valid_mask[5] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h5;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h6) begin
      if (valid_mask[6] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h6;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_high_done) begin
    if (next_0_valid_d1 < 3'h7) begin
      if (valid_mask[7] == 1'h0) begin
        next_0_valid_high_found = 1'h1;
        next_0_valid_high = 3'h7;
        next_0_valid_high_done = 1'h1;
      end
    end
  end
end
always_comb begin
  next_0_valid_low_found = 1'h0;
  next_0_valid_low = 3'h0;
  next_0_valid_low_done = 1'h0;
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h0) begin
      if (valid_mask[0] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h0;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h1) begin
      if (valid_mask[1] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h1;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h2) begin
      if (valid_mask[2] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h2;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h3) begin
      if (valid_mask[3] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h3;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h4) begin
      if (valid_mask[4] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h4;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h5) begin
      if (valid_mask[5] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h5;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h6) begin
      if (valid_mask[6] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h6;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
  if (~next_0_valid_low_done) begin
    if (next_0_valid_d1 > 3'h7) begin
      if (valid_mask[7] == 1'h0) begin
        next_0_valid_low_found = 1'h1;
        next_0_valid_low = 3'h7;
        next_0_valid_low_done = 1'h1;
      end
    end
  end
end
assign inc_reserve_count = write_alloc & (~write_fill);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    reserve_count <= 16'h0;
  end
  else if (flush) begin
    reserve_count <= 16'h0;
  end
  else if (clk_en) begin
    if (clk_en) begin
      if (inc_reserve_count) begin
        reserve_count <= reserve_count + 16'h1;
      end
    end
  end
end
endmodule   // reservation_fifo_depth_8_w_17_num_per_2

module scanner_pipe (
  input logic addr_out_0_ready,
  input logic addr_out_1_ready,
  input logic block_mode,
  input logic block_rd_out_ready,
  input logic clk,
  input logic clk_en,
  input logic coord_out_ready,
  input logic dense,
  input logic [15:0] dim_size,
  input logic do_repeat,
  input logic flush,
  input logic [15:0] inner_dim_offset,
  input logic lookup,
  input logic op_out_0_ready,
  input logic op_out_1_ready,
  input logic output_matrix_fully_accumulated,
  input logic output_row_fully_accumulated,
  input logic pos_out_ready,
  input logic [16:0] pos_to_read_scanner_from_vr_fsm,
  input logic [0:0] [16:0] rd_rsp_data_in_0,
  input logic rd_rsp_data_in_0_valid,
  input logic [0:0] [16:0] rd_rsp_data_in_1,
  input logic rd_rsp_data_in_1_valid,
  input logic [15:0] repeat_factor,
  input logic repeat_outer_inner_n,
  input logic root,
  input logic rst_n,
  input logic tile_en,
  input logic [16:0] us_pos_in,
  input logic us_pos_in_valid,
  input logic vector_reduce_mode,
  input logic vr_fsm_state_init_blank_DONE,
  input logic vr_fsm_state_init_blank_S0,
  output logic [0:0] [16:0] addr_out_0,
  output logic addr_out_0_valid,
  output logic [0:0] [16:0] addr_out_1,
  output logic addr_out_1_valid,
  output logic [16:0] block_rd_out,
  output logic block_rd_out_valid,
  output logic [16:0] coord_out,
  output logic coord_out_valid,
  output logic [0:0] [16:0] op_out_0,
  output logic op_out_0_valid,
  output logic [0:0] [16:0] op_out_1,
  output logic op_out_1_valid,
  output logic [16:0] pos_out,
  output logic pos_out_valid,
  output logic rd_rsp_data_in_0_ready,
  output logic rd_rsp_data_in_1_ready,
  output logic rs_has_prepped_ds_row,
  output logic us_pos_in_ready
);

typedef enum logic[3:0] {
  BLOCK_1_RD = 4'h0,
  BLOCK_1_SIZE_REC = 4'h1,
  BLOCK_1_SIZE_REQ = 4'h2,
  BLOCK_2_RD = 4'h3,
  BLOCK_2_SIZE_REC = 4'h4,
  BLOCK_2_SIZE_REQ = 4'h5,
  DENSE_STRM = 4'h6,
  DONE_CRD = 4'h7,
  FREE_CRD = 4'h8,
  FREE_CRD2 = 4'h9,
  PASS_DONE_CRD = 4'hA,
  READOUT_SYNC_LOCK = 4'hB,
  SEQ_STRM = 4'hC,
  START_CRD = 4'hD
} scan_seq_crd_state;
typedef enum logic[3:0] {
  DONE_SEG = 4'h0,
  FREE_SEG = 4'h1,
  INJECT_0 = 4'h2,
  INJECT_DONE = 4'h3,
  INJECT_ROUTING = 4'h4,
  LOOKUP = 4'h5,
  PASS_DONE_SEG = 4'h6,
  PASS_STOP_SEG = 4'h7,
  READ = 4'h8,
  READ_ALT = 4'h9,
  START_SEG = 4'hA
} scan_seq_seg_state;
logic [15:0] READS_MADE;
logic [15:0] READS_REC_CRD_READ;
logic [16:0] S_level_0;
logic [0:0][16:0] addr_out_fifo_0_data_in;
logic addr_out_fifo_0_empty;
logic addr_out_fifo_0_full;
logic addr_out_fifo_0_push;
logic [0:0][16:0] addr_out_fifo_1_data_in;
logic addr_out_fifo_1_empty;
logic addr_out_fifo_1_full;
logic addr_out_fifo_1_push;
logic [0:0][15:0] addr_out_to_fifo_0;
logic [0:0][15:0] addr_out_to_fifo_1;
logic [1:0] base_rr_0;
logic [1:0] base_rr_1;
logic block_rd_fifo_empty;
logic block_rd_fifo_full;
logic block_rd_fifo_push;
logic clr_fiber_addr;
logic clr_final_pushed_done;
logic clr_pop_infifo_sticky;
logic clr_pushed_done_crd;
logic clr_pushed_done_seg;
logic clr_readout_loop_crd;
logic clr_readout_loop_seg;
logic clr_rep;
logic clr_req_made_crd;
logic clr_req_made_seg;
logic clr_req_rec_crd;
logic clr_req_rec_seg;
logic clr_seen_root_eos;
logic clr_used_data;
logic [16:0] coord_fifo_in_packed;
logic [16:0] coord_fifo_out_packed;
logic coord_fifo_push;
logic coordinate_fifo_empty;
logic coordinate_fifo_full;
logic [0:0][15:0] crd_ID_out_to_fifo;
logic [0:0][15:0] crd_addr_out_to_fifo;
logic crd_grant_push_0;
logic crd_grant_push_1;
logic crd_in_done_state;
logic [0:0][15:0] crd_op_out_to_fifo;
logic [16:0] crd_out_to_fifo;
logic crd_pop_infifo;
logic crd_rd_rsp_fifo_pop;
logic crd_req_push;
logic [16:0] crd_res_fifo_data_in_0;
logic [16:0] crd_res_fifo_data_out;
logic [16:0] crd_res_fifo_fill_data_in;
logic crd_res_fifo_full;
logic crd_res_fifo_pop;
logic crd_res_fifo_push_alloc;
logic crd_res_fifo_push_alloc_0;
logic crd_res_fifo_push_fill;
logic crd_res_fifo_push_fill_0;
logic crd_res_fifo_push_reserve_0;
logic crd_res_fifo_valid;
logic done_in;
logic [16:0] done_token;
logic en_reg_data_in;
logic eos_in;
logic [15:0] fiber_addr;
logic [15:0] fiber_addr_pre;
logic [15:0] fiber_addr_pre_d1;
logic [15:0] fiber_addr_pre_d1_d1;
logic fifo_full;
logic [1:0] fifo_full_pre;
logic [16:0] fifo_out_us_packed;
logic fifo_us_full;
logic [16:0] fifo_us_in_packed;
logic final_pushed_done_sticky_sticky;
logic final_pushed_done_sticky_was_high;
logic gclk;
logic inc_fiber_addr;
logic inc_rep;
logic inc_req_made_crd;
logic inc_req_made_seg;
logic inc_req_rec_crd;
logic inc_req_rec_seg;
logic inc_requests_REC_CRD_READ;
logic inc_requests_made_CRDDD_READ;
logic infifo_eos_in;
logic [15:0] infifo_pos_in;
logic [15:0] infifo_pos_in_d1;
logic infifo_pos_in_d1_en;
logic infifo_valid_in;
logic [0:0][16:0] input_fifo_data_out;
logic input_fifo_empty;
logic iter_finish_sticky;
logic iter_finish_was_high;
logic [16:0] last_stop_token;
logic last_valid_accepting;
logic maybe_in;
logic [15:0] next_seq_addr;
logic [15:0] next_seq_length;
logic no_outfifo_full_0;
logic no_outfifo_full_1;
logic non_vr_coord_fifo_push;
logic non_vr_pos_out_fifo_push;
logic [15:0] num_reps;
logic [15:0] num_req_made_crd;
logic [15:0] num_req_made_seg;
logic [15:0] num_req_rec_crd;
logic [15:0] num_req_rec_seg;
logic [0:0][16:0] op_out_fifo_0_data_in;
logic op_out_fifo_0_empty;
logic op_out_fifo_0_full;
logic op_out_fifo_0_push;
logic [0:0][16:0] op_out_fifo_1_data_in;
logic op_out_fifo_1_empty;
logic op_out_fifo_1_full;
logic op_out_fifo_1_push;
logic [0:0][15:0] op_out_to_fifo_0;
logic [0:0][15:0] op_out_to_fifo_1;
logic [15:0] payload_ptr;
logic pop_infifo;
logic pop_infifo_sticky_sticky;
logic pop_infifo_sticky_was_high;
logic [15:0] pos_addr;
logic pos_fifo_empty;
logic pos_fifo_full;
logic [16:0] pos_fifo_in_packed;
logic [16:0] pos_fifo_out_packed;
logic pos_out_fifo_push;
logic [16:0] pos_out_to_fifo;
logic pos_out_valid_mux_sel;
logic [15:0] ptr_in;
logic [15:0] ptr_in_d1;
logic ptr_reg_en;
logic pushed_done_sticky_sticky;
logic pushed_done_sticky_was_high;
logic rd_rsp_fifo_0_empty;
logic rd_rsp_fifo_0_full;
logic [16:0] rd_rsp_fifo_0_out_data;
logic [16:0] rd_rsp_fifo_0_out_data_d1;
logic rd_rsp_fifo_0_valid;
logic rd_rsp_fifo_1_empty;
logic rd_rsp_fifo_1_full;
logic [16:0] rd_rsp_fifo_1_out_data;
logic rd_rsp_fifo_1_valid;
logic readout_dst_crd;
logic readout_dst_seg;
logic readout_loop_sticky_sticky;
logic readout_loop_sticky_was_high;
logic rep_finish_sticky;
logic rep_finish_was_high;
logic [1:0] rr_arbiter_0_grant_out;
logic [1:0] rr_arbiter_1_grant_out;
logic rs_has_prepped_ds_row_sticky_sticky;
logic rs_has_prepped_ds_row_sticky_was_high;
scan_seq_crd_state scan_seq_crd_current_state;
scan_seq_crd_state scan_seq_crd_next_state;
scan_seq_seg_state scan_seq_seg_current_state;
scan_seq_seg_state scan_seq_seg_next_state;
logic seen_root_eos_sticky;
logic seen_root_eos_was_high;
logic [0:0][15:0] seg_ID_out_to_fifo;
logic [0:0][15:0] seg_addr_out_to_fifo;
logic seg_grant_push_0;
logic seg_grant_push_1;
logic seg_in_done_state;
logic seg_in_start_state;
logic [0:0][15:0] seg_op_out_to_fifo;
logic seg_pop_infifo;
logic seg_rd_rsp_fifo_pop;
logic seg_req_push;
logic [16:0] seg_res_fifo_data_out_0;
logic [16:0] seg_res_fifo_data_out_1;
logic seg_res_fifo_done_out;
logic [16:0] seg_res_fifo_fill_data_in;
logic seg_res_fifo_full;
logic seg_res_fifo_pop;
logic seg_res_fifo_pop_0;
logic seg_res_fifo_push_alloc;
logic seg_res_fifo_push_alloc_0;
logic seg_res_fifo_push_fill;
logic seg_res_fifo_push_fill_0;
logic seg_res_fifo_push_reserve_0;
logic seg_res_fifo_valid;
logic [15:0] seq_addr;
logic [15:0] seq_length;
logic [15:0] seq_length_ptr_math;
logic set_final_pushed_done;
logic set_pushed_done_crd;
logic set_pushed_done_seg;
logic set_readout_loop_crd;
logic set_readout_loop_seg;
logic update_seq_state;
logic [15:0] us_fifo_inject_data;
logic us_fifo_inject_eos;
logic us_fifo_inject_push;
logic us_fifo_push;
logic use_data_sticky_sticky;
logic use_data_sticky_was_high;
logic [15:0] valid_cnt;
logic valid_inc;
logic valid_rst;
assign gclk = clk & tile_en;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    fiber_addr_pre <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      fiber_addr_pre <= 16'h0;
    end
    else if (clr_fiber_addr) begin
      fiber_addr_pre <= 16'h0;
    end
    else if (inc_fiber_addr) begin
      fiber_addr_pre <= fiber_addr_pre + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_reps <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      num_reps <= 16'h0;
    end
    else if (clr_rep) begin
      num_reps <= 16'h0;
    end
    else if (inc_rep) begin
      num_reps <= num_reps + 16'h1;
    end
  end
end
assign fifo_us_in_packed[16] = root ? us_fifo_inject_eos: us_pos_in[16];
assign fifo_us_in_packed[15:0] = root ? us_fifo_inject_data: us_pos_in[15:0];
assign us_fifo_push = root ? us_fifo_inject_push: us_pos_in_valid;
assign infifo_eos_in = fifo_out_us_packed[16];
assign infifo_pos_in = fifo_out_us_packed[15:0];
assign pop_infifo = seg_pop_infifo | crd_pop_infifo;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pop_infifo_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pop_infifo_sticky_was_high <= 1'h0;
    end
    else if (clr_pop_infifo_sticky) begin
      pop_infifo_sticky_was_high <= 1'h0;
    end
    else if (pop_infifo) begin
      pop_infifo_sticky_was_high <= 1'h1;
    end
  end
end
assign pop_infifo_sticky_sticky = pop_infifo_sticky_was_high;
assign fifo_out_us_packed = input_fifo_data_out;
assign us_pos_in_ready = ~fifo_us_full;
assign infifo_valid_in = ~input_fifo_empty;
assign rd_rsp_data_in_0_ready = ~rd_rsp_fifo_0_full;
assign rd_rsp_data_in_1_ready = ~rd_rsp_fifo_1_full;
assign rd_rsp_fifo_0_valid = ~rd_rsp_fifo_0_empty;
assign rd_rsp_fifo_1_valid = ~rd_rsp_fifo_1_empty;
assign no_outfifo_full_0 = ~(op_out_fifo_0_full | addr_out_fifo_0_full);
assign no_outfifo_full_1 = ~(op_out_fifo_1_full | addr_out_fifo_1_full);
assign base_rr_0 = {crd_req_push & (16'h0 == crd_ID_out_to_fifo), seg_req_push & (16'h0 ==
    seg_ID_out_to_fifo)};
assign base_rr_1 = {crd_req_push & (16'h1 == crd_ID_out_to_fifo), seg_req_push & (16'h1 ==
    seg_ID_out_to_fifo)};
assign {crd_grant_push_0, seg_grant_push_0} = rr_arbiter_0_grant_out;
assign {crd_grant_push_1, seg_grant_push_1} = rr_arbiter_1_grant_out;
assign addr_out_to_fifo_0 = crd_grant_push_0 ? crd_addr_out_to_fifo: seg_addr_out_to_fifo;
assign addr_out_to_fifo_1 = crd_grant_push_1 ? crd_addr_out_to_fifo: seg_addr_out_to_fifo;
assign op_out_to_fifo_0 = crd_grant_push_0 ? crd_op_out_to_fifo: seg_op_out_to_fifo;
assign op_out_to_fifo_1 = crd_grant_push_1 ? crd_op_out_to_fifo: seg_op_out_to_fifo;
assign addr_out_fifo_0_push = seg_grant_push_0 | crd_grant_push_0;
assign addr_out_fifo_1_push = seg_grant_push_1 | crd_grant_push_1;
assign op_out_fifo_0_push = seg_grant_push_0 | crd_grant_push_0;
assign op_out_fifo_1_push = seg_grant_push_1 | crd_grant_push_1;
assign addr_out_fifo_0_data_in = {1'h0, addr_out_to_fifo_0};
assign addr_out_fifo_1_data_in = {1'h0, addr_out_to_fifo_1};
assign addr_out_0_valid = ~addr_out_fifo_0_empty;
assign addr_out_1_valid = ~addr_out_fifo_1_empty;
assign op_out_fifo_0_data_in = {1'h0, op_out_to_fifo_0};
assign op_out_fifo_1_data_in = {1'h0, op_out_to_fifo_1};
assign op_out_0_valid = ~op_out_fifo_0_empty;
assign op_out_1_valid = ~op_out_fifo_1_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    pushed_done_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      pushed_done_sticky_was_high <= 1'h0;
    end
    else if (clr_pushed_done_seg | clr_pushed_done_crd) begin
      pushed_done_sticky_was_high <= 1'h0;
    end
    else if (set_pushed_done_seg | set_pushed_done_crd) begin
      pushed_done_sticky_was_high <= 1'h1;
    end
  end
end
assign pushed_done_sticky_sticky = pushed_done_sticky_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    readout_loop_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      readout_loop_sticky_was_high <= 1'h0;
    end
    else if (clr_readout_loop_seg | clr_readout_loop_crd) begin
      readout_loop_sticky_was_high <= 1'h0;
    end
    else if (set_readout_loop_seg | set_readout_loop_crd) begin
      readout_loop_sticky_was_high <= 1'h1;
    end
  end
end
assign readout_loop_sticky_sticky = readout_loop_sticky_was_high;
assign seg_res_fifo_push_alloc_0 = seg_res_fifo_push_alloc & (~lookup);
assign seg_res_fifo_push_reserve_0 = rd_rsp_fifo_0_valid & (rd_rsp_fifo_0_out_data[16] == 1'h0) & (~block_mode) &
    (~lookup);
assign seg_res_fifo_push_fill_0 = seg_res_fifo_push_fill & (~lookup);
assign seg_res_fifo_pop_0 = seg_res_fifo_pop & (~lookup);
assign crd_res_fifo_data_in_0 = lookup ? {1'h0, rd_rsp_fifo_0_out_data[15:0]}: block_mode ? rd_rsp_fifo_0_valid
    ? {1'h0, rd_rsp_fifo_0_out_data[15:0]}: {1'h0, rd_rsp_fifo_1_out_data[15:0]}:
    {1'h0, rd_rsp_fifo_1_out_data[15:0]};
assign crd_res_fifo_fill_data_in = lookup ? seg_res_fifo_fill_data_in: dense ? crd_out_to_fifo:
    seg_res_fifo_data_out_0;
assign crd_res_fifo_push_alloc_0 = lookup ? seg_res_fifo_push_alloc: crd_res_fifo_push_alloc;
assign crd_res_fifo_push_reserve_0 = rd_rsp_fifo_1_valid | (rd_rsp_fifo_0_valid & (block_mode | lookup));
assign crd_res_fifo_push_fill_0 = lookup ? seg_res_fifo_push_fill: crd_res_fifo_push_fill;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    last_stop_token <= 17'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      last_stop_token <= 17'h0;
    end
    else if (1'h0) begin
      last_stop_token <= 17'h0;
    end
    else if (seg_in_start_state ? 1'h0: seg_res_fifo_push_fill & seg_res_fifo_push_alloc & (lookup ? ~crd_res_fifo_full: ~seg_res_fifo_full) & seg_res_fifo_fill_data_in[16] & (seg_res_fifo_fill_data_in[9:8] == 2'h0)) begin
      last_stop_token <= seg_in_start_state ? input_fifo_data_out + 17'h1: seg_res_fifo_fill_data_in;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    use_data_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      use_data_sticky_was_high <= 1'h0;
    end
    else if (clr_used_data) begin
      use_data_sticky_was_high <= 1'h0;
    end
    else if (infifo_valid_in & (~infifo_eos_in)) begin
      use_data_sticky_was_high <= 1'h1;
    end
  end
end
assign use_data_sticky_sticky = use_data_sticky_was_high;
assign clr_used_data = 1'h0;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_cnt <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_cnt <= 16'h0;
    end
    else if (valid_rst) begin
      valid_cnt <= 16'h0;
    end
    else if (valid_inc) begin
      valid_cnt <= valid_cnt + 16'h1;
    end
  end
end
assign ptr_in = block_mode ? rd_rsp_fifo_0_valid ? rd_rsp_fifo_0_out_data[15:0]:
    rd_rsp_fifo_1_out_data[15:0]: rd_rsp_fifo_0_out_data[15:0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    ptr_in_d1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      ptr_in_d1 <= 16'h0;
    end
    else if (1'h0) begin
      ptr_in_d1 <= 16'h0;
    end
    else if (ptr_reg_en) begin
      ptr_in_d1 <= ptr_in;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    fiber_addr_pre_d1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      fiber_addr_pre_d1 <= 16'h0;
    end
    else if (1'h0) begin
      fiber_addr_pre_d1 <= 16'h0;
    end
    else if (1'h1) begin
      fiber_addr_pre_d1 <= fiber_addr_pre;
    end
  end
end
assign seq_length_ptr_math = seg_res_fifo_data_out_1 - seg_res_fifo_data_out_0[15:0];
assign pos_addr = root ? 16'h0: infifo_pos_in;
assign next_seq_addr = ptr_in_d1 + inner_dim_offset;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    seq_length <= 16'h0;
    seq_addr <= 16'h0;
    payload_ptr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      seq_length <= 16'h0;
      seq_addr <= 16'h0;
      payload_ptr <= 16'h0;
    end
    else if (update_seq_state) begin
      seq_length <= next_seq_length;
      seq_addr <= next_seq_addr;
      payload_ptr <= ptr_in_d1;
    end
  end
end
assign fiber_addr = fiber_addr_pre + seq_addr;
assign fifo_full = |fifo_full_pre;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    iter_finish_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      iter_finish_was_high <= 1'h0;
    end
    else if (clr_fiber_addr) begin
      iter_finish_was_high <= 1'h0;
    end
    else if (last_valid_accepting) begin
      iter_finish_was_high <= 1'h1;
    end
  end
end
assign iter_finish_sticky = last_valid_accepting | iter_finish_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rep_finish_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      rep_finish_was_high <= 1'h0;
    end
    else if (clr_rep) begin
      rep_finish_was_high <= 1'h0;
    end
    else if (((repeat_factor - 16'h1) == num_reps) & inc_rep) begin
      rep_finish_was_high <= 1'h1;
    end
  end
end
assign rep_finish_sticky = (((repeat_factor - 16'h1) == num_reps) & inc_rep) | rep_finish_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    seen_root_eos_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      seen_root_eos_was_high <= 1'h0;
    end
    else if (clr_seen_root_eos) begin
      seen_root_eos_was_high <= 1'h0;
    end
    else if (infifo_eos_in & (infifo_pos_in == 16'h0)) begin
      seen_root_eos_was_high <= 1'h1;
    end
  end
end
assign seen_root_eos_sticky = (infifo_eos_in & (infifo_pos_in == 16'h0)) | seen_root_eos_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rd_rsp_fifo_0_out_data_d1 <= 17'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      rd_rsp_fifo_0_out_data_d1 <= 17'h0;
    end
    else if (1'h0) begin
      rd_rsp_fifo_0_out_data_d1 <= 17'h0;
    end
    else if (en_reg_data_in) begin
      rd_rsp_fifo_0_out_data_d1 <= rd_rsp_fifo_0_out_data;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    fiber_addr_pre_d1_d1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      fiber_addr_pre_d1_d1 <= 16'h0;
    end
    else if (1'h0) begin
      fiber_addr_pre_d1_d1 <= 16'h0;
    end
    else if (en_reg_data_in) begin
      fiber_addr_pre_d1_d1 <= fiber_addr_pre_d1;
    end
  end
end
assign done_in = infifo_eos_in & infifo_valid_in & (infifo_pos_in[9:8] == 2'h1);
assign eos_in = infifo_eos_in & infifo_valid_in & (infifo_pos_in[9:8] == 2'h0);
assign maybe_in = infifo_eos_in & infifo_valid_in & (infifo_pos_in[9:8] == 2'h2);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    infifo_pos_in_d1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      infifo_pos_in_d1 <= 16'h0;
    end
    else if (1'h0) begin
      infifo_pos_in_d1 <= 16'h0;
    end
    else if (infifo_pos_in_d1_en) begin
      infifo_pos_in_d1 <= infifo_pos_in;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_req_made_seg <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      num_req_made_seg <= 16'h0;
    end
    else if (clr_req_made_seg) begin
      num_req_made_seg <= 16'h0;
    end
    else if (inc_req_made_seg) begin
      num_req_made_seg <= num_req_made_seg + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_req_rec_seg <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      num_req_rec_seg <= 16'h0;
    end
    else if (clr_req_rec_seg) begin
      num_req_rec_seg <= 16'h0;
    end
    else if (inc_req_rec_seg) begin
      num_req_rec_seg <= num_req_rec_seg + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_req_made_crd <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      num_req_made_crd <= 16'h0;
    end
    else if (clr_req_made_crd) begin
      num_req_made_crd <= 16'h0;
    end
    else if (inc_req_made_crd) begin
      num_req_made_crd <= num_req_made_crd + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    num_req_rec_crd <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      num_req_rec_crd <= 16'h0;
    end
    else if (clr_req_rec_crd) begin
      num_req_rec_crd <= 16'h0;
    end
    else if (inc_req_rec_crd) begin
      num_req_rec_crd <= num_req_rec_crd + 16'h1;
    end
  end
end
assign seg_res_fifo_done_out = seg_res_fifo_valid & seg_res_fifo_data_out_0[16] & (seg_res_fifo_data_out_0[9:8]
    == 2'h1);
assign readout_dst_crd = readout_loop_sticky_sticky;
assign readout_dst_seg = readout_loop_sticky_sticky;
assign inc_requests_made_CRDDD_READ = |{crd_grant_push_0, crd_grant_push_1};

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    READS_MADE <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      READS_MADE <= 16'h0;
    end
    else if (inc_requests_made_CRDDD_READ) begin
      READS_MADE <= READS_MADE + 16'h1;
    end
  end
end
assign inc_requests_REC_CRD_READ = rd_rsp_fifo_1_valid & (rd_rsp_fifo_1_out_data[16] == 1'h1);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    READS_REC_CRD_READ <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      READS_REC_CRD_READ <= 16'h0;
    end
    else if (inc_requests_REC_CRD_READ) begin
      READS_REC_CRD_READ <= READS_REC_CRD_READ + 16'h1;
    end
  end
end
assign coord_fifo_in_packed = crd_res_fifo_data_out;
assign S_level_0 = {1'h1, 16'h0};
assign done_token = {1'h1, 7'h0, 1'h1, 8'h0};
assign coord_out[16] = vr_fsm_state_init_blank_DONE ? done_token[16]: vr_fsm_state_init_blank_S0 ?
    S_level_0[16]: coord_fifo_out_packed[16];
assign coord_out[15:0] = vr_fsm_state_init_blank_DONE ? done_token[15:0]: vr_fsm_state_init_blank_S0 ?
    S_level_0[15:0]: coord_fifo_out_packed[15:0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    final_pushed_done_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      final_pushed_done_sticky_was_high <= 1'h0;
    end
    else if (clr_final_pushed_done) begin
      final_pushed_done_sticky_was_high <= 1'h0;
    end
    else if (set_final_pushed_done) begin
      final_pushed_done_sticky_was_high <= 1'h1;
    end
  end
end
assign final_pushed_done_sticky_sticky = final_pushed_done_sticky_was_high;
assign set_final_pushed_done = 1'h0;
assign clr_final_pushed_done = 1'h0;
assign non_vr_coord_fifo_push = crd_res_fifo_valid & (~block_mode) & (~(crd_res_fifo_data_out[16] &
    crd_res_fifo_valid & (crd_res_fifo_data_out[9:8] == 2'h3)));
assign coord_fifo_push = vector_reduce_mode ? non_vr_coord_fifo_push & (~output_row_fully_accumulated):
    non_vr_coord_fifo_push;
assign coord_out_valid = vr_fsm_state_init_blank_S0 | vr_fsm_state_init_blank_DONE |
    (~coordinate_fifo_empty);
assign fifo_full_pre[0] = coordinate_fifo_full;
assign crd_res_fifo_pop = (vector_reduce_mode & output_row_fully_accumulated) ? ~pos_fifo_full: block_mode
    ? ~block_rd_fifo_full: ~coordinate_fifo_full;
assign pos_fifo_in_packed = vector_reduce_mode ? coord_fifo_in_packed: pos_out_to_fifo;
assign pos_out_fifo_push = vector_reduce_mode ? non_vr_coord_fifo_push & output_row_fully_accumulated:
    non_vr_pos_out_fifo_push;
assign pos_out[16] = pos_fifo_out_packed[16];
assign pos_out[15:0] = pos_fifo_out_packed[15:0];
assign pos_out_valid_mux_sel = vector_reduce_mode & (pos_out == done_token);
assign pos_out_valid = ((~pos_out_valid_mux_sel) & (~pos_fifo_empty)) | (pos_out_valid_mux_sel &
    output_matrix_fully_accumulated & (~pos_fifo_empty));
assign fifo_full_pre[1] = pos_fifo_full;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rs_has_prepped_ds_row_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      rs_has_prepped_ds_row_sticky_was_high <= 1'h0;
    end
    else if (vr_fsm_state_init_blank_S0) begin
      rs_has_prepped_ds_row_sticky_was_high <= 1'h0;
    end
    else if ((pos_fifo_in_packed == done_token) & pos_out_fifo_push & (~fifo_full_pre[1])) begin
      rs_has_prepped_ds_row_sticky_was_high <= 1'h1;
    end
  end
end
assign rs_has_prepped_ds_row_sticky_sticky = ((pos_fifo_in_packed == done_token) & pos_out_fifo_push & (~fifo_full_pre[1])) |
    rs_has_prepped_ds_row_sticky_was_high;
assign rs_has_prepped_ds_row = rs_has_prepped_ds_row_sticky_sticky;
assign block_rd_fifo_push = crd_res_fifo_valid & block_mode & (~(crd_res_fifo_data_out[16] &
    crd_res_fifo_valid & (crd_res_fifo_data_out[9:8] == 2'h3)));
assign block_rd_out_valid = ~block_rd_fifo_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    scan_seq_crd_current_state <= START_CRD;
  end
  else if (clk_en) begin
    if (flush) begin
      scan_seq_crd_current_state <= START_CRD;
    end
    else scan_seq_crd_current_state <= scan_seq_crd_next_state;
  end
end
always_comb begin
  scan_seq_crd_next_state = scan_seq_crd_current_state;
  unique case (scan_seq_crd_current_state)
    BLOCK_1_RD: begin
        if ((num_req_rec_crd == ptr_in_d1) & (~lookup)) begin
          scan_seq_crd_next_state = BLOCK_2_SIZE_REQ;
        end
        else if ((num_req_rec_crd == ptr_in_d1) & lookup) begin
          scan_seq_crd_next_state = FREE_CRD;
        end
        else scan_seq_crd_next_state = BLOCK_1_RD;
      end
    BLOCK_1_SIZE_REC: begin
        if (rd_rsp_fifo_0_valid) begin
          scan_seq_crd_next_state = BLOCK_1_RD;
        end
        else scan_seq_crd_next_state = BLOCK_1_SIZE_REC;
      end
    BLOCK_1_SIZE_REQ: begin
        if (|{crd_grant_push_0, crd_grant_push_1}) begin
          scan_seq_crd_next_state = BLOCK_1_SIZE_REC;
        end
        else scan_seq_crd_next_state = BLOCK_1_SIZE_REQ;
      end
    BLOCK_2_RD: begin
        if (num_req_rec_crd == ptr_in_d1) begin
          scan_seq_crd_next_state = FREE_CRD;
        end
        else scan_seq_crd_next_state = BLOCK_2_RD;
      end
    BLOCK_2_SIZE_REC: begin
        if (rd_rsp_fifo_1_valid) begin
          scan_seq_crd_next_state = BLOCK_2_RD;
        end
        else scan_seq_crd_next_state = BLOCK_2_SIZE_REC;
      end
    BLOCK_2_SIZE_REQ: begin
        if (|{crd_grant_push_0, crd_grant_push_1}) begin
          scan_seq_crd_next_state = BLOCK_2_SIZE_REC;
        end
        else scan_seq_crd_next_state = BLOCK_2_SIZE_REQ;
      end
    DENSE_STRM: scan_seq_crd_next_state = DENSE_STRM;
    DONE_CRD: scan_seq_crd_next_state = START_CRD;
    FREE_CRD: begin
        if ((|{crd_grant_push_0, crd_grant_push_1}) & block_mode & (~lookup)) begin
          scan_seq_crd_next_state = FREE_CRD2;
        end
        else if (|{crd_grant_push_0, crd_grant_push_1}) begin
          scan_seq_crd_next_state = DONE_CRD;
        end
        else scan_seq_crd_next_state = FREE_CRD;
      end
    FREE_CRD2: begin
        if (|{crd_grant_push_0, crd_grant_push_1}) begin
          scan_seq_crd_next_state = DONE_CRD;
        end
        else scan_seq_crd_next_state = FREE_CRD2;
      end
    PASS_DONE_CRD: begin
        if ((~crd_res_fifo_full) & (~pos_fifo_full)) begin
          scan_seq_crd_next_state = DONE_CRD;
        end
      end
    READOUT_SYNC_LOCK: scan_seq_crd_next_state = DONE_CRD;
    SEQ_STRM: begin
        if (seg_res_fifo_done_out & (~crd_res_fifo_full) & (~pos_fifo_full)) begin
          scan_seq_crd_next_state = FREE_CRD;
        end
      end
    START_CRD: begin
        if (block_mode & tile_en) begin
          scan_seq_crd_next_state = BLOCK_1_SIZE_REQ;
        end
        else if (dense & (~lookup) & tile_en) begin
          scan_seq_crd_next_state = DENSE_STRM;
        end
        else if ((~dense) & (~lookup) & tile_en) begin
          scan_seq_crd_next_state = SEQ_STRM;
        end
      end
    default: scan_seq_crd_next_state = scan_seq_crd_current_state;
  endcase
end
always_comb begin
  unique case (scan_seq_crd_current_state)
    BLOCK_1_RD: begin :scan_seq_crd_BLOCK_1_RD_Output
        crd_addr_out_to_fifo = num_req_made_crd;
        crd_op_out_to_fifo = 16'h1;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = (num_req_made_crd < ptr_in_d1) & (~crd_res_fifo_full);
        crd_rd_rsp_fifo_pop = num_req_rec_crd < ptr_in_d1;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = (num_req_made_crd < ptr_in_d1) & (|{crd_grant_push_0, crd_grant_push_1}) &
            (~crd_res_fifo_full);
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = (num_req_rec_crd < ptr_in_d1) & rd_rsp_fifo_0_valid;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = (num_req_made_crd < ptr_in_d1) & (|{crd_grant_push_0, crd_grant_push_1}) &
            (~crd_res_fifo_full);
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_BLOCK_1_RD_Output
    BLOCK_1_SIZE_REC: begin :scan_seq_crd_BLOCK_1_SIZE_REC_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h1;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h1;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = rd_rsp_fifo_0_valid;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_BLOCK_1_SIZE_REC_Output
    BLOCK_1_SIZE_REQ: begin :scan_seq_crd_BLOCK_1_SIZE_REQ_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h2;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = ~crd_res_fifo_full;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = (~crd_res_fifo_full) & (|{crd_grant_push_0, crd_grant_push_1});
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_BLOCK_1_SIZE_REQ_Output
    BLOCK_2_RD: begin :scan_seq_crd_BLOCK_2_RD_Output
        crd_addr_out_to_fifo = num_req_made_crd;
        crd_op_out_to_fifo = 16'h1;
        crd_ID_out_to_fifo = 16'h1;
        crd_req_push = (num_req_made_crd < ptr_in_d1) & (~crd_res_fifo_full);
        crd_rd_rsp_fifo_pop = num_req_rec_crd < ptr_in_d1;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = (num_req_made_crd < ptr_in_d1) & (|{crd_grant_push_0, crd_grant_push_1}) &
            (~crd_res_fifo_full);
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = (num_req_rec_crd < ptr_in_d1) & rd_rsp_fifo_1_valid;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = (num_req_made_crd < ptr_in_d1) & (|{crd_grant_push_0, crd_grant_push_1}) &
            (~crd_res_fifo_full);
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_BLOCK_2_RD_Output
    BLOCK_2_SIZE_REC: begin :scan_seq_crd_BLOCK_2_SIZE_REC_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h1;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h1;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h1;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = rd_rsp_fifo_1_valid;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_BLOCK_2_SIZE_REC_Output
    BLOCK_2_SIZE_REQ: begin :scan_seq_crd_BLOCK_2_SIZE_REQ_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h2;
        crd_ID_out_to_fifo = 16'h1;
        crd_req_push = ~crd_res_fifo_full;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = (~crd_res_fifo_full) & (|{crd_grant_push_0, crd_grant_push_1});
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_BLOCK_2_SIZE_REQ_Output
    DENSE_STRM: begin :scan_seq_crd_DENSE_STRM_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = seg_res_fifo_valid & (~pos_fifo_full) & (~crd_res_fifo_full) &
            (seg_res_fifo_data_out_0[16] ? 1'h1: dim_size > num_req_made_crd);
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = seg_res_fifo_data_out_0[16] ? seg_res_fifo_data_out_0:
            17'((seg_res_fifo_data_out_0[15:0] * dim_size) + num_req_made_crd);
        crd_out_to_fifo = seg_res_fifo_data_out_0[16] ? seg_res_fifo_data_out_0: 17'(num_req_made_crd);
        inc_req_made_crd = seg_res_fifo_valid & (dim_size > num_req_made_crd) &
            (~seg_res_fifo_data_out_0[16]) & (~pos_fifo_full) & (~crd_res_fifo_full);
        clr_req_made_crd = seg_res_fifo_valid & seg_res_fifo_data_out_0[16];
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = seg_res_fifo_valid & (~pos_fifo_full) & (~crd_res_fifo_full) &
            (seg_res_fifo_data_out_0[16] ? 1'h1: dim_size > num_req_made_crd);
        crd_res_fifo_push_fill = seg_res_fifo_valid & (~pos_fifo_full) & (~crd_res_fifo_full) &
            (seg_res_fifo_data_out_0[16] ? 1'h1: dim_size > num_req_made_crd);
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = seg_res_fifo_valid & (~pos_fifo_full) & (~crd_res_fifo_full) &
            (seg_res_fifo_data_out_0[16] ? 1'h1: ((dim_size - 16'h1) == num_req_made_crd) &
            inc_req_made_crd);
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_DENSE_STRM_Output
    DONE_CRD: begin :scan_seq_crd_DONE_CRD_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h1;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h1;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        crd_in_done_state = 1'h1;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_DONE_CRD_Output
    FREE_CRD: begin :scan_seq_crd_FREE_CRD_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = block_mode ? 16'h0: 16'h1;
        crd_req_push = 1'h1;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h1;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h1;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_FREE_CRD_Output
    FREE_CRD2: begin :scan_seq_crd_FREE_CRD2_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h1;
        crd_req_push = 1'h1;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h1;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h1;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_FREE_CRD2_Output
    PASS_DONE_CRD: begin :scan_seq_crd_PASS_DONE_CRD_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = (~pos_fifo_full) & (~crd_res_fifo_full) & seg_res_fifo_done_out &
            seg_res_fifo_valid;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = {1'h1, 6'h0, 2'h1, 8'h0};
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = (~pos_fifo_full) & (~crd_res_fifo_full) & seg_res_fifo_done_out &
            seg_res_fifo_valid;
        crd_res_fifo_push_fill = (~pos_fifo_full) & (~crd_res_fifo_full) & seg_res_fifo_done_out &
            seg_res_fifo_valid;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = (~pos_fifo_full) & (~crd_res_fifo_full) & seg_res_fifo_done_out &
            seg_res_fifo_valid;
        clr_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        crd_in_done_state = 1'h0;
      end :scan_seq_crd_PASS_DONE_CRD_Output
    READOUT_SYNC_LOCK: begin :scan_seq_crd_READOUT_SYNC_LOCK_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        clr_pushed_done_crd = 1'h1;
        clr_readout_loop_crd = 1'h1;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        crd_in_done_state = 1'h0;
      end :scan_seq_crd_READOUT_SYNC_LOCK_Output
    SEQ_STRM: begin :scan_seq_crd_SEQ_STRM_Output
        crd_addr_out_to_fifo = num_req_made_crd + seg_res_fifo_data_out_0[15:0];
        crd_op_out_to_fifo = 16'h1;
        crd_ID_out_to_fifo = 16'h1;
        crd_req_push = seg_res_fifo_valid & (~seg_res_fifo_data_out_0[16]) & (~crd_res_fifo_full) &
            (num_req_made_crd < seq_length_ptr_math) & (~pos_fifo_full);
        crd_rd_rsp_fifo_pop = 1'h1;
        non_vr_pos_out_fifo_push = seg_res_fifo_data_out_0[16] ? (~pos_fifo_full) & (~crd_res_fifo_full) &
            seg_res_fifo_valid: (|{crd_grant_push_0, crd_grant_push_1}) & (num_req_made_crd
            < seq_length_ptr_math) & (~pos_fifo_full) & (~crd_res_fifo_full) &
            seg_res_fifo_valid;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = seg_res_fifo_data_out_0[16] ? seg_res_fifo_data_out_0: {1'h0, num_req_made_crd +
            seg_res_fifo_data_out_0[15:0]};
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = (|{crd_grant_push_0, crd_grant_push_1}) & (num_req_made_crd <
            seq_length_ptr_math) & (~pos_fifo_full) & (~crd_res_fifo_full) &
            seg_res_fifo_valid;
        clr_req_made_crd = (((|{crd_grant_push_0, crd_grant_push_1}) & ((seq_length_ptr_math - 16'h1) ==
            num_req_made_crd)) | (seq_length_ptr_math == 16'h0)) & (~pos_fifo_full) &
            (~crd_res_fifo_full) & seg_res_fifo_valid;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = seg_res_fifo_data_out_0[16] ? (~pos_fifo_full) & (~crd_res_fifo_full) &
            seg_res_fifo_valid: (|{crd_grant_push_0, crd_grant_push_1}) & (num_req_made_crd
            < seq_length_ptr_math) & (~pos_fifo_full) & (~crd_res_fifo_full) &
            seg_res_fifo_valid;
        crd_res_fifo_push_fill = seg_res_fifo_valid & seg_res_fifo_data_out_0[16] & (~pos_fifo_full) &
            (~crd_res_fifo_full);
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = clr_req_made_crd | (seg_res_fifo_valid & seg_res_fifo_data_out_0[16] &
            (~pos_fifo_full) & (~crd_res_fifo_full));
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_SEQ_STRM_Output
    START_CRD: begin :scan_seq_crd_START_CRD_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_START_CRD_Output
    default: begin :scan_seq_crd_default_Output
        crd_addr_out_to_fifo = 16'h0;
        crd_op_out_to_fifo = 16'h0;
        crd_ID_out_to_fifo = 16'h0;
        crd_req_push = 1'h0;
        crd_rd_rsp_fifo_pop = 1'h0;
        non_vr_pos_out_fifo_push = 1'h0;
        crd_pop_infifo = 1'h0;
        en_reg_data_in = 1'h0;
        pos_out_to_fifo = 17'h0;
        crd_out_to_fifo = 17'h0;
        inc_req_made_crd = 1'h0;
        clr_req_made_crd = 1'h0;
        inc_req_rec_crd = 1'h0;
        clr_req_rec_crd = 1'h0;
        crd_res_fifo_push_alloc = 1'h0;
        crd_res_fifo_push_fill = 1'h0;
        ptr_reg_en = 1'h0;
        seg_res_fifo_pop = 1'h0;
        set_readout_loop_crd = 1'h0;
        set_pushed_done_crd = 1'h0;
        clr_readout_loop_crd = 1'h0;
        crd_in_done_state = 1'h0;
        clr_pushed_done_crd = 1'h0;
      end :scan_seq_crd_default_Output
  endcase
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    scan_seq_seg_current_state <= START_SEG;
  end
  else if (clk_en) begin
    if (flush) begin
      scan_seq_seg_current_state <= START_SEG;
    end
    else scan_seq_seg_current_state <= scan_seq_seg_next_state;
  end
end
always_comb begin
  scan_seq_seg_next_state = scan_seq_seg_current_state;
  unique case (scan_seq_seg_current_state)
    DONE_SEG: begin
        if (lookup ? 1'h1: (~dense) & crd_in_done_state) begin
          scan_seq_seg_next_state = START_SEG;
        end
      end
    FREE_SEG: begin
        if (|{seg_grant_push_0, seg_grant_push_1}) begin
          scan_seq_seg_next_state = DONE_SEG;
        end
      end
    INJECT_0: scan_seq_seg_next_state = INJECT_DONE;
    INJECT_DONE: scan_seq_seg_next_state = READ;
    INJECT_ROUTING: begin
        if (~seg_res_fifo_full) begin
          scan_seq_seg_next_state = READ;
        end
      end
    LOOKUP: begin
        if (done_in & (~crd_res_fifo_full)) begin
          scan_seq_seg_next_state = FREE_SEG;
        end
        else scan_seq_seg_next_state = LOOKUP;
      end
    PASS_DONE_SEG: scan_seq_seg_next_state = FREE_SEG;
    PASS_STOP_SEG: begin
        if (readout_loop_sticky_sticky & (~seg_res_fifo_full)) begin
          scan_seq_seg_next_state = PASS_DONE_SEG;
        end
        else if (infifo_valid_in & (~lookup) & (~seg_res_fifo_full)) begin
          scan_seq_seg_next_state = READ;
        end
        else scan_seq_seg_next_state = PASS_STOP_SEG;
      end
    READ: begin
        if (maybe_in | (dense & (~done_in) & (~seg_res_fifo_full) & infifo_valid_in) | ((~dense) & eos_in)) begin
          scan_seq_seg_next_state = PASS_STOP_SEG;
        end
        else if ((|{seg_grant_push_0, seg_grant_push_1}) & (~seg_res_fifo_full)) begin
          scan_seq_seg_next_state = READ_ALT;
        end
        else if (done_in & (~seg_res_fifo_full)) begin
          scan_seq_seg_next_state = FREE_SEG;
        end
        else scan_seq_seg_next_state = READ;
      end
    READ_ALT: begin
        if ((|{seg_grant_push_0, seg_grant_push_1}) & (~seg_res_fifo_full) & infifo_valid_in) begin
          scan_seq_seg_next_state = READ;
        end
        else if ((|{seg_grant_push_0, seg_grant_push_1}) & (~seg_res_fifo_full)) begin
          scan_seq_seg_next_state = PASS_STOP_SEG;
        end
        else scan_seq_seg_next_state = READ_ALT;
      end
    START_SEG: begin
        if (block_mode) begin
          scan_seq_seg_next_state = START_SEG;
        end
        else if ((~root) & (~lookup) & (~block_mode) & tile_en) begin
          scan_seq_seg_next_state = READ;
        end
        else if (root & (~lookup) & (~block_mode) & tile_en) begin
          scan_seq_seg_next_state = INJECT_0;
        end
        else if ((~root) & lookup & (~block_mode) & tile_en) begin
          scan_seq_seg_next_state = LOOKUP;
        end
      end
    default: scan_seq_seg_next_state = scan_seq_seg_current_state;
  endcase
end
always_comb begin
  unique case (scan_seq_seg_current_state)
    DONE_SEG: begin :scan_seq_seg_DONE_SEG_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = 1'h0;
        seg_res_fifo_push_fill = 1'h0;
        seg_res_fifo_fill_data_in = 17'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
        seg_in_done_state = 1'h1;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_start_state = 1'h0;
      end :scan_seq_seg_DONE_SEG_Output
    FREE_SEG: begin :scan_seq_seg_FREE_SEG_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h1;
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = 1'h0;
        seg_res_fifo_push_fill = 1'h0;
        seg_res_fifo_fill_data_in = 17'h0;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_FREE_SEG_Output
    INJECT_0: begin :scan_seq_seg_INJECT_0_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h0;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h1;
        seg_res_fifo_push_alloc = 1'h0;
        seg_res_fifo_push_fill = 1'h0;
        seg_res_fifo_fill_data_in = 17'h0;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_INJECT_0_Output
    INJECT_DONE: begin :scan_seq_seg_INJECT_DONE_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h0;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h100;
        us_fifo_inject_eos = 1'h1;
        us_fifo_inject_push = 1'h1;
        seg_res_fifo_push_alloc = 1'h0;
        seg_res_fifo_push_fill = 1'h0;
        seg_res_fifo_fill_data_in = 17'h0;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_INJECT_DONE_Output
    INJECT_ROUTING: begin :scan_seq_seg_INJECT_ROUTING_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h0;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h1;
        seg_res_fifo_push_alloc = ~seg_res_fifo_full;
        seg_res_fifo_push_fill = ~seg_res_fifo_full;
        seg_res_fifo_fill_data_in = {1'h1, 6'h0, 2'h3, 7'h0, readout_loop_sticky_sticky};
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_INJECT_ROUTING_Output
    LOOKUP: begin :scan_seq_seg_LOOKUP_Output
        seg_addr_out_to_fifo = infifo_pos_in;
        seg_op_out_to_fifo = 16'h1;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = infifo_valid_in & (~infifo_eos_in) & (~crd_res_fifo_full);
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = infifo_valid_in & (~crd_res_fifo_full) & (infifo_eos_in ? 1'h1:
            |{seg_grant_push_0, seg_grant_push_1});
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h1;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h1;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = (~crd_res_fifo_full) & ((|{seg_grant_push_0, seg_grant_push_1}) |
            (infifo_valid_in & infifo_eos_in));
        seg_res_fifo_push_fill = infifo_valid_in & infifo_eos_in & (~crd_res_fifo_full);
        seg_res_fifo_fill_data_in = (infifo_eos_in & (infifo_pos_in[9:8] == 2'h2)) ? 17'h0: {infifo_eos_in,
            infifo_pos_in};
        set_pushed_done_seg = 1'h0;
        set_readout_loop_seg = 1'h0;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_LOOKUP_Output
    PASS_DONE_SEG: begin :scan_seq_seg_PASS_DONE_SEG_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = (~readout_loop_sticky_sticky) & done_in & (~seg_res_fifo_full);
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = readout_loop_sticky_sticky & pushed_done_sticky_sticky & (~seg_res_fifo_full);
        seg_res_fifo_push_fill = readout_loop_sticky_sticky & pushed_done_sticky_sticky & (~seg_res_fifo_full);
        seg_res_fifo_fill_data_in = {1'h1, 6'h0, 2'h1, 8'h0};
        set_pushed_done_seg = 1'h0;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_PASS_DONE_SEG_Output
    PASS_STOP_SEG: begin :scan_seq_seg_PASS_STOP_SEG_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = (~seg_res_fifo_full) & eos_in & (~readout_loop_sticky_sticky);
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h1;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h1;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = (infifo_valid_in | readout_loop_sticky_sticky) & (~seg_res_fifo_full);
        seg_res_fifo_push_fill = (infifo_valid_in | readout_loop_sticky_sticky) & (~seg_res_fifo_full);
        seg_res_fifo_fill_data_in = readout_loop_sticky_sticky ? last_stop_token - 17'h1: eos_in ? {1'h1,
            infifo_pos_in + 16'h1}: {1'h1, 16'h0};
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_PASS_STOP_SEG_Output
    READ: begin :scan_seq_seg_READ_Output
        seg_addr_out_to_fifo = readout_loop_sticky_sticky ? 16'h0: infifo_pos_in;
        seg_op_out_to_fifo = 16'h1;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = ((infifo_valid_in & (~infifo_eos_in) & (~dense)) | readout_loop_sticky_sticky) &
            (~seg_res_fifo_full);
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = (((done_in | (infifo_valid_in & (~eos_in) & (dense | (|{seg_grant_push_0,
            seg_grant_push_1})))) & (~seg_res_fifo_full)) | maybe_in) &
            (~readout_loop_sticky_sticky);
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = (done_in | dense) ? (~seg_res_fifo_full) & infifo_valid_in & (~eos_in):
            (~seg_res_fifo_full) & (|{seg_grant_push_0, seg_grant_push_1}) & (~maybe_in);
        seg_res_fifo_push_fill = (done_in | (dense & infifo_valid_in & (~eos_in))) & (~seg_res_fifo_full);
        seg_res_fifo_fill_data_in = {infifo_eos_in, infifo_pos_in};
        infifo_pos_in_d1_en = (|{seg_grant_push_0, seg_grant_push_1}) & (~seg_res_fifo_full);
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_READ_Output
    READ_ALT: begin :scan_seq_seg_READ_ALT_Output
        seg_addr_out_to_fifo = readout_loop_sticky_sticky ? 16'h1: infifo_pos_in_d1 + 16'h1;
        seg_op_out_to_fifo = 16'h1;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = ~seg_res_fifo_full;
        seg_rd_rsp_fifo_pop = 1'h1;
        seg_pop_infifo = eos_in & (~done_in) & (|{seg_grant_push_0, seg_grant_push_1}) &
            (~seg_res_fifo_full) & (~readout_loop_sticky_sticky);
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = infifo_valid_in & (|{seg_grant_push_0, seg_grant_push_1}) & (~seg_res_fifo_full)
            & (~readout_loop_sticky_sticky);
        seg_res_fifo_push_fill = infifo_valid_in & (|{seg_grant_push_0, seg_grant_push_1}) & (~seg_res_fifo_full)
            & (~readout_loop_sticky_sticky);
        seg_res_fifo_fill_data_in = eos_in ? {1'h1, infifo_pos_in + 16'h1}: {1'h1, 16'h0};
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        seg_in_start_state = 1'h0;
        set_readout_loop_seg = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_READ_ALT_Output
    START_SEG: begin :scan_seq_seg_START_SEG_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h0;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = 1'h0;
        seg_res_fifo_push_fill = 1'h0;
        seg_res_fifo_fill_data_in = 17'h0;
        set_readout_loop_seg = 1'h0;
        seg_in_start_state = 1'h1;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_START_SEG_Output
    default: begin :scan_seq_seg_default_Output
        seg_addr_out_to_fifo = 16'h0;
        seg_op_out_to_fifo = 16'h0;
        seg_ID_out_to_fifo = 16'h0;
        seg_req_push = 1'h0;
        seg_rd_rsp_fifo_pop = 1'h0;
        seg_pop_infifo = 1'h0;
        inc_req_made_seg = 1'h0;
        clr_req_made_seg = 1'h0;
        inc_req_rec_seg = 1'h0;
        clr_req_rec_seg = 1'h0;
        us_fifo_inject_data = 16'h0;
        us_fifo_inject_eos = 1'h0;
        us_fifo_inject_push = 1'h0;
        seg_res_fifo_push_alloc = 1'h0;
        seg_res_fifo_push_fill = 1'h0;
        seg_res_fifo_fill_data_in = 17'h0;
        set_readout_loop_seg = 1'h0;
        seg_in_start_state = 1'h1;
        infifo_pos_in_d1_en = 1'h0;
        clr_pushed_done_seg = 1'h0;
        set_pushed_done_seg = 1'h0;
        seg_in_done_state = 1'h0;
        clr_readout_loop_seg = 1'h0;
      end :scan_seq_seg_default_Output
  endcase
end
reg_fifo_depth_2_w_17_afd_2 input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(fifo_us_in_packed),
  .flush(flush),
  .pop(pop_infifo),
  .push(us_fifo_push),
  .rst_n(rst_n),
  .data_out(input_fifo_data_out),
  .empty(input_fifo_empty),
  .full(fifo_us_full)
);

reg_fifo_depth_2_w_17_afd_2 rd_rsp_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_rsp_data_in_0),
  .flush(flush),
  .pop(1'h1),
  .push(rd_rsp_data_in_0_valid),
  .rst_n(rst_n),
  .data_out(rd_rsp_fifo_0_out_data),
  .empty(rd_rsp_fifo_0_empty),
  .full(rd_rsp_fifo_0_full)
);

reg_fifo_depth_2_w_17_afd_2 rd_rsp_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(rd_rsp_data_in_1),
  .flush(flush),
  .pop(1'h1),
  .push(rd_rsp_data_in_1_valid),
  .rst_n(rst_n),
  .data_out(rd_rsp_fifo_1_out_data),
  .empty(rd_rsp_fifo_1_empty),
  .full(rd_rsp_fifo_1_full)
);

arbiter_2_in_PRIO_algo rr_arbiter_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .flush(flush),
  .request_in(base_rr_0),
  .resource_ready(no_outfifo_full_0),
  .rst_n(rst_n),
  .grant_out(rr_arbiter_0_grant_out)
);

arbiter_2_in_PRIO_algo rr_arbiter_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .flush(flush),
  .request_in(base_rr_1),
  .resource_ready(no_outfifo_full_1),
  .rst_n(rst_n),
  .grant_out(rr_arbiter_1_grant_out)
);

reg_fifo_depth_2_w_17_afd_2 addr_out_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(addr_out_fifo_0_data_in),
  .flush(flush),
  .pop(addr_out_0_ready),
  .push(addr_out_fifo_0_push),
  .rst_n(rst_n),
  .data_out(addr_out_0),
  .empty(addr_out_fifo_0_empty),
  .full(addr_out_fifo_0_full)
);

reg_fifo_depth_2_w_17_afd_2 addr_out_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(addr_out_fifo_1_data_in),
  .flush(flush),
  .pop(addr_out_1_ready),
  .push(addr_out_fifo_1_push),
  .rst_n(rst_n),
  .data_out(addr_out_1),
  .empty(addr_out_fifo_1_empty),
  .full(addr_out_fifo_1_full)
);

reg_fifo_depth_2_w_17_afd_2 op_out_fifo_0 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(op_out_fifo_0_data_in),
  .flush(flush),
  .pop(op_out_0_ready),
  .push(op_out_fifo_0_push),
  .rst_n(rst_n),
  .data_out(op_out_0),
  .empty(op_out_fifo_0_empty),
  .full(op_out_fifo_0_full)
);

reg_fifo_depth_2_w_17_afd_2 op_out_fifo_1 (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(op_out_fifo_1_data_in),
  .flush(flush),
  .pop(op_out_1_ready),
  .push(op_out_fifo_1_push),
  .rst_n(rst_n),
  .data_out(op_out_1),
  .empty(op_out_fifo_1_empty),
  .full(op_out_fifo_1_full)
);

reservation_fifo_depth_8_w_17_num_per_2 seg_res_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in_0(rd_rsp_fifo_0_out_data),
  .data_in_1(rd_rsp_fifo_0_out_data),
  .fill_data_in(seg_res_fifo_fill_data_in),
  .flush(flush),
  .pop(seg_res_fifo_pop_0),
  .push_alloc(seg_res_fifo_push_alloc_0),
  .push_fill(seg_res_fifo_push_fill_0),
  .push_reserve(seg_res_fifo_push_reserve_0),
  .rst_n(rst_n),
  .data_out_0(seg_res_fifo_data_out_0),
  .data_out_1(seg_res_fifo_data_out_1),
  .full(seg_res_fifo_full),
  .valid(seg_res_fifo_valid)
);

reservation_fifo_depth_32_w_17_num_per_1 crd_res_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in_0(crd_res_fifo_data_in_0),
  .fill_data_in(crd_res_fifo_fill_data_in),
  .flush(flush),
  .pop(crd_res_fifo_pop),
  .push_alloc(crd_res_fifo_push_alloc_0),
  .push_fill(crd_res_fifo_push_fill_0),
  .push_reserve(crd_res_fifo_push_reserve_0),
  .rst_n(rst_n),
  .data_out_0(crd_res_fifo_data_out),
  .full(crd_res_fifo_full),
  .valid(crd_res_fifo_valid)
);

reg_fifo_depth_0_w_17_afd_2 coordinate_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(coord_fifo_in_packed),
  .flush(flush),
  .pop(coord_out_ready),
  .push(coord_fifo_push),
  .rst_n(rst_n),
  .data_out(coord_fifo_out_packed),
  .empty(coordinate_fifo_empty),
  .full(coordinate_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 pos_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(pos_fifo_in_packed),
  .flush(flush),
  .pop(pos_out_ready),
  .push(pos_out_fifo_push),
  .rst_n(rst_n),
  .data_out(pos_fifo_out_packed),
  .empty(pos_fifo_empty),
  .full(pos_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 block_rd_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(crd_res_fifo_data_out),
  .flush(flush),
  .pop(block_rd_out_ready),
  .push(block_rd_fifo_push),
  .rst_n(rst_n),
  .data_out(block_rd_out),
  .empty(block_rd_fifo_empty),
  .full(block_rd_fifo_full)
);

endmodule   // scanner_pipe

module sched_gen_3_16 (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic enable,
  input logic finished,
  input logic flush,
  input logic [1:0] mux_sel,
  input logic rst_n,
  input logic [15:0] sched_addr_gen_starting_addr,
  input logic [15:0] sched_addr_gen_strides_0,
  input logic [15:0] sched_addr_gen_strides_1,
  input logic [15:0] sched_addr_gen_strides_2,
  output logic valid_output
);

logic [15:0] addr_out;
logic [2:0][15:0] sched_addr_gen_strides;
logic valid_gate;
logic valid_gate_inv;
logic valid_out;
assign valid_gate = ~valid_gate_inv;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_gate_inv <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_gate_inv <= 1'h0;
    end
    else if (finished) begin
      valid_gate_inv <= 1'h1;
    end
  end
end
always_comb begin
  valid_out = (cycle_count == addr_out) & valid_gate & enable;
end
always_comb begin
  valid_output = valid_out;
end
assign sched_addr_gen_strides[0] = sched_addr_gen_strides_0;
assign sched_addr_gen_strides[1] = sched_addr_gen_strides_1;
assign sched_addr_gen_strides[2] = sched_addr_gen_strides_2;
addr_gen_3_16 sched_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(mux_sel),
  .restart(finished),
  .rst_n(rst_n),
  .starting_addr(sched_addr_gen_starting_addr),
  .step(valid_out),
  .strides(sched_addr_gen_strides),
  .addr_out(addr_out)
);

endmodule   // sched_gen_3_16

module sched_gen_6_16 (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic enable,
  input logic finished,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic rst_n,
  input logic [15:0] sched_addr_gen_starting_addr,
  input logic [15:0] sched_addr_gen_strides_0,
  input logic [15:0] sched_addr_gen_strides_1,
  input logic [15:0] sched_addr_gen_strides_2,
  input logic [15:0] sched_addr_gen_strides_3,
  input logic [15:0] sched_addr_gen_strides_4,
  input logic [15:0] sched_addr_gen_strides_5,
  output logic valid_output
);

logic [15:0] addr_out;
logic [5:0][15:0] sched_addr_gen_strides;
logic valid_gate;
logic valid_gate_inv;
logic valid_out;
assign valid_gate = ~valid_gate_inv;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_gate_inv <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_gate_inv <= 1'h0;
    end
    else if (finished) begin
      valid_gate_inv <= 1'h1;
    end
  end
end
always_comb begin
  valid_out = (cycle_count == addr_out) & valid_gate & enable;
end
always_comb begin
  valid_output = valid_out;
end
assign sched_addr_gen_strides[0] = sched_addr_gen_strides_0;
assign sched_addr_gen_strides[1] = sched_addr_gen_strides_1;
assign sched_addr_gen_strides[2] = sched_addr_gen_strides_2;
assign sched_addr_gen_strides[3] = sched_addr_gen_strides_3;
assign sched_addr_gen_strides[4] = sched_addr_gen_strides_4;
assign sched_addr_gen_strides[5] = sched_addr_gen_strides_5;
addr_gen_6_16 sched_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(mux_sel),
  .restart(finished),
  .rst_n(rst_n),
  .starting_addr(sched_addr_gen_starting_addr),
  .step(valid_out),
  .strides(sched_addr_gen_strides),
  .addr_out(addr_out)
);

endmodule   // sched_gen_6_16

module sched_gen_6_16_delay_addr_10_4 (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic enable,
  input logic finished,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic rst_n,
  input logic [9:0] sched_addr_gen_delay,
  input logic [15:0] sched_addr_gen_starting_addr,
  input logic [15:0] sched_addr_gen_strides_0,
  input logic [15:0] sched_addr_gen_strides_1,
  input logic [15:0] sched_addr_gen_strides_2,
  input logic [15:0] sched_addr_gen_strides_3,
  input logic [15:0] sched_addr_gen_strides_4,
  input logic [15:0] sched_addr_gen_strides_5,
  output logic delay_en_out,
  output logic valid_output,
  output logic valid_output_d
);

logic [3:0][10:0] addr_fifo;
logic addr_fifo_empty_n;
logic [10:0] addr_fifo_in;
logic [10:0] addr_fifo_out;
logic addr_fifo_wr_en;
logic [15:0] addr_out;
logic [15:0] addr_out_d;
logic delay_en;
logic [1:0] next_rd_ptr;
logic [1:0] rd_ptr;
logic [9:0] sched_addr_gen_delay_out;
logic [5:0][15:0] sched_addr_gen_strides;
logic valid_gate;
logic valid_gate_inv;
logic valid_out;
logic valid_out_d;
logic [1:0] wr_ptr;
assign valid_gate = ~valid_gate_inv;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_gate_inv <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_gate_inv <= 1'h0;
    end
    else if (finished) begin
      valid_gate_inv <= 1'h1;
    end
  end
end
assign delay_en_out = delay_en;
assign delay_en = sched_addr_gen_delay_out > 10'h0;
assign next_rd_ptr = rd_ptr + 2'h1;
assign addr_fifo_wr_en = valid_out;
assign addr_fifo_in = addr_out_d[10:0];
assign addr_fifo_out = addr_fifo[rd_ptr];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr <= 2'h0;
    rd_ptr <= 2'h0;
    addr_fifo <= 44'h0;
    addr_fifo_empty_n <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      wr_ptr <= 2'h0;
      rd_ptr <= 2'h0;
      addr_fifo <= 44'h0;
      addr_fifo_empty_n <= 1'h0;
    end
    else if (delay_en) begin
      if (addr_fifo_wr_en) begin
        wr_ptr <= wr_ptr + 2'h1;
        addr_fifo[wr_ptr] <= addr_fifo_in;
      end
      if (valid_out_d) begin
        rd_ptr <= next_rd_ptr;
      end
      if (addr_fifo_wr_en) begin
        addr_fifo_empty_n <= 1'h1;
      end
      else if (valid_out_d) begin
        addr_fifo_empty_n <= ~(next_rd_ptr == wr_ptr);
      end
      else addr_fifo_empty_n <= addr_fifo_empty_n;
    end
  end
end
always_comb begin
  valid_out_d = (cycle_count[10:0] == addr_fifo_out) & addr_fifo_empty_n & enable;
  valid_output_d = valid_out_d;
end
always_comb begin
  valid_out = (cycle_count == addr_out) & valid_gate & enable;
end
always_comb begin
  valid_output = valid_out;
end
assign sched_addr_gen_strides[0] = sched_addr_gen_strides_0;
assign sched_addr_gen_strides[1] = sched_addr_gen_strides_1;
assign sched_addr_gen_strides[2] = sched_addr_gen_strides_2;
assign sched_addr_gen_strides[3] = sched_addr_gen_strides_3;
assign sched_addr_gen_strides[4] = sched_addr_gen_strides_4;
assign sched_addr_gen_strides[5] = sched_addr_gen_strides_5;
addr_gen_6_16_delay_addr_10 sched_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .delay(sched_addr_gen_delay),
  .flush(flush),
  .mux_sel(mux_sel),
  .restart(finished),
  .rst_n(rst_n),
  .starting_addr(sched_addr_gen_starting_addr),
  .step(valid_out),
  .strides(sched_addr_gen_strides),
  .addr_out(addr_out),
  .delay_out(sched_addr_gen_delay_out),
  .delayed_addr_out(addr_out_d)
);

endmodule   // sched_gen_6_16_delay_addr_10_4

module sched_gen_6_16_delay_addr_10_8 (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic enable,
  input logic finished,
  input logic flush,
  input logic [2:0] mux_sel,
  input logic rst_n,
  input logic [9:0] sched_addr_gen_delay,
  input logic [15:0] sched_addr_gen_starting_addr,
  input logic [15:0] sched_addr_gen_strides_0,
  input logic [15:0] sched_addr_gen_strides_1,
  input logic [15:0] sched_addr_gen_strides_2,
  input logic [15:0] sched_addr_gen_strides_3,
  input logic [15:0] sched_addr_gen_strides_4,
  input logic [15:0] sched_addr_gen_strides_5,
  output logic delay_en_out,
  output logic valid_output,
  output logic valid_output_d
);

logic [7:0][10:0] addr_fifo;
logic addr_fifo_empty_n;
logic [10:0] addr_fifo_in;
logic [10:0] addr_fifo_out;
logic addr_fifo_wr_en;
logic [15:0] addr_out;
logic [15:0] addr_out_d;
logic delay_en;
logic [2:0] next_rd_ptr;
logic [2:0] rd_ptr;
logic [9:0] sched_addr_gen_delay_out;
logic [5:0][15:0] sched_addr_gen_strides;
logic valid_gate;
logic valid_gate_inv;
logic valid_out;
logic valid_out_d;
logic [2:0] wr_ptr;
assign valid_gate = ~valid_gate_inv;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_gate_inv <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_gate_inv <= 1'h0;
    end
    else if (finished) begin
      valid_gate_inv <= 1'h1;
    end
  end
end
assign delay_en_out = delay_en;
assign delay_en = sched_addr_gen_delay_out > 10'h0;
assign next_rd_ptr = rd_ptr + 3'h1;
assign addr_fifo_wr_en = valid_out;
assign addr_fifo_in = addr_out_d[10:0];
assign addr_fifo_out = addr_fifo[rd_ptr];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr <= 3'h0;
    rd_ptr <= 3'h0;
    addr_fifo <= 88'h0;
    addr_fifo_empty_n <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      wr_ptr <= 3'h0;
      rd_ptr <= 3'h0;
      addr_fifo <= 88'h0;
      addr_fifo_empty_n <= 1'h0;
    end
    else if (delay_en) begin
      if (addr_fifo_wr_en) begin
        wr_ptr <= wr_ptr + 3'h1;
        addr_fifo[wr_ptr] <= addr_fifo_in;
      end
      if (valid_out_d) begin
        rd_ptr <= next_rd_ptr;
      end
      if (addr_fifo_wr_en) begin
        addr_fifo_empty_n <= 1'h1;
      end
      else if (valid_out_d) begin
        addr_fifo_empty_n <= ~(next_rd_ptr == wr_ptr);
      end
      else addr_fifo_empty_n <= addr_fifo_empty_n;
    end
  end
end
always_comb begin
  valid_out_d = (cycle_count[10:0] == addr_fifo_out) & addr_fifo_empty_n & enable;
  valid_output_d = valid_out_d;
end
always_comb begin
  valid_out = (cycle_count == addr_out) & valid_gate & enable;
end
always_comb begin
  valid_output = valid_out;
end
assign sched_addr_gen_strides[0] = sched_addr_gen_strides_0;
assign sched_addr_gen_strides[1] = sched_addr_gen_strides_1;
assign sched_addr_gen_strides[2] = sched_addr_gen_strides_2;
assign sched_addr_gen_strides[3] = sched_addr_gen_strides_3;
assign sched_addr_gen_strides[4] = sched_addr_gen_strides_4;
assign sched_addr_gen_strides[5] = sched_addr_gen_strides_5;
addr_gen_6_16_delay_addr_10 sched_addr_gen (
  .clk(clk),
  .clk_en(clk_en),
  .delay(sched_addr_gen_delay),
  .flush(flush),
  .mux_sel(mux_sel),
  .restart(finished),
  .rst_n(rst_n),
  .starting_addr(sched_addr_gen_starting_addr),
  .step(valid_out),
  .strides(sched_addr_gen_strides),
  .addr_out(addr_out),
  .delay_out(sched_addr_gen_delay_out),
  .delayed_addr_out(addr_out_d)
);

endmodule   // sched_gen_6_16_delay_addr_10_8

module sram_sp__0 (
  input logic clk,
  input logic clk_en,
  input logic [63:0] data_in_p0,
  input logic flush,
  input logic [8:0] read_addr_p0,
  input logic read_enable_p0,
  input logic [8:0] write_addr_p0,
  input logic write_enable_p0,
  output logic [63:0] data_out_p0
);

logic [63:0] data_array [511:0];

always_ff @(posedge clk) begin
  if (clk_en) begin
    if (write_enable_p0 == 1'h1) begin
      data_array[write_addr_p0] <= data_in_p0;
    end
    else if (read_enable_p0) begin
      data_out_p0 <= data_array[read_addr_p0];
    end
  end
end
endmodule   // sram_sp__0

module stencil_valid (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [3:0] loops_stencil_valid_dimensionality,
  input logic [10:0] loops_stencil_valid_ranges_0,
  input logic [10:0] loops_stencil_valid_ranges_1,
  input logic [10:0] loops_stencil_valid_ranges_2,
  input logic [10:0] loops_stencil_valid_ranges_3,
  input logic [10:0] loops_stencil_valid_ranges_4,
  input logic [10:0] loops_stencil_valid_ranges_5,
  input logic rst_n,
  input logic stencil_valid_sched_gen_enable,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_starting_addr,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_strides_0,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_strides_1,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_strides_2,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_strides_3,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_strides_4,
  input logic [15:0] stencil_valid_sched_gen_sched_addr_gen_strides_5,
  output logic stencil_valid
);

logic [15:0] cycle_count;
logic flushed;
logic [2:0] loops_stencil_valid_mux_sel_out;
logic [5:0][10:0] loops_stencil_valid_ranges;
logic loops_stencil_valid_restart;
logic stencil_valid_internal;
assign stencil_valid = stencil_valid_internal & flushed;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    cycle_count <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      cycle_count <= 16'h0;
    end
    else if (flushed) begin
      cycle_count <= cycle_count + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    flushed <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      flushed <= 1'h1;
    end
  end
end
assign loops_stencil_valid_ranges[0] = loops_stencil_valid_ranges_0;
assign loops_stencil_valid_ranges[1] = loops_stencil_valid_ranges_1;
assign loops_stencil_valid_ranges[2] = loops_stencil_valid_ranges_2;
assign loops_stencil_valid_ranges[3] = loops_stencil_valid_ranges_3;
assign loops_stencil_valid_ranges[4] = loops_stencil_valid_ranges_4;
assign loops_stencil_valid_ranges[5] = loops_stencil_valid_ranges_5;
for_loop_6_11 loops_stencil_valid (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_stencil_valid_dimensionality),
  .flush(flush),
  .ranges(loops_stencil_valid_ranges),
  .rst_n(rst_n),
  .step(stencil_valid_internal),
  .mux_sel_out(loops_stencil_valid_mux_sel_out),
  .restart(loops_stencil_valid_restart)
);

sched_gen_6_16 stencil_valid_sched_gen (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(stencil_valid_sched_gen_enable),
  .finished(loops_stencil_valid_restart),
  .flush(flush),
  .mux_sel(loops_stencil_valid_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_starting_addr(stencil_valid_sched_gen_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(stencil_valid_sched_gen_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(stencil_valid_sched_gen_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(stencil_valid_sched_gen_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(stencil_valid_sched_gen_sched_addr_gen_strides_3),
  .sched_addr_gen_strides_4(stencil_valid_sched_gen_sched_addr_gen_strides_4),
  .sched_addr_gen_strides_5(stencil_valid_sched_gen_sched_addr_gen_strides_5),
  .valid_output(stencil_valid_internal)
);

endmodule   // stencil_valid

module stencil_valid_flat (
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic rst_n,
  input logic [3:0] stencil_valid_inst_loops_stencil_valid_dimensionality,
  input logic [10:0] stencil_valid_inst_loops_stencil_valid_ranges_0,
  input logic [10:0] stencil_valid_inst_loops_stencil_valid_ranges_1,
  input logic [10:0] stencil_valid_inst_loops_stencil_valid_ranges_2,
  input logic [10:0] stencil_valid_inst_loops_stencil_valid_ranges_3,
  input logic [10:0] stencil_valid_inst_loops_stencil_valid_ranges_4,
  input logic [10:0] stencil_valid_inst_loops_stencil_valid_ranges_5,
  input logic stencil_valid_inst_stencil_valid_sched_gen_enable,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_starting_addr,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_0,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_1,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_2,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_3,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_4,
  input logic [15:0] stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_5,
  output logic stencil_valid_f_
);

stencil_valid stencil_valid_inst (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .loops_stencil_valid_dimensionality(stencil_valid_inst_loops_stencil_valid_dimensionality),
  .loops_stencil_valid_ranges_0(stencil_valid_inst_loops_stencil_valid_ranges_0),
  .loops_stencil_valid_ranges_1(stencil_valid_inst_loops_stencil_valid_ranges_1),
  .loops_stencil_valid_ranges_2(stencil_valid_inst_loops_stencil_valid_ranges_2),
  .loops_stencil_valid_ranges_3(stencil_valid_inst_loops_stencil_valid_ranges_3),
  .loops_stencil_valid_ranges_4(stencil_valid_inst_loops_stencil_valid_ranges_4),
  .loops_stencil_valid_ranges_5(stencil_valid_inst_loops_stencil_valid_ranges_5),
  .rst_n(rst_n),
  .stencil_valid_sched_gen_enable(stencil_valid_inst_stencil_valid_sched_gen_enable),
  .stencil_valid_sched_gen_sched_addr_gen_starting_addr(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_starting_addr),
  .stencil_valid_sched_gen_sched_addr_gen_strides_0(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_0),
  .stencil_valid_sched_gen_sched_addr_gen_strides_1(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_1),
  .stencil_valid_sched_gen_sched_addr_gen_strides_2(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_2),
  .stencil_valid_sched_gen_sched_addr_gen_strides_3(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_3),
  .stencil_valid_sched_gen_sched_addr_gen_strides_4(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_4),
  .stencil_valid_sched_gen_sched_addr_gen_strides_5(stencil_valid_inst_stencil_valid_sched_gen_sched_addr_gen_strides_5),
  .stencil_valid(stencil_valid_f_)
);

endmodule   // stencil_valid_flat

module storage_config_seq_2_64_16 (
  input logic clk,
  input logic clk_en,
  input logic [7:0] config_addr_in,
  input logic [15:0] config_data_in,
  input logic [1:0] config_en,
  input logic config_rd,
  input logic config_wr,
  input logic flush,
  input logic [0:0][3:0] [15:0] rd_data_stg,
  input logic rst_n,
  output logic [8:0] addr_out,
  output logic [1:0] [15:0] rd_data_out,
  output logic ren_out,
  output logic wen_out,
  output logic [3:0] [15:0] wr_data
);

logic [1:0] cnt;
logic [2:0][15:0] data_wr_reg;
logic [1:0] rd_cnt;
logic rd_valid;
logic [1:0] reduce_en;
logic set_to_addr;
assign reduce_en[0] = |config_en[0];
assign reduce_en[1] = |config_en[1];
always_comb begin
  set_to_addr = 1'h0;
  for (int unsigned i = 0; i < 2; i += 1) begin
      if (reduce_en[1'(i)]) begin
        set_to_addr = 1'(i);
      end
    end
end
assign addr_out = {set_to_addr, config_addr_in};

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    cnt <= 2'h0;
  end
  else if (flush) begin
    cnt <= 2'h0;
  end
  else if (config_wr & (|config_en)) begin
    cnt <= cnt + 2'h1;
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rd_valid <= 1'h0;
  end
  else if (flush) begin
    rd_valid <= 1'h0;
  end
  else rd_valid <= config_rd & (|config_en);
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    rd_cnt <= 2'h0;
  end
  else if (flush) begin
    rd_cnt <= 2'h0;
  end
  else if (rd_valid & (~(config_rd & (|config_en)))) begin
    rd_cnt <= rd_cnt + 2'h1;
  end
end
assign rd_data_out[0] = rd_data_stg[0][rd_cnt];
assign rd_data_out[1] = rd_data_stg[0][rd_cnt];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    data_wr_reg <= 48'h0;
  end
  else if (flush) begin
    data_wr_reg <= 48'h0;
  end
  else if (config_wr & (cnt < 2'h3)) begin
    data_wr_reg[cnt] <= config_data_in;
  end
end
assign wr_data[0] = data_wr_reg[0];
assign wr_data[1] = data_wr_reg[1];
assign wr_data[2] = data_wr_reg[2];
assign wr_data[3] = config_data_in;
assign wen_out = config_wr & (cnt == 2'h3);
assign ren_out = config_rd;
endmodule   // storage_config_seq_2_64_16

module strg_ram_64_512_delay1 (
  input logic clk,
  input logic clk_en,
  input logic [0:0][3:0] [15:0] data_from_strg,
  input logic [16:0] data_in,
  input logic flush,
  input logic [16:0] rd_addr_in,
  input logic ren,
  input logic rst_n,
  input logic wen,
  input logic [16:0] wr_addr_in,
  output logic [0:0] [8:0] addr_out,
  output logic [16:0] data_out,
  output logic [0:0][3:0] [15:0] data_to_strg,
  output logic ready,
  output logic ren_to_strg,
  output logic valid_out,
  output logic wen_to_strg
);

typedef enum logic[1:0] {
  IDLE = 2'h0,
  MODIFY = 2'h1,
  READ = 2'h2,
  _DEFAULT = 2'h3
} r_w_seq_state;
logic [15:0] addr_to_write;
logic [3:0][15:0] data_combined;
logic [15:0] data_to_write;
r_w_seq_state r_w_seq_current_state;
r_w_seq_state r_w_seq_next_state;
logic [15:0] rd_addr;
logic rd_bank;
logic read_gate;
logic [15:0] wr_addr;
logic write_gate;
assign wr_addr = wr_addr_in[15:0];
assign rd_addr = wr_addr_in[15:0];
assign rd_bank = 1'h0;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    data_to_write <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      data_to_write <= 16'h0;
    end
    else data_to_write <= data_in[15:0];
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    addr_to_write <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      addr_to_write <= 16'h0;
    end
    else addr_to_write <= wr_addr;
  end
end
assign data_to_strg[0] = data_combined;
assign ren_to_strg = (wen | ren) & read_gate;
assign wen_to_strg = write_gate;
always_comb begin
  addr_out[0] = rd_addr[10:2];
  if (wen & (~write_gate)) begin
    addr_out[0] = wr_addr[10:2];
  end
  else if (write_gate) begin
    addr_out[0] = addr_to_write[10:2];
  end
end
always_comb begin
  if (addr_to_write[1:0] == 2'h0) begin
    data_combined[0] = data_to_write;
  end
  else data_combined[0] = data_from_strg[rd_bank][0];
end
always_comb begin
  if (addr_to_write[1:0] == 2'h1) begin
    data_combined[1] = data_to_write;
  end
  else data_combined[1] = data_from_strg[rd_bank][1];
end
always_comb begin
  if (addr_to_write[1:0] == 2'h2) begin
    data_combined[2] = data_to_write;
  end
  else data_combined[2] = data_from_strg[rd_bank][2];
end
always_comb begin
  if (addr_to_write[1:0] == 2'h3) begin
    data_combined[3] = data_to_write;
  end
  else data_combined[3] = data_from_strg[rd_bank][3];
end

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    r_w_seq_current_state <= IDLE;
  end
  else r_w_seq_current_state <= r_w_seq_next_state;
end
always_comb begin
  r_w_seq_next_state = r_w_seq_current_state;
  unique case (r_w_seq_current_state)
    IDLE: begin
        if ((~wen) & (~ren)) begin
          r_w_seq_next_state = IDLE;
        end
        else if (ren & (~wen)) begin
          r_w_seq_next_state = READ;
        end
        else if (wen) begin
          r_w_seq_next_state = MODIFY;
        end
      end
    MODIFY: begin
        if (1'h1) begin
          r_w_seq_next_state = IDLE;
        end
      end
    READ: begin
        if ((~wen) & (~ren)) begin
          r_w_seq_next_state = IDLE;
        end
        else if (ren & (~wen)) begin
          r_w_seq_next_state = READ;
        end
        else if (wen) begin
          r_w_seq_next_state = MODIFY;
        end
      end
    _DEFAULT: begin
        if (1'h1) begin
          r_w_seq_next_state = _DEFAULT;
        end
      end
    default: begin end
  endcase
end
always_comb begin
  unique case (r_w_seq_current_state)
    IDLE: begin :r_w_seq_IDLE_Output
        ready = 1'h1;
        valid_out = 1'h0;
        data_out[15:0] = 16'h0;
        data_out[16] = 1'h0;
        write_gate = 1'h0;
        read_gate = 1'h1;
      end :r_w_seq_IDLE_Output
    MODIFY: begin :r_w_seq_MODIFY_Output
        ready = 1'h0;
        valid_out = 1'h0;
        data_out[15:0] = 16'h0;
        data_out[16] = 1'h0;
        write_gate = 1'h1;
        read_gate = 1'h0;
      end :r_w_seq_MODIFY_Output
    READ: begin :r_w_seq_READ_Output
        ready = 1'h1;
        valid_out = 1'h1;
        data_out[15:0] = data_from_strg[rd_bank][addr_to_write[1:0]];
        data_out[16] = 1'h0;
        write_gate = 1'h0;
        read_gate = 1'h1;
      end :r_w_seq_READ_Output
    _DEFAULT: begin :r_w_seq__DEFAULT_Output
        ready = 1'h0;
        valid_out = 1'h0;
        data_out[15:0] = 16'h0;
        data_out[16] = 1'h0;
        write_gate = 1'h0;
        read_gate = 1'h0;
      end :r_w_seq__DEFAULT_Output
    default: begin end
  endcase
end
endmodule   // strg_ram_64_512_delay1

module strg_ram_64_512_delay1_flat (
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] data_in_f_,
  input logic flush,
  input logic [0:0] [16:0] rd_addr_in_f_,
  input logic ren_f_,
  input logic rst_n,
  input logic [0:0][3:0] [15:0] strg_ram_64_512_delay1_inst_data_from_strg_lifted,
  input logic wen_f_,
  input logic [0:0] [16:0] wr_addr_in_f_,
  output logic [0:0] [16:0] data_out_f_,
  output logic ready_f_,
  output logic [0:0] [8:0] strg_ram_64_512_delay1_inst_addr_out_lifted,
  output logic [0:0][3:0] [15:0] strg_ram_64_512_delay1_inst_data_to_strg_lifted,
  output logic strg_ram_64_512_delay1_inst_ren_to_strg_lifted,
  output logic strg_ram_64_512_delay1_inst_wen_to_strg_lifted,
  output logic valid_out_f_
);

strg_ram_64_512_delay1 strg_ram_64_512_delay1_inst (
  .clk(clk),
  .clk_en(clk_en),
  .data_from_strg(strg_ram_64_512_delay1_inst_data_from_strg_lifted),
  .data_in(data_in_f_),
  .flush(flush),
  .rd_addr_in(rd_addr_in_f_),
  .ren(ren_f_),
  .rst_n(rst_n),
  .wen(wen_f_),
  .wr_addr_in(wr_addr_in_f_),
  .addr_out(strg_ram_64_512_delay1_inst_addr_out_lifted),
  .data_out(data_out_f_),
  .data_to_strg(strg_ram_64_512_delay1_inst_data_to_strg_lifted),
  .ready(ready_f_),
  .ren_to_strg(strg_ram_64_512_delay1_inst_ren_to_strg_lifted),
  .valid_out(valid_out_f_),
  .wen_to_strg(strg_ram_64_512_delay1_inst_wen_to_strg_lifted)
);

endmodule   // strg_ram_64_512_delay1_flat

module strg_ub_agg_only (
  input logic [1:0] agg_read,
  input logic [2:0] agg_write_addr_gen_0_starting_addr,
  input logic [2:0] agg_write_addr_gen_0_strides_0,
  input logic [2:0] agg_write_addr_gen_0_strides_1,
  input logic [2:0] agg_write_addr_gen_0_strides_2,
  input logic [2:0] agg_write_addr_gen_1_starting_addr,
  input logic [2:0] agg_write_addr_gen_1_strides_0,
  input logic [2:0] agg_write_addr_gen_1_strides_1,
  input logic [2:0] agg_write_addr_gen_1_strides_2,
  input logic agg_write_sched_gen_0_enable,
  input logic [15:0] agg_write_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] agg_write_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] agg_write_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] agg_write_sched_gen_0_sched_addr_gen_strides_2,
  input logic agg_write_sched_gen_1_enable,
  input logic [15:0] agg_write_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] agg_write_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] agg_write_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] agg_write_sched_gen_1_sched_addr_gen_strides_2,
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic [1:0] [15:0] data_in,
  input logic flush,
  input logic [2:0] loops_in2buf_0_dimensionality,
  input logic [10:0] loops_in2buf_0_ranges_0,
  input logic [10:0] loops_in2buf_0_ranges_1,
  input logic [10:0] loops_in2buf_0_ranges_2,
  input logic [2:0] loops_in2buf_1_dimensionality,
  input logic [10:0] loops_in2buf_1_ranges_0,
  input logic [10:0] loops_in2buf_1_ranges_1,
  input logic [10:0] loops_in2buf_1_ranges_2,
  input logic rst_n,
  input logic [1:0] [8:0] sram_read_addr_in,
  input logic [1:0] [2:0] tb_read_addr_d_in,
  input logic [1:0] tb_read_d_in,
  input logic [1:0] [1:0] update_mode_in,
  output logic [1:0][3:0] [15:0] agg_data_out,
  output logic [1:0] [1:0] agg_write_addr_l2b_out,
  output logic [1:0] [2:0] agg_write_mux_sel_out,
  output logic [1:0] agg_write_out,
  output logic [1:0] agg_write_restart_out
);

logic [1:0][1:0][3:0][15:0] agg;
logic [1:0] agg_read_addr;
logic [1:0][7:0] agg_read_addr_gen_out;
logic [1:0] agg_read_addr_in;
logic [1:0] agg_write;
logic [1:0][2:0] agg_write_addr;
logic [2:0] agg_write_addr_gen_0_addr_out;
logic [2:0][2:0] agg_write_addr_gen_0_strides;
logic [2:0] agg_write_addr_gen_1_addr_out;
logic [2:0][2:0] agg_write_addr_gen_1_strides;
logic agg_write_sched_gen_0_valid_output;
logic agg_write_sched_gen_1_valid_output;
logic [2:0] fl_mux_sel_0;
logic [2:0] fl_mux_sel_1;
logic [1:0] loops_in2buf_0_mux_sel_out;
logic [2:0][10:0] loops_in2buf_0_ranges;
logic loops_in2buf_0_restart;
logic [1:0] loops_in2buf_1_mux_sel_out;
logic [2:0][10:0] loops_in2buf_1_ranges;
logic loops_in2buf_1_restart;
logic [1:0] mode_0;
logic [1:0] mode_1;
logic [2:0] tb_addr_0;
logic [2:0] tb_addr_1;
logic tb_read_0;
logic tb_read_1;
assign agg_write_out = agg_write;
assign mode_0 = update_mode_in[0];
assign agg_write_addr_l2b_out[0] = agg_write_addr[0][1:0];
assign tb_read_0 = mode_0[0] ? tb_read_d_in[1]: tb_read_d_in[0];
assign tb_addr_0 = mode_0[0] ? tb_read_addr_d_in[1]: tb_read_addr_d_in[0];
assign fl_mux_sel_0[1:0] = loops_in2buf_0_mux_sel_out;
assign fl_mux_sel_0[2] = 1'h0;
assign agg_write_mux_sel_out[0] = fl_mux_sel_0;
assign agg_write_restart_out[0] = loops_in2buf_0_restart;
assign agg_write_addr[0] = mode_0[1] ? tb_addr_0: agg_write_addr_gen_0_addr_out;
assign agg_write[0] = mode_0[1] ? tb_read_0: agg_write_sched_gen_0_valid_output;

always_ff @(posedge clk) begin
  if (clk_en) begin
    if (agg_write[0]) begin
      agg[0][agg_write_addr[0][2]][agg_write_addr[0][1:0]] <= data_in[0];
    end
  end
end
assign agg_read_addr_in[0] = sram_read_addr_in[0][0];
assign agg_read_addr_gen_out[0][0] = agg_read_addr_in[0];
assign agg_read_addr_gen_out[0][7:1] = 7'h0;
assign agg_read_addr[0] = agg_read_addr_gen_out[0][0];
always_comb begin
  agg_data_out[0] = agg[0][agg_read_addr[0]];
end
assign mode_1 = update_mode_in[1];
assign agg_write_addr_l2b_out[1] = agg_write_addr[1][1:0];
assign tb_read_1 = mode_1[0] ? tb_read_d_in[1]: tb_read_d_in[0];
assign tb_addr_1 = mode_1[0] ? tb_read_addr_d_in[1]: tb_read_addr_d_in[0];
assign fl_mux_sel_1[1:0] = loops_in2buf_1_mux_sel_out;
assign fl_mux_sel_1[2] = 1'h0;
assign agg_write_mux_sel_out[1] = fl_mux_sel_1;
assign agg_write_restart_out[1] = loops_in2buf_1_restart;
assign agg_write_addr[1] = mode_1[1] ? tb_addr_1: agg_write_addr_gen_1_addr_out;
assign agg_write[1] = mode_1[1] ? tb_read_1: agg_write_sched_gen_1_valid_output;

always_ff @(posedge clk) begin
  if (clk_en) begin
    if (agg_write[1]) begin
      agg[1][agg_write_addr[1][2]][agg_write_addr[1][1:0]] <= data_in[1];
    end
  end
end
assign agg_read_addr_in[1] = sram_read_addr_in[1][0];
assign agg_read_addr_gen_out[1][0] = agg_read_addr_in[1];
assign agg_read_addr_gen_out[1][7:1] = 7'h0;
assign agg_read_addr[1] = agg_read_addr_gen_out[1][0];
always_comb begin
  agg_data_out[1] = agg[1][agg_read_addr[1]];
end
assign loops_in2buf_0_ranges[0] = loops_in2buf_0_ranges_0;
assign loops_in2buf_0_ranges[1] = loops_in2buf_0_ranges_1;
assign loops_in2buf_0_ranges[2] = loops_in2buf_0_ranges_2;
assign agg_write_addr_gen_0_strides[0] = agg_write_addr_gen_0_strides_0;
assign agg_write_addr_gen_0_strides[1] = agg_write_addr_gen_0_strides_1;
assign agg_write_addr_gen_0_strides[2] = agg_write_addr_gen_0_strides_2;
assign loops_in2buf_1_ranges[0] = loops_in2buf_1_ranges_0;
assign loops_in2buf_1_ranges[1] = loops_in2buf_1_ranges_1;
assign loops_in2buf_1_ranges[2] = loops_in2buf_1_ranges_2;
assign agg_write_addr_gen_1_strides[0] = agg_write_addr_gen_1_strides_0;
assign agg_write_addr_gen_1_strides[1] = agg_write_addr_gen_1_strides_1;
assign agg_write_addr_gen_1_strides[2] = agg_write_addr_gen_1_strides_2;
for_loop_3_11 loops_in2buf_0 (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_in2buf_0_dimensionality),
  .flush(flush),
  .ranges(loops_in2buf_0_ranges),
  .rst_n(rst_n),
  .step(agg_write[0]),
  .mux_sel_out(loops_in2buf_0_mux_sel_out),
  .restart(loops_in2buf_0_restart)
);

addr_gen_3_3 agg_write_addr_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(loops_in2buf_0_mux_sel_out),
  .restart(loops_in2buf_0_restart),
  .rst_n(rst_n),
  .starting_addr(agg_write_addr_gen_0_starting_addr),
  .step(agg_write[0]),
  .strides(agg_write_addr_gen_0_strides),
  .addr_out(agg_write_addr_gen_0_addr_out)
);

sched_gen_3_16 agg_write_sched_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(agg_write_sched_gen_0_enable),
  .finished(loops_in2buf_0_restart),
  .flush(flush),
  .mux_sel(loops_in2buf_0_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_starting_addr(agg_write_sched_gen_0_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(agg_write_sched_gen_0_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(agg_write_sched_gen_0_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(agg_write_sched_gen_0_sched_addr_gen_strides_2),
  .valid_output(agg_write_sched_gen_0_valid_output)
);

for_loop_3_11 loops_in2buf_1 (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_in2buf_1_dimensionality),
  .flush(flush),
  .ranges(loops_in2buf_1_ranges),
  .rst_n(rst_n),
  .step(agg_write[1]),
  .mux_sel_out(loops_in2buf_1_mux_sel_out),
  .restart(loops_in2buf_1_restart)
);

addr_gen_3_3 agg_write_addr_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(loops_in2buf_1_mux_sel_out),
  .restart(loops_in2buf_1_restart),
  .rst_n(rst_n),
  .starting_addr(agg_write_addr_gen_1_starting_addr),
  .step(agg_write[1]),
  .strides(agg_write_addr_gen_1_strides),
  .addr_out(agg_write_addr_gen_1_addr_out)
);

sched_gen_3_16 agg_write_sched_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(agg_write_sched_gen_1_enable),
  .finished(loops_in2buf_1_restart),
  .flush(flush),
  .mux_sel(loops_in2buf_1_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_starting_addr(agg_write_sched_gen_1_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(agg_write_sched_gen_1_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(agg_write_sched_gen_1_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(agg_write_sched_gen_1_sched_addr_gen_strides_2),
  .valid_output(agg_write_sched_gen_1_valid_output)
);

endmodule   // strg_ub_agg_only

module strg_ub_agg_sram_shared (
  input logic [7:0] agg_read_sched_gen_0_agg_read_padding,
  input logic [7:0] agg_read_sched_gen_1_agg_read_padding,
  input logic [8:0] agg_sram_shared_addr_gen_0_starting_addr,
  input logic [8:0] agg_sram_shared_addr_gen_1_starting_addr,
  input logic [1:0] [1:0] agg_write_addr_l2b_in,
  input logic [1:0] agg_write_in,
  input logic [1:0] [2:0] agg_write_mux_sel_in,
  input logic [1:0] agg_write_restart_in,
  input logic clk,
  input logic clk_en,
  input logic flush,
  input logic [1:0] mode_0,
  input logic [1:0] mode_1,
  input logic rst_n,
  input logic [1:0] [8:0] sram_read_addr_in,
  input logic [1:0] sram_read_d_in,
  input logic [1:0] sram_read_in,
  output logic [1:0] agg_read_out,
  output logic [1:0] [8:0] agg_sram_shared_addr_out,
  output logic [1:0] [1:0] update_mode_out
);

logic [1:0] agg_read;
logic agg_read_sched_gen_0_valid_output;
logic agg_read_sched_gen_1_valid_output;
logic [8:0] agg_sram_shared_addr_gen_0_addr_out;
logic [8:0] agg_sram_shared_addr_gen_1_addr_out;
assign agg_read_out = agg_read;
assign update_mode_out[0] = mode_0;
assign agg_read[0] = agg_read_sched_gen_0_valid_output;
assign agg_sram_shared_addr_out[0] = agg_sram_shared_addr_gen_0_addr_out;
assign update_mode_out[1] = mode_1;
assign agg_read[1] = agg_read_sched_gen_1_valid_output;
assign agg_sram_shared_addr_out[1] = agg_sram_shared_addr_gen_1_addr_out;
agg_sram_shared_sched_gen agg_read_sched_gen_0 (
  .agg_read_padding(agg_read_sched_gen_0_agg_read_padding),
  .agg_write(agg_write_in[0]),
  .agg_write_addr_l2b(agg_write_addr_l2b_in[0]),
  .agg_write_mux_sel(agg_write_mux_sel_in[0]),
  .agg_write_restart(agg_write_restart_in[0]),
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mode(mode_0),
  .rst_n(rst_n),
  .sram_read_d(sram_read_d_in),
  .valid_output(agg_read_sched_gen_0_valid_output)
);

agg_sram_shared_addr_gen agg_sram_shared_addr_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mode(mode_0),
  .rst_n(rst_n),
  .sram_read(sram_read_in),
  .sram_read_addr(sram_read_addr_in),
  .starting_addr(agg_sram_shared_addr_gen_0_starting_addr),
  .step(agg_read[0]),
  .addr_out(agg_sram_shared_addr_gen_0_addr_out)
);

agg_sram_shared_sched_gen agg_read_sched_gen_1 (
  .agg_read_padding(agg_read_sched_gen_1_agg_read_padding),
  .agg_write(agg_write_in[1]),
  .agg_write_addr_l2b(agg_write_addr_l2b_in[1]),
  .agg_write_mux_sel(agg_write_mux_sel_in[1]),
  .agg_write_restart(agg_write_restart_in[1]),
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mode(mode_1),
  .rst_n(rst_n),
  .sram_read_d(sram_read_d_in),
  .valid_output(agg_read_sched_gen_1_valid_output)
);

agg_sram_shared_addr_gen agg_sram_shared_addr_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mode(mode_1),
  .rst_n(rst_n),
  .sram_read(sram_read_in),
  .sram_read_addr(sram_read_addr_in),
  .starting_addr(agg_sram_shared_addr_gen_1_starting_addr),
  .step(agg_read[1]),
  .addr_out(agg_sram_shared_addr_gen_1_addr_out)
);

endmodule   // strg_ub_agg_sram_shared

module strg_ub_sram_only (
  input logic [1:0][3:0] [15:0] agg_data_out,
  input logic [1:0] agg_read,
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic flush,
  input logic [1:0] [2:0] loops_sram2tb_mux_sel,
  input logic [1:0] loops_sram2tb_restart,
  input logic [8:0] output_addr_gen_0_starting_addr,
  input logic [8:0] output_addr_gen_0_strides_0,
  input logic [8:0] output_addr_gen_0_strides_1,
  input logic [8:0] output_addr_gen_0_strides_2,
  input logic [8:0] output_addr_gen_0_strides_3,
  input logic [8:0] output_addr_gen_0_strides_4,
  input logic [8:0] output_addr_gen_0_strides_5,
  input logic [8:0] output_addr_gen_1_starting_addr,
  input logic [8:0] output_addr_gen_1_strides_0,
  input logic [8:0] output_addr_gen_1_strides_1,
  input logic [8:0] output_addr_gen_1_strides_2,
  input logic [8:0] output_addr_gen_1_strides_3,
  input logic [8:0] output_addr_gen_1_strides_4,
  input logic [8:0] output_addr_gen_1_strides_5,
  input logic rst_n,
  input logic [1:0] [8:0] sram_read_addr_in,
  input logic [1:0] t_read,
  output logic [8:0] addr_to_sram,
  output logic cen_to_sram,
  output logic [3:0] [15:0] data_to_sram,
  output logic [1:0] [8:0] sram_read_addr_out,
  output logic wen_to_sram
);

logic [8:0] addr;
logic [3:0][15:0] decode_ret_agg_read_agg_data_out;
logic [15:0] decode_ret_agg_read_s_write_addr;
logic [15:0] decode_ret_t_read_s_read_addr;
logic decode_sel_done_agg_read_agg_data_out;
logic decode_sel_done_agg_read_s_write_addr;
logic decode_sel_done_t_read_s_read_addr;
logic [8:0] output_addr_gen_0_addr_out;
logic [5:0][8:0] output_addr_gen_0_strides;
logic [8:0] output_addr_gen_1_addr_out;
logic [5:0][8:0] output_addr_gen_1_strides;
logic read;
logic [1:0][15:0] s_read_addr;
logic [1:0][15:0] s_write_addr;
logic [3:0][15:0] sram_write_data;
logic write;
assign s_write_addr[0][8:0] = sram_read_addr_in[0];
assign s_write_addr[0][15:9] = 7'h0;
assign s_write_addr[1][8:0] = sram_read_addr_in[1];
assign s_write_addr[1][15:9] = 7'h0;
assign s_read_addr[0][8:0] = output_addr_gen_0_addr_out;
assign s_read_addr[0][15:9] = 7'h0;
assign sram_read_addr_out[0] = output_addr_gen_0_addr_out;
assign s_read_addr[1][8:0] = output_addr_gen_1_addr_out;
assign s_read_addr[1][15:9] = 7'h0;
assign sram_read_addr_out[1] = output_addr_gen_1_addr_out;
assign data_to_sram = sram_write_data;
assign wen_to_sram = write;
always_comb begin
  decode_sel_done_agg_read_s_write_addr = 1'h0;
  decode_ret_agg_read_s_write_addr = 16'h0;
  for (int unsigned i = 0; i < 2; i += 1) begin
      if ((~decode_sel_done_agg_read_s_write_addr) & agg_read[1'(i)]) begin
        decode_ret_agg_read_s_write_addr = s_write_addr[1'(i)];
        decode_sel_done_agg_read_s_write_addr = 1'h1;
      end
    end
end
always_comb begin
  decode_sel_done_t_read_s_read_addr = 1'h0;
  decode_ret_t_read_s_read_addr = 16'h0;
  for (int unsigned i = 0; i < 2; i += 1) begin
      if ((~decode_sel_done_t_read_s_read_addr) & t_read[1'(i)]) begin
        decode_ret_t_read_s_read_addr = s_read_addr[1'(i)];
        decode_sel_done_t_read_s_read_addr = 1'h1;
      end
    end
end
assign cen_to_sram = write | read;
assign addr_to_sram = addr;
always_comb begin
  if (write) begin
    addr = decode_ret_agg_read_s_write_addr[8:0];
  end
  else addr = decode_ret_t_read_s_read_addr[8:0];
end
assign write = |agg_read;
assign read = |t_read;
always_comb begin
  decode_sel_done_agg_read_agg_data_out = 1'h0;
  decode_ret_agg_read_agg_data_out = 64'h0;
  for (int unsigned i = 0; i < 2; i += 1) begin
      if ((~decode_sel_done_agg_read_agg_data_out) & agg_read[1'(i)]) begin
        decode_ret_agg_read_agg_data_out = agg_data_out[1'(i)];
        decode_sel_done_agg_read_agg_data_out = 1'h1;
      end
    end
end
assign sram_write_data = decode_ret_agg_read_agg_data_out;
assign output_addr_gen_0_strides[0] = output_addr_gen_0_strides_0;
assign output_addr_gen_0_strides[1] = output_addr_gen_0_strides_1;
assign output_addr_gen_0_strides[2] = output_addr_gen_0_strides_2;
assign output_addr_gen_0_strides[3] = output_addr_gen_0_strides_3;
assign output_addr_gen_0_strides[4] = output_addr_gen_0_strides_4;
assign output_addr_gen_0_strides[5] = output_addr_gen_0_strides_5;
assign output_addr_gen_1_strides[0] = output_addr_gen_1_strides_0;
assign output_addr_gen_1_strides[1] = output_addr_gen_1_strides_1;
assign output_addr_gen_1_strides[2] = output_addr_gen_1_strides_2;
assign output_addr_gen_1_strides[3] = output_addr_gen_1_strides_3;
assign output_addr_gen_1_strides[4] = output_addr_gen_1_strides_4;
assign output_addr_gen_1_strides[5] = output_addr_gen_1_strides_5;
addr_gen_6_9 output_addr_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(loops_sram2tb_mux_sel[0]),
  .restart(loops_sram2tb_restart[0]),
  .rst_n(rst_n),
  .starting_addr(output_addr_gen_0_starting_addr),
  .step(t_read[0]),
  .strides(output_addr_gen_0_strides),
  .addr_out(output_addr_gen_0_addr_out)
);

addr_gen_6_9 output_addr_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(loops_sram2tb_mux_sel[1]),
  .restart(loops_sram2tb_restart[1]),
  .rst_n(rst_n),
  .starting_addr(output_addr_gen_1_starting_addr),
  .step(t_read[1]),
  .strides(output_addr_gen_1_strides),
  .addr_out(output_addr_gen_1_addr_out)
);

endmodule   // strg_ub_sram_only

module strg_ub_sram_tb_shared (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic flush,
  input logic [3:0] loops_buf2out_autovec_read_0_dimensionality,
  input logic [10:0] loops_buf2out_autovec_read_0_ranges_0,
  input logic [10:0] loops_buf2out_autovec_read_0_ranges_1,
  input logic [10:0] loops_buf2out_autovec_read_0_ranges_2,
  input logic [10:0] loops_buf2out_autovec_read_0_ranges_3,
  input logic [10:0] loops_buf2out_autovec_read_0_ranges_4,
  input logic [10:0] loops_buf2out_autovec_read_0_ranges_5,
  input logic [3:0] loops_buf2out_autovec_read_1_dimensionality,
  input logic [10:0] loops_buf2out_autovec_read_1_ranges_0,
  input logic [10:0] loops_buf2out_autovec_read_1_ranges_1,
  input logic [10:0] loops_buf2out_autovec_read_1_ranges_2,
  input logic [10:0] loops_buf2out_autovec_read_1_ranges_3,
  input logic [10:0] loops_buf2out_autovec_read_1_ranges_4,
  input logic [10:0] loops_buf2out_autovec_read_1_ranges_5,
  input logic output_sched_gen_0_enable,
  input logic [9:0] output_sched_gen_0_sched_addr_gen_delay,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_strides_2,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_strides_3,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_strides_4,
  input logic [15:0] output_sched_gen_0_sched_addr_gen_strides_5,
  input logic output_sched_gen_1_enable,
  input logic [9:0] output_sched_gen_1_sched_addr_gen_delay,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_strides_2,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_strides_3,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_strides_4,
  input logic [15:0] output_sched_gen_1_sched_addr_gen_strides_5,
  input logic rst_n,
  output logic [1:0] [2:0] loops_sram2tb_mux_sel,
  output logic [1:0] loops_sram2tb_restart,
  output logic [1:0] sram_read_d,
  output logic [1:0] t_read_out
);

logic [2:0] loops_buf2out_autovec_read_0_mux_sel_out;
logic [5:0][10:0] loops_buf2out_autovec_read_0_ranges;
logic loops_buf2out_autovec_read_0_restart;
logic [2:0] loops_buf2out_autovec_read_1_mux_sel_out;
logic [5:0][10:0] loops_buf2out_autovec_read_1_ranges;
logic loops_buf2out_autovec_read_1_restart;
logic output_sched_gen_0_valid_output;
logic output_sched_gen_0_valid_output_d;
logic output_sched_gen_1_valid_output;
logic output_sched_gen_1_valid_output_d;
logic [1:0] t_read;
assign t_read_out = t_read;
assign loops_sram2tb_mux_sel[0] = loops_buf2out_autovec_read_0_mux_sel_out;
assign loops_sram2tb_restart[0] = loops_buf2out_autovec_read_0_restart;
assign t_read[0] = output_sched_gen_0_valid_output;
assign sram_read_d[0] = output_sched_gen_0_valid_output_d;
assign loops_sram2tb_mux_sel[1] = loops_buf2out_autovec_read_1_mux_sel_out;
assign loops_sram2tb_restart[1] = loops_buf2out_autovec_read_1_restart;
assign t_read[1] = output_sched_gen_1_valid_output;
assign sram_read_d[1] = output_sched_gen_1_valid_output_d;
assign loops_buf2out_autovec_read_0_ranges[0] = loops_buf2out_autovec_read_0_ranges_0;
assign loops_buf2out_autovec_read_0_ranges[1] = loops_buf2out_autovec_read_0_ranges_1;
assign loops_buf2out_autovec_read_0_ranges[2] = loops_buf2out_autovec_read_0_ranges_2;
assign loops_buf2out_autovec_read_0_ranges[3] = loops_buf2out_autovec_read_0_ranges_3;
assign loops_buf2out_autovec_read_0_ranges[4] = loops_buf2out_autovec_read_0_ranges_4;
assign loops_buf2out_autovec_read_0_ranges[5] = loops_buf2out_autovec_read_0_ranges_5;
assign loops_buf2out_autovec_read_1_ranges[0] = loops_buf2out_autovec_read_1_ranges_0;
assign loops_buf2out_autovec_read_1_ranges[1] = loops_buf2out_autovec_read_1_ranges_1;
assign loops_buf2out_autovec_read_1_ranges[2] = loops_buf2out_autovec_read_1_ranges_2;
assign loops_buf2out_autovec_read_1_ranges[3] = loops_buf2out_autovec_read_1_ranges_3;
assign loops_buf2out_autovec_read_1_ranges[4] = loops_buf2out_autovec_read_1_ranges_4;
assign loops_buf2out_autovec_read_1_ranges[5] = loops_buf2out_autovec_read_1_ranges_5;
for_loop_6_11 loops_buf2out_autovec_read_0 (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_buf2out_autovec_read_0_dimensionality),
  .flush(flush),
  .ranges(loops_buf2out_autovec_read_0_ranges),
  .rst_n(rst_n),
  .step(t_read[0]),
  .mux_sel_out(loops_buf2out_autovec_read_0_mux_sel_out),
  .restart(loops_buf2out_autovec_read_0_restart)
);

sched_gen_6_16_delay_addr_10_4 output_sched_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(output_sched_gen_0_enable),
  .finished(loops_buf2out_autovec_read_0_restart),
  .flush(flush),
  .mux_sel(loops_buf2out_autovec_read_0_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_delay(output_sched_gen_0_sched_addr_gen_delay),
  .sched_addr_gen_starting_addr(output_sched_gen_0_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(output_sched_gen_0_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(output_sched_gen_0_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(output_sched_gen_0_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(output_sched_gen_0_sched_addr_gen_strides_3),
  .sched_addr_gen_strides_4(output_sched_gen_0_sched_addr_gen_strides_4),
  .sched_addr_gen_strides_5(output_sched_gen_0_sched_addr_gen_strides_5),
  .valid_output(output_sched_gen_0_valid_output),
  .valid_output_d(output_sched_gen_0_valid_output_d)
);

for_loop_6_11 loops_buf2out_autovec_read_1 (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_buf2out_autovec_read_1_dimensionality),
  .flush(flush),
  .ranges(loops_buf2out_autovec_read_1_ranges),
  .rst_n(rst_n),
  .step(t_read[1]),
  .mux_sel_out(loops_buf2out_autovec_read_1_mux_sel_out),
  .restart(loops_buf2out_autovec_read_1_restart)
);

sched_gen_6_16_delay_addr_10_4 output_sched_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(output_sched_gen_1_enable),
  .finished(loops_buf2out_autovec_read_1_restart),
  .flush(flush),
  .mux_sel(loops_buf2out_autovec_read_1_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_delay(output_sched_gen_1_sched_addr_gen_delay),
  .sched_addr_gen_starting_addr(output_sched_gen_1_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(output_sched_gen_1_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(output_sched_gen_1_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(output_sched_gen_1_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(output_sched_gen_1_sched_addr_gen_strides_3),
  .sched_addr_gen_strides_4(output_sched_gen_1_sched_addr_gen_strides_4),
  .sched_addr_gen_strides_5(output_sched_gen_1_sched_addr_gen_strides_5),
  .valid_output(output_sched_gen_1_valid_output),
  .valid_output_d(output_sched_gen_1_valid_output_d)
);

endmodule   // strg_ub_sram_tb_shared

module strg_ub_tb_only (
  input logic clk,
  input logic clk_en,
  input logic [15:0] cycle_count,
  input logic flush,
  input logic [3:0] loops_buf2out_read_0_dimensionality,
  input logic [10:0] loops_buf2out_read_0_ranges_0,
  input logic [10:0] loops_buf2out_read_0_ranges_1,
  input logic [10:0] loops_buf2out_read_0_ranges_2,
  input logic [10:0] loops_buf2out_read_0_ranges_3,
  input logic [10:0] loops_buf2out_read_0_ranges_4,
  input logic [10:0] loops_buf2out_read_0_ranges_5,
  input logic [3:0] loops_buf2out_read_1_dimensionality,
  input logic [10:0] loops_buf2out_read_1_ranges_0,
  input logic [10:0] loops_buf2out_read_1_ranges_1,
  input logic [10:0] loops_buf2out_read_1_ranges_2,
  input logic [10:0] loops_buf2out_read_1_ranges_3,
  input logic [10:0] loops_buf2out_read_1_ranges_4,
  input logic [10:0] loops_buf2out_read_1_ranges_5,
  input logic [1:0] [2:0] loops_sram2tb_mux_sel,
  input logic [1:0] loops_sram2tb_restart,
  input logic rst_n,
  input logic shared_tb_0,
  input logic [3:0] [15:0] sram_read_data,
  input logic [1:0] t_read,
  input logic [3:0] tb_read_addr_gen_0_starting_addr,
  input logic [3:0] tb_read_addr_gen_0_strides_0,
  input logic [3:0] tb_read_addr_gen_0_strides_1,
  input logic [3:0] tb_read_addr_gen_0_strides_2,
  input logic [3:0] tb_read_addr_gen_0_strides_3,
  input logic [3:0] tb_read_addr_gen_0_strides_4,
  input logic [3:0] tb_read_addr_gen_0_strides_5,
  input logic [3:0] tb_read_addr_gen_1_starting_addr,
  input logic [3:0] tb_read_addr_gen_1_strides_0,
  input logic [3:0] tb_read_addr_gen_1_strides_1,
  input logic [3:0] tb_read_addr_gen_1_strides_2,
  input logic [3:0] tb_read_addr_gen_1_strides_3,
  input logic [3:0] tb_read_addr_gen_1_strides_4,
  input logic [3:0] tb_read_addr_gen_1_strides_5,
  input logic tb_read_sched_gen_0_enable,
  input logic [9:0] tb_read_sched_gen_0_sched_addr_gen_delay,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_strides_2,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_strides_3,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_strides_4,
  input logic [15:0] tb_read_sched_gen_0_sched_addr_gen_strides_5,
  input logic tb_read_sched_gen_1_enable,
  input logic [9:0] tb_read_sched_gen_1_sched_addr_gen_delay,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_strides_2,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_strides_3,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_strides_4,
  input logic [15:0] tb_read_sched_gen_1_sched_addr_gen_strides_5,
  input logic [3:0] tb_write_addr_gen_0_starting_addr,
  input logic [3:0] tb_write_addr_gen_0_strides_0,
  input logic [3:0] tb_write_addr_gen_0_strides_1,
  input logic [3:0] tb_write_addr_gen_0_strides_2,
  input logic [3:0] tb_write_addr_gen_0_strides_3,
  input logic [3:0] tb_write_addr_gen_0_strides_4,
  input logic [3:0] tb_write_addr_gen_0_strides_5,
  input logic [3:0] tb_write_addr_gen_1_starting_addr,
  input logic [3:0] tb_write_addr_gen_1_strides_0,
  input logic [3:0] tb_write_addr_gen_1_strides_1,
  input logic [3:0] tb_write_addr_gen_1_strides_2,
  input logic [3:0] tb_write_addr_gen_1_strides_3,
  input logic [3:0] tb_write_addr_gen_1_strides_4,
  input logic [3:0] tb_write_addr_gen_1_strides_5,
  output logic [1:0] accessor_output,
  output logic [1:0] [15:0] data_out,
  output logic [1:0] [2:0] tb_read_addr_d_out,
  output logic [1:0] tb_read_d_out
);

logic [2:0] addr_fifo_in_0;
logic [2:0] addr_fifo_in_1;
logic delay_en_0;
logic delay_en_1;
logic [2:0] loops_buf2out_read_0_mux_sel_out;
logic [5:0][10:0] loops_buf2out_read_0_ranges;
logic loops_buf2out_read_0_restart;
logic [2:0] loops_buf2out_read_1_mux_sel_out;
logic [5:0][10:0] loops_buf2out_read_1_ranges;
logic loops_buf2out_read_1_restart;
logic [1:0][2:0] mux_sel_d1;
logic [2:0] rd_ptr_0;
logic [2:0] rd_ptr_1;
logic [1:0] restart_d1;
logic [1:0] t_read_d1;
logic [1:0][1:0][3:0][15:0] tb;
logic [7:0][2:0] tb_addr_fifo_0;
logic [7:0][2:0] tb_addr_fifo_1;
logic [1:0] tb_read;
logic [1:0][3:0] tb_read_addr;
logic [3:0] tb_read_addr_gen_0_addr_out;
logic [5:0][3:0] tb_read_addr_gen_0_strides;
logic [3:0] tb_read_addr_gen_1_addr_out;
logic [5:0][3:0] tb_read_addr_gen_1_strides;
logic tb_read_d_0;
logic tb_read_d_1;
logic tb_read_sched_gen_0_valid_output;
logic tb_read_sched_gen_1_valid_output;
logic tb_read_sel_0;
logic [1:0][2:0] tb_write_addr;
logic [3:0] tb_write_addr_gen_0_addr_out;
logic [5:0][3:0] tb_write_addr_gen_0_strides;
logic [3:0] tb_write_addr_gen_1_addr_out;
logic [5:0][3:0] tb_write_addr_gen_1_strides;
logic tb_write_sel_0;
logic [2:0] wr_ptr_0;
logic [2:0] wr_ptr_1;
assign accessor_output = tb_read;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    t_read_d1[0] <= 1'h0;
    mux_sel_d1[0] <= 3'h0;
    restart_d1[0] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      t_read_d1[0] <= 1'h0;
      mux_sel_d1[0] <= 3'h0;
      restart_d1[0] <= 1'h0;
    end
    else begin
      t_read_d1[0] <= t_read[0];
      mux_sel_d1[0] <= loops_sram2tb_mux_sel[0];
      restart_d1[0] <= loops_sram2tb_restart[0];
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    t_read_d1[1] <= 1'h0;
    mux_sel_d1[1] <= 3'h0;
    restart_d1[1] <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      t_read_d1[1] <= 1'h0;
      mux_sel_d1[1] <= 3'h0;
      restart_d1[1] <= 1'h0;
    end
    else begin
      t_read_d1[1] <= t_read[1];
      mux_sel_d1[1] <= loops_sram2tb_mux_sel[1];
      restart_d1[1] <= loops_sram2tb_restart[1];
    end
  end
end
assign tb_write_sel_0 = shared_tb_0 ? tb_write_addr[0][1]: 1'h0;
assign tb_read_sel_0 = shared_tb_0 ? tb_read_addr[0][3]: 1'h0;
assign tb_write_addr[0] = tb_write_addr_gen_0_addr_out[2:0];
assign tb_read_addr[0] = tb_read_addr_gen_0_addr_out;
assign tb_read[0] = tb_read_sched_gen_0_valid_output;
assign addr_fifo_in_0 = tb_read_addr_gen_0_addr_out[2:0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr_0 <= 3'h0;
    rd_ptr_0 <= 3'h0;
    tb_addr_fifo_0 <= 24'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      wr_ptr_0 <= 3'h0;
      rd_ptr_0 <= 3'h0;
      tb_addr_fifo_0 <= 24'h0;
    end
    else if (delay_en_0) begin
      if (tb_read[0]) begin
        tb_addr_fifo_0[wr_ptr_0] <= addr_fifo_in_0;
        wr_ptr_0 <= wr_ptr_0 + 3'h1;
      end
      if (tb_read_d_0) begin
        rd_ptr_0 <= rd_ptr_0 + 3'h1;
      end
    end
  end
end
assign tb_read_d_out[0] = delay_en_0 ? tb_read_d_0: tb_read[0];
assign tb_read_addr_d_out[0] = delay_en_0 ? tb_addr_fifo_0[rd_ptr_0]: addr_fifo_in_0;
always_comb begin
  data_out[0] = tb[tb_read_sel_0][tb_read_addr[0][2]][tb_read_addr[0][1:0]];
end
assign tb_write_addr[1] = tb_write_addr_gen_1_addr_out[2:0];
assign tb_read_addr[1] = tb_read_addr_gen_1_addr_out;
assign tb_read[1] = tb_read_sched_gen_1_valid_output;
assign addr_fifo_in_1 = tb_read_addr_gen_1_addr_out[2:0];

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wr_ptr_1 <= 3'h0;
    rd_ptr_1 <= 3'h0;
    tb_addr_fifo_1 <= 24'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      wr_ptr_1 <= 3'h0;
      rd_ptr_1 <= 3'h0;
      tb_addr_fifo_1 <= 24'h0;
    end
    else if (delay_en_1) begin
      if (tb_read[1]) begin
        tb_addr_fifo_1[wr_ptr_1] <= addr_fifo_in_1;
        wr_ptr_1 <= wr_ptr_1 + 3'h1;
      end
      if (tb_read_d_1) begin
        rd_ptr_1 <= rd_ptr_1 + 3'h1;
      end
    end
  end
end
assign tb_read_d_out[1] = delay_en_1 ? tb_read_d_1: tb_read[1];
assign tb_read_addr_d_out[1] = delay_en_1 ? tb_addr_fifo_1[rd_ptr_1]: addr_fifo_in_1;
always_comb begin
  data_out[1] = tb[1][tb_read_addr[1][2]][tb_read_addr[1][1:0]];
end

always_ff @(posedge clk) begin
  if (clk_en) begin
    for (int unsigned i = 0; i < 2; i += 1) begin
        if (t_read_d1[1'(i)]) begin
          if (i == 32'h0) begin
            tb[tb_write_sel_0][tb_write_addr[1'(i)][0]] <= sram_read_data;
          end
          else tb[1'(i)][tb_write_addr[1'(i)][0]] <= sram_read_data;
        end
      end
  end
end
assign tb_write_addr_gen_0_strides[0] = tb_write_addr_gen_0_strides_0;
assign tb_write_addr_gen_0_strides[1] = tb_write_addr_gen_0_strides_1;
assign tb_write_addr_gen_0_strides[2] = tb_write_addr_gen_0_strides_2;
assign tb_write_addr_gen_0_strides[3] = tb_write_addr_gen_0_strides_3;
assign tb_write_addr_gen_0_strides[4] = tb_write_addr_gen_0_strides_4;
assign tb_write_addr_gen_0_strides[5] = tb_write_addr_gen_0_strides_5;
assign loops_buf2out_read_0_ranges[0] = loops_buf2out_read_0_ranges_0;
assign loops_buf2out_read_0_ranges[1] = loops_buf2out_read_0_ranges_1;
assign loops_buf2out_read_0_ranges[2] = loops_buf2out_read_0_ranges_2;
assign loops_buf2out_read_0_ranges[3] = loops_buf2out_read_0_ranges_3;
assign loops_buf2out_read_0_ranges[4] = loops_buf2out_read_0_ranges_4;
assign loops_buf2out_read_0_ranges[5] = loops_buf2out_read_0_ranges_5;
assign tb_read_addr_gen_0_strides[0] = tb_read_addr_gen_0_strides_0;
assign tb_read_addr_gen_0_strides[1] = tb_read_addr_gen_0_strides_1;
assign tb_read_addr_gen_0_strides[2] = tb_read_addr_gen_0_strides_2;
assign tb_read_addr_gen_0_strides[3] = tb_read_addr_gen_0_strides_3;
assign tb_read_addr_gen_0_strides[4] = tb_read_addr_gen_0_strides_4;
assign tb_read_addr_gen_0_strides[5] = tb_read_addr_gen_0_strides_5;
assign tb_write_addr_gen_1_strides[0] = tb_write_addr_gen_1_strides_0;
assign tb_write_addr_gen_1_strides[1] = tb_write_addr_gen_1_strides_1;
assign tb_write_addr_gen_1_strides[2] = tb_write_addr_gen_1_strides_2;
assign tb_write_addr_gen_1_strides[3] = tb_write_addr_gen_1_strides_3;
assign tb_write_addr_gen_1_strides[4] = tb_write_addr_gen_1_strides_4;
assign tb_write_addr_gen_1_strides[5] = tb_write_addr_gen_1_strides_5;
assign loops_buf2out_read_1_ranges[0] = loops_buf2out_read_1_ranges_0;
assign loops_buf2out_read_1_ranges[1] = loops_buf2out_read_1_ranges_1;
assign loops_buf2out_read_1_ranges[2] = loops_buf2out_read_1_ranges_2;
assign loops_buf2out_read_1_ranges[3] = loops_buf2out_read_1_ranges_3;
assign loops_buf2out_read_1_ranges[4] = loops_buf2out_read_1_ranges_4;
assign loops_buf2out_read_1_ranges[5] = loops_buf2out_read_1_ranges_5;
assign tb_read_addr_gen_1_strides[0] = tb_read_addr_gen_1_strides_0;
assign tb_read_addr_gen_1_strides[1] = tb_read_addr_gen_1_strides_1;
assign tb_read_addr_gen_1_strides[2] = tb_read_addr_gen_1_strides_2;
assign tb_read_addr_gen_1_strides[3] = tb_read_addr_gen_1_strides_3;
assign tb_read_addr_gen_1_strides[4] = tb_read_addr_gen_1_strides_4;
assign tb_read_addr_gen_1_strides[5] = tb_read_addr_gen_1_strides_5;
addr_gen_6_4 tb_write_addr_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(mux_sel_d1[0]),
  .restart(restart_d1[0]),
  .rst_n(rst_n),
  .starting_addr(tb_write_addr_gen_0_starting_addr),
  .step(t_read_d1[0]),
  .strides(tb_write_addr_gen_0_strides),
  .addr_out(tb_write_addr_gen_0_addr_out)
);

for_loop_6_11 loops_buf2out_read_0 (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_buf2out_read_0_dimensionality),
  .flush(flush),
  .ranges(loops_buf2out_read_0_ranges),
  .rst_n(rst_n),
  .step(tb_read[0]),
  .mux_sel_out(loops_buf2out_read_0_mux_sel_out),
  .restart(loops_buf2out_read_0_restart)
);

addr_gen_6_4 tb_read_addr_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(loops_buf2out_read_0_mux_sel_out),
  .restart(loops_buf2out_read_0_restart),
  .rst_n(rst_n),
  .starting_addr(tb_read_addr_gen_0_starting_addr),
  .step(tb_read[0]),
  .strides(tb_read_addr_gen_0_strides),
  .addr_out(tb_read_addr_gen_0_addr_out)
);

sched_gen_6_16_delay_addr_10_8 tb_read_sched_gen_0 (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(tb_read_sched_gen_0_enable),
  .finished(loops_buf2out_read_0_restart),
  .flush(flush),
  .mux_sel(loops_buf2out_read_0_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_delay(tb_read_sched_gen_0_sched_addr_gen_delay),
  .sched_addr_gen_starting_addr(tb_read_sched_gen_0_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(tb_read_sched_gen_0_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(tb_read_sched_gen_0_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(tb_read_sched_gen_0_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(tb_read_sched_gen_0_sched_addr_gen_strides_3),
  .sched_addr_gen_strides_4(tb_read_sched_gen_0_sched_addr_gen_strides_4),
  .sched_addr_gen_strides_5(tb_read_sched_gen_0_sched_addr_gen_strides_5),
  .delay_en_out(delay_en_0),
  .valid_output(tb_read_sched_gen_0_valid_output),
  .valid_output_d(tb_read_d_0)
);

addr_gen_6_4 tb_write_addr_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(mux_sel_d1[1]),
  .restart(restart_d1[1]),
  .rst_n(rst_n),
  .starting_addr(tb_write_addr_gen_1_starting_addr),
  .step(t_read_d1[1]),
  .strides(tb_write_addr_gen_1_strides),
  .addr_out(tb_write_addr_gen_1_addr_out)
);

for_loop_6_11 loops_buf2out_read_1 (
  .clk(clk),
  .clk_en(clk_en),
  .dimensionality(loops_buf2out_read_1_dimensionality),
  .flush(flush),
  .ranges(loops_buf2out_read_1_ranges),
  .rst_n(rst_n),
  .step(tb_read[1]),
  .mux_sel_out(loops_buf2out_read_1_mux_sel_out),
  .restart(loops_buf2out_read_1_restart)
);

addr_gen_6_4 tb_read_addr_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mux_sel(loops_buf2out_read_1_mux_sel_out),
  .restart(loops_buf2out_read_1_restart),
  .rst_n(rst_n),
  .starting_addr(tb_read_addr_gen_1_starting_addr),
  .step(tb_read[1]),
  .strides(tb_read_addr_gen_1_strides),
  .addr_out(tb_read_addr_gen_1_addr_out)
);

sched_gen_6_16_delay_addr_10_8 tb_read_sched_gen_1 (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .enable(tb_read_sched_gen_1_enable),
  .finished(loops_buf2out_read_1_restart),
  .flush(flush),
  .mux_sel(loops_buf2out_read_1_mux_sel_out),
  .rst_n(rst_n),
  .sched_addr_gen_delay(tb_read_sched_gen_1_sched_addr_gen_delay),
  .sched_addr_gen_starting_addr(tb_read_sched_gen_1_sched_addr_gen_starting_addr),
  .sched_addr_gen_strides_0(tb_read_sched_gen_1_sched_addr_gen_strides_0),
  .sched_addr_gen_strides_1(tb_read_sched_gen_1_sched_addr_gen_strides_1),
  .sched_addr_gen_strides_2(tb_read_sched_gen_1_sched_addr_gen_strides_2),
  .sched_addr_gen_strides_3(tb_read_sched_gen_1_sched_addr_gen_strides_3),
  .sched_addr_gen_strides_4(tb_read_sched_gen_1_sched_addr_gen_strides_4),
  .sched_addr_gen_strides_5(tb_read_sched_gen_1_sched_addr_gen_strides_5),
  .delay_en_out(delay_en_1),
  .valid_output(tb_read_sched_gen_1_valid_output),
  .valid_output_d(tb_read_d_1)
);

endmodule   // strg_ub_tb_only

module strg_ub_vec (
  input logic [2:0] agg_only_agg_write_addr_gen_0_starting_addr,
  input logic [2:0] agg_only_agg_write_addr_gen_0_strides_0,
  input logic [2:0] agg_only_agg_write_addr_gen_0_strides_1,
  input logic [2:0] agg_only_agg_write_addr_gen_0_strides_2,
  input logic [2:0] agg_only_agg_write_addr_gen_1_starting_addr,
  input logic [2:0] agg_only_agg_write_addr_gen_1_strides_0,
  input logic [2:0] agg_only_agg_write_addr_gen_1_strides_1,
  input logic [2:0] agg_only_agg_write_addr_gen_1_strides_2,
  input logic agg_only_agg_write_sched_gen_0_enable,
  input logic [15:0] agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2,
  input logic agg_only_agg_write_sched_gen_1_enable,
  input logic [15:0] agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2,
  input logic [2:0] agg_only_loops_in2buf_0_dimensionality,
  input logic [10:0] agg_only_loops_in2buf_0_ranges_0,
  input logic [10:0] agg_only_loops_in2buf_0_ranges_1,
  input logic [10:0] agg_only_loops_in2buf_0_ranges_2,
  input logic [2:0] agg_only_loops_in2buf_1_dimensionality,
  input logic [10:0] agg_only_loops_in2buf_1_ranges_0,
  input logic [10:0] agg_only_loops_in2buf_1_ranges_1,
  input logic [10:0] agg_only_loops_in2buf_1_ranges_2,
  input logic [7:0] agg_sram_shared_agg_read_sched_gen_0_agg_read_padding,
  input logic [7:0] agg_sram_shared_agg_read_sched_gen_1_agg_read_padding,
  input logic [8:0] agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr,
  input logic [8:0] agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr,
  input logic [1:0] agg_sram_shared_mode_0,
  input logic [1:0] agg_sram_shared_mode_1,
  input logic chain_chain_en,
  input logic [1:0] [16:0] chain_data_in,
  input logic clk,
  input logic clk_en,
  input logic [3:0] [15:0] data_from_strg,
  input logic [1:0] [16:0] data_in,
  input logic flush,
  input logic rst_n,
  input logic [8:0] sram_only_output_addr_gen_0_starting_addr,
  input logic [8:0] sram_only_output_addr_gen_0_strides_0,
  input logic [8:0] sram_only_output_addr_gen_0_strides_1,
  input logic [8:0] sram_only_output_addr_gen_0_strides_2,
  input logic [8:0] sram_only_output_addr_gen_0_strides_3,
  input logic [8:0] sram_only_output_addr_gen_0_strides_4,
  input logic [8:0] sram_only_output_addr_gen_0_strides_5,
  input logic [8:0] sram_only_output_addr_gen_1_starting_addr,
  input logic [8:0] sram_only_output_addr_gen_1_strides_0,
  input logic [8:0] sram_only_output_addr_gen_1_strides_1,
  input logic [8:0] sram_only_output_addr_gen_1_strides_2,
  input logic [8:0] sram_only_output_addr_gen_1_strides_3,
  input logic [8:0] sram_only_output_addr_gen_1_strides_4,
  input logic [8:0] sram_only_output_addr_gen_1_strides_5,
  input logic [3:0] sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5,
  input logic [3:0] sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4,
  input logic [10:0] sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5,
  input logic sram_tb_shared_output_sched_gen_0_enable,
  input logic [9:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4,
  input logic [15:0] sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5,
  input logic sram_tb_shared_output_sched_gen_1_enable,
  input logic [9:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4,
  input logic [15:0] sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5,
  input logic [3:0] tb_only_loops_buf2out_read_0_dimensionality,
  input logic [10:0] tb_only_loops_buf2out_read_0_ranges_0,
  input logic [10:0] tb_only_loops_buf2out_read_0_ranges_1,
  input logic [10:0] tb_only_loops_buf2out_read_0_ranges_2,
  input logic [10:0] tb_only_loops_buf2out_read_0_ranges_3,
  input logic [10:0] tb_only_loops_buf2out_read_0_ranges_4,
  input logic [10:0] tb_only_loops_buf2out_read_0_ranges_5,
  input logic [3:0] tb_only_loops_buf2out_read_1_dimensionality,
  input logic [10:0] tb_only_loops_buf2out_read_1_ranges_0,
  input logic [10:0] tb_only_loops_buf2out_read_1_ranges_1,
  input logic [10:0] tb_only_loops_buf2out_read_1_ranges_2,
  input logic [10:0] tb_only_loops_buf2out_read_1_ranges_3,
  input logic [10:0] tb_only_loops_buf2out_read_1_ranges_4,
  input logic [10:0] tb_only_loops_buf2out_read_1_ranges_5,
  input logic tb_only_shared_tb_0,
  input logic [3:0] tb_only_tb_read_addr_gen_0_starting_addr,
  input logic [3:0] tb_only_tb_read_addr_gen_0_strides_0,
  input logic [3:0] tb_only_tb_read_addr_gen_0_strides_1,
  input logic [3:0] tb_only_tb_read_addr_gen_0_strides_2,
  input logic [3:0] tb_only_tb_read_addr_gen_0_strides_3,
  input logic [3:0] tb_only_tb_read_addr_gen_0_strides_4,
  input logic [3:0] tb_only_tb_read_addr_gen_0_strides_5,
  input logic [3:0] tb_only_tb_read_addr_gen_1_starting_addr,
  input logic [3:0] tb_only_tb_read_addr_gen_1_strides_0,
  input logic [3:0] tb_only_tb_read_addr_gen_1_strides_1,
  input logic [3:0] tb_only_tb_read_addr_gen_1_strides_2,
  input logic [3:0] tb_only_tb_read_addr_gen_1_strides_3,
  input logic [3:0] tb_only_tb_read_addr_gen_1_strides_4,
  input logic [3:0] tb_only_tb_read_addr_gen_1_strides_5,
  input logic tb_only_tb_read_sched_gen_0_enable,
  input logic [9:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_delay,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4,
  input logic [15:0] tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5,
  input logic tb_only_tb_read_sched_gen_1_enable,
  input logic [9:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_delay,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4,
  input logic [15:0] tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5,
  input logic [3:0] tb_only_tb_write_addr_gen_0_starting_addr,
  input logic [3:0] tb_only_tb_write_addr_gen_0_strides_0,
  input logic [3:0] tb_only_tb_write_addr_gen_0_strides_1,
  input logic [3:0] tb_only_tb_write_addr_gen_0_strides_2,
  input logic [3:0] tb_only_tb_write_addr_gen_0_strides_3,
  input logic [3:0] tb_only_tb_write_addr_gen_0_strides_4,
  input logic [3:0] tb_only_tb_write_addr_gen_0_strides_5,
  input logic [3:0] tb_only_tb_write_addr_gen_1_starting_addr,
  input logic [3:0] tb_only_tb_write_addr_gen_1_strides_0,
  input logic [3:0] tb_only_tb_write_addr_gen_1_strides_1,
  input logic [3:0] tb_only_tb_write_addr_gen_1_strides_2,
  input logic [3:0] tb_only_tb_write_addr_gen_1_strides_3,
  input logic [3:0] tb_only_tb_write_addr_gen_1_strides_4,
  input logic [3:0] tb_only_tb_write_addr_gen_1_strides_5,
  output logic [1:0] accessor_output,
  output logic [8:0] addr_out,
  output logic [1:0] [16:0] data_out,
  output logic [3:0] [15:0] data_to_strg,
  output logic ren_to_strg,
  output logic wen_to_strg
);

logic [1:0] accessor_output_int;
logic [1:0][3:0][15:0] agg_only_agg_data_out;
logic [1:0] agg_only_agg_read;
logic [1:0][1:0] agg_only_agg_write_addr_l2b_out;
logic [1:0][2:0] agg_only_agg_write_mux_sel_out;
logic [1:0] agg_only_agg_write_out;
logic [1:0] agg_only_agg_write_restart_out;
logic [1:0][8:0] agg_only_sram_read_addr_in;
logic [1:0][2:0] agg_only_tb_read_addr_d_in;
logic [1:0] agg_only_tb_read_d_in;
logic [1:0][1:0] agg_only_update_mode_in;
logic [1:0] agg_sram_shared_agg_read_out;
logic [1:0][8:0] agg_sram_shared_agg_sram_shared_addr_out;
logic [1:0][8:0] agg_sram_shared_sram_read_addr_in;
logic [1:0] agg_sram_shared_sram_read_d_in;
logic [1:0] agg_sram_shared_sram_read_in;
logic [1:0][15:0] chain_data_in_thin;
logic [15:0] cycle_count;
logic [1:0][15:0] data_in_thin;
logic [1:0][15:0] data_out_int;
logic [1:0][15:0] data_out_int_thin;
logic [1:0][2:0] sram_only_loops_sram2tb_mux_sel;
logic [1:0] sram_only_loops_sram2tb_restart;
logic [1:0] sram_only_t_read;
logic [1:0][2:0] sram_tb_shared_loops_sram2tb_mux_sel;
logic [1:0] sram_tb_shared_loops_sram2tb_restart;
logic [1:0] sram_tb_shared_t_read_out;
assign data_in_thin[0] = data_in[0][15:0];
assign data_in_thin[1] = data_in[1][15:0];
assign data_out[0][15:0] = data_out_int_thin[0];
assign data_out[0][16] = 1'h0;
assign data_out[1][15:0] = data_out_int_thin[1];
assign data_out[1][16] = 1'h0;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    cycle_count <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      cycle_count <= 16'h0;
    end
    else if (1'h1) begin
      cycle_count <= cycle_count + 16'h1;
    end
  end
end
assign agg_only_sram_read_addr_in = agg_sram_shared_agg_sram_shared_addr_out;
assign agg_sram_shared_sram_read_in = sram_tb_shared_t_read_out;
assign agg_only_agg_read = agg_sram_shared_agg_read_out;
assign sram_only_loops_sram2tb_mux_sel = sram_tb_shared_loops_sram2tb_mux_sel;
assign sram_only_loops_sram2tb_restart = sram_tb_shared_loops_sram2tb_restart;
assign sram_only_t_read = sram_tb_shared_t_read_out;
assign ren_to_strg = |sram_tb_shared_t_read_out;
assign chain_data_in_thin[0] = chain_data_in[0][15:0];
assign chain_data_in_thin[1] = chain_data_in[1][15:0];
assign accessor_output = accessor_output_int;
strg_ub_agg_only agg_only (
  .agg_read(agg_only_agg_read),
  .agg_write_addr_gen_0_starting_addr(agg_only_agg_write_addr_gen_0_starting_addr),
  .agg_write_addr_gen_0_strides_0(agg_only_agg_write_addr_gen_0_strides_0),
  .agg_write_addr_gen_0_strides_1(agg_only_agg_write_addr_gen_0_strides_1),
  .agg_write_addr_gen_0_strides_2(agg_only_agg_write_addr_gen_0_strides_2),
  .agg_write_addr_gen_1_starting_addr(agg_only_agg_write_addr_gen_1_starting_addr),
  .agg_write_addr_gen_1_strides_0(agg_only_agg_write_addr_gen_1_strides_0),
  .agg_write_addr_gen_1_strides_1(agg_only_agg_write_addr_gen_1_strides_1),
  .agg_write_addr_gen_1_strides_2(agg_only_agg_write_addr_gen_1_strides_2),
  .agg_write_sched_gen_0_enable(agg_only_agg_write_sched_gen_0_enable),
  .agg_write_sched_gen_0_sched_addr_gen_starting_addr(agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr),
  .agg_write_sched_gen_0_sched_addr_gen_strides_0(agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0),
  .agg_write_sched_gen_0_sched_addr_gen_strides_1(agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1),
  .agg_write_sched_gen_0_sched_addr_gen_strides_2(agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2),
  .agg_write_sched_gen_1_enable(agg_only_agg_write_sched_gen_1_enable),
  .agg_write_sched_gen_1_sched_addr_gen_starting_addr(agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr),
  .agg_write_sched_gen_1_sched_addr_gen_strides_0(agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0),
  .agg_write_sched_gen_1_sched_addr_gen_strides_1(agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1),
  .agg_write_sched_gen_1_sched_addr_gen_strides_2(agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2),
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .data_in(data_in_thin),
  .flush(flush),
  .loops_in2buf_0_dimensionality(agg_only_loops_in2buf_0_dimensionality),
  .loops_in2buf_0_ranges_0(agg_only_loops_in2buf_0_ranges_0),
  .loops_in2buf_0_ranges_1(agg_only_loops_in2buf_0_ranges_1),
  .loops_in2buf_0_ranges_2(agg_only_loops_in2buf_0_ranges_2),
  .loops_in2buf_1_dimensionality(agg_only_loops_in2buf_1_dimensionality),
  .loops_in2buf_1_ranges_0(agg_only_loops_in2buf_1_ranges_0),
  .loops_in2buf_1_ranges_1(agg_only_loops_in2buf_1_ranges_1),
  .loops_in2buf_1_ranges_2(agg_only_loops_in2buf_1_ranges_2),
  .rst_n(rst_n),
  .sram_read_addr_in(agg_only_sram_read_addr_in),
  .tb_read_addr_d_in(agg_only_tb_read_addr_d_in),
  .tb_read_d_in(agg_only_tb_read_d_in),
  .update_mode_in(agg_only_update_mode_in),
  .agg_data_out(agg_only_agg_data_out),
  .agg_write_addr_l2b_out(agg_only_agg_write_addr_l2b_out),
  .agg_write_mux_sel_out(agg_only_agg_write_mux_sel_out),
  .agg_write_out(agg_only_agg_write_out),
  .agg_write_restart_out(agg_only_agg_write_restart_out)
);

strg_ub_agg_sram_shared agg_sram_shared (
  .agg_read_sched_gen_0_agg_read_padding(agg_sram_shared_agg_read_sched_gen_0_agg_read_padding),
  .agg_read_sched_gen_1_agg_read_padding(agg_sram_shared_agg_read_sched_gen_1_agg_read_padding),
  .agg_sram_shared_addr_gen_0_starting_addr(agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr),
  .agg_sram_shared_addr_gen_1_starting_addr(agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr),
  .agg_write_addr_l2b_in(agg_only_agg_write_addr_l2b_out),
  .agg_write_in(agg_only_agg_write_out),
  .agg_write_mux_sel_in(agg_only_agg_write_mux_sel_out),
  .agg_write_restart_in(agg_only_agg_write_restart_out),
  .clk(clk),
  .clk_en(clk_en),
  .flush(flush),
  .mode_0(agg_sram_shared_mode_0),
  .mode_1(agg_sram_shared_mode_1),
  .rst_n(rst_n),
  .sram_read_addr_in(agg_sram_shared_sram_read_addr_in),
  .sram_read_d_in(agg_sram_shared_sram_read_d_in),
  .sram_read_in(agg_sram_shared_sram_read_in),
  .agg_read_out(agg_sram_shared_agg_read_out),
  .agg_sram_shared_addr_out(agg_sram_shared_agg_sram_shared_addr_out),
  .update_mode_out(agg_only_update_mode_in)
);

strg_ub_sram_only sram_only (
  .agg_data_out(agg_only_agg_data_out),
  .agg_read(agg_sram_shared_agg_read_out),
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .flush(flush),
  .loops_sram2tb_mux_sel(sram_only_loops_sram2tb_mux_sel),
  .loops_sram2tb_restart(sram_only_loops_sram2tb_restart),
  .output_addr_gen_0_starting_addr(sram_only_output_addr_gen_0_starting_addr),
  .output_addr_gen_0_strides_0(sram_only_output_addr_gen_0_strides_0),
  .output_addr_gen_0_strides_1(sram_only_output_addr_gen_0_strides_1),
  .output_addr_gen_0_strides_2(sram_only_output_addr_gen_0_strides_2),
  .output_addr_gen_0_strides_3(sram_only_output_addr_gen_0_strides_3),
  .output_addr_gen_0_strides_4(sram_only_output_addr_gen_0_strides_4),
  .output_addr_gen_0_strides_5(sram_only_output_addr_gen_0_strides_5),
  .output_addr_gen_1_starting_addr(sram_only_output_addr_gen_1_starting_addr),
  .output_addr_gen_1_strides_0(sram_only_output_addr_gen_1_strides_0),
  .output_addr_gen_1_strides_1(sram_only_output_addr_gen_1_strides_1),
  .output_addr_gen_1_strides_2(sram_only_output_addr_gen_1_strides_2),
  .output_addr_gen_1_strides_3(sram_only_output_addr_gen_1_strides_3),
  .output_addr_gen_1_strides_4(sram_only_output_addr_gen_1_strides_4),
  .output_addr_gen_1_strides_5(sram_only_output_addr_gen_1_strides_5),
  .rst_n(rst_n),
  .sram_read_addr_in(agg_sram_shared_agg_sram_shared_addr_out),
  .t_read(sram_only_t_read),
  .addr_to_sram(addr_out),
  .data_to_sram(data_to_strg),
  .sram_read_addr_out(agg_sram_shared_sram_read_addr_in),
  .wen_to_sram(wen_to_strg)
);

strg_ub_sram_tb_shared sram_tb_shared (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .flush(flush),
  .loops_buf2out_autovec_read_0_dimensionality(sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality),
  .loops_buf2out_autovec_read_0_ranges_0(sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0),
  .loops_buf2out_autovec_read_0_ranges_1(sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1),
  .loops_buf2out_autovec_read_0_ranges_2(sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2),
  .loops_buf2out_autovec_read_0_ranges_3(sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3),
  .loops_buf2out_autovec_read_0_ranges_4(sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4),
  .loops_buf2out_autovec_read_0_ranges_5(sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5),
  .loops_buf2out_autovec_read_1_dimensionality(sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality),
  .loops_buf2out_autovec_read_1_ranges_0(sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0),
  .loops_buf2out_autovec_read_1_ranges_1(sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1),
  .loops_buf2out_autovec_read_1_ranges_2(sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2),
  .loops_buf2out_autovec_read_1_ranges_3(sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3),
  .loops_buf2out_autovec_read_1_ranges_4(sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4),
  .loops_buf2out_autovec_read_1_ranges_5(sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5),
  .output_sched_gen_0_enable(sram_tb_shared_output_sched_gen_0_enable),
  .output_sched_gen_0_sched_addr_gen_delay(sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay),
  .output_sched_gen_0_sched_addr_gen_starting_addr(sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr),
  .output_sched_gen_0_sched_addr_gen_strides_0(sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0),
  .output_sched_gen_0_sched_addr_gen_strides_1(sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1),
  .output_sched_gen_0_sched_addr_gen_strides_2(sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2),
  .output_sched_gen_0_sched_addr_gen_strides_3(sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3),
  .output_sched_gen_0_sched_addr_gen_strides_4(sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4),
  .output_sched_gen_0_sched_addr_gen_strides_5(sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5),
  .output_sched_gen_1_enable(sram_tb_shared_output_sched_gen_1_enable),
  .output_sched_gen_1_sched_addr_gen_delay(sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay),
  .output_sched_gen_1_sched_addr_gen_starting_addr(sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr),
  .output_sched_gen_1_sched_addr_gen_strides_0(sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0),
  .output_sched_gen_1_sched_addr_gen_strides_1(sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1),
  .output_sched_gen_1_sched_addr_gen_strides_2(sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2),
  .output_sched_gen_1_sched_addr_gen_strides_3(sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3),
  .output_sched_gen_1_sched_addr_gen_strides_4(sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4),
  .output_sched_gen_1_sched_addr_gen_strides_5(sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5),
  .rst_n(rst_n),
  .loops_sram2tb_mux_sel(sram_tb_shared_loops_sram2tb_mux_sel),
  .loops_sram2tb_restart(sram_tb_shared_loops_sram2tb_restart),
  .sram_read_d(agg_sram_shared_sram_read_d_in),
  .t_read_out(sram_tb_shared_t_read_out)
);

strg_ub_tb_only tb_only (
  .clk(clk),
  .clk_en(clk_en),
  .cycle_count(cycle_count),
  .flush(flush),
  .loops_buf2out_read_0_dimensionality(tb_only_loops_buf2out_read_0_dimensionality),
  .loops_buf2out_read_0_ranges_0(tb_only_loops_buf2out_read_0_ranges_0),
  .loops_buf2out_read_0_ranges_1(tb_only_loops_buf2out_read_0_ranges_1),
  .loops_buf2out_read_0_ranges_2(tb_only_loops_buf2out_read_0_ranges_2),
  .loops_buf2out_read_0_ranges_3(tb_only_loops_buf2out_read_0_ranges_3),
  .loops_buf2out_read_0_ranges_4(tb_only_loops_buf2out_read_0_ranges_4),
  .loops_buf2out_read_0_ranges_5(tb_only_loops_buf2out_read_0_ranges_5),
  .loops_buf2out_read_1_dimensionality(tb_only_loops_buf2out_read_1_dimensionality),
  .loops_buf2out_read_1_ranges_0(tb_only_loops_buf2out_read_1_ranges_0),
  .loops_buf2out_read_1_ranges_1(tb_only_loops_buf2out_read_1_ranges_1),
  .loops_buf2out_read_1_ranges_2(tb_only_loops_buf2out_read_1_ranges_2),
  .loops_buf2out_read_1_ranges_3(tb_only_loops_buf2out_read_1_ranges_3),
  .loops_buf2out_read_1_ranges_4(tb_only_loops_buf2out_read_1_ranges_4),
  .loops_buf2out_read_1_ranges_5(tb_only_loops_buf2out_read_1_ranges_5),
  .loops_sram2tb_mux_sel(sram_tb_shared_loops_sram2tb_mux_sel),
  .loops_sram2tb_restart(sram_tb_shared_loops_sram2tb_restart),
  .rst_n(rst_n),
  .shared_tb_0(tb_only_shared_tb_0),
  .sram_read_data(data_from_strg),
  .t_read(sram_tb_shared_t_read_out),
  .tb_read_addr_gen_0_starting_addr(tb_only_tb_read_addr_gen_0_starting_addr),
  .tb_read_addr_gen_0_strides_0(tb_only_tb_read_addr_gen_0_strides_0),
  .tb_read_addr_gen_0_strides_1(tb_only_tb_read_addr_gen_0_strides_1),
  .tb_read_addr_gen_0_strides_2(tb_only_tb_read_addr_gen_0_strides_2),
  .tb_read_addr_gen_0_strides_3(tb_only_tb_read_addr_gen_0_strides_3),
  .tb_read_addr_gen_0_strides_4(tb_only_tb_read_addr_gen_0_strides_4),
  .tb_read_addr_gen_0_strides_5(tb_only_tb_read_addr_gen_0_strides_5),
  .tb_read_addr_gen_1_starting_addr(tb_only_tb_read_addr_gen_1_starting_addr),
  .tb_read_addr_gen_1_strides_0(tb_only_tb_read_addr_gen_1_strides_0),
  .tb_read_addr_gen_1_strides_1(tb_only_tb_read_addr_gen_1_strides_1),
  .tb_read_addr_gen_1_strides_2(tb_only_tb_read_addr_gen_1_strides_2),
  .tb_read_addr_gen_1_strides_3(tb_only_tb_read_addr_gen_1_strides_3),
  .tb_read_addr_gen_1_strides_4(tb_only_tb_read_addr_gen_1_strides_4),
  .tb_read_addr_gen_1_strides_5(tb_only_tb_read_addr_gen_1_strides_5),
  .tb_read_sched_gen_0_enable(tb_only_tb_read_sched_gen_0_enable),
  .tb_read_sched_gen_0_sched_addr_gen_delay(tb_only_tb_read_sched_gen_0_sched_addr_gen_delay),
  .tb_read_sched_gen_0_sched_addr_gen_starting_addr(tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr),
  .tb_read_sched_gen_0_sched_addr_gen_strides_0(tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0),
  .tb_read_sched_gen_0_sched_addr_gen_strides_1(tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1),
  .tb_read_sched_gen_0_sched_addr_gen_strides_2(tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2),
  .tb_read_sched_gen_0_sched_addr_gen_strides_3(tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3),
  .tb_read_sched_gen_0_sched_addr_gen_strides_4(tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4),
  .tb_read_sched_gen_0_sched_addr_gen_strides_5(tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5),
  .tb_read_sched_gen_1_enable(tb_only_tb_read_sched_gen_1_enable),
  .tb_read_sched_gen_1_sched_addr_gen_delay(tb_only_tb_read_sched_gen_1_sched_addr_gen_delay),
  .tb_read_sched_gen_1_sched_addr_gen_starting_addr(tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr),
  .tb_read_sched_gen_1_sched_addr_gen_strides_0(tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0),
  .tb_read_sched_gen_1_sched_addr_gen_strides_1(tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1),
  .tb_read_sched_gen_1_sched_addr_gen_strides_2(tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2),
  .tb_read_sched_gen_1_sched_addr_gen_strides_3(tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3),
  .tb_read_sched_gen_1_sched_addr_gen_strides_4(tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4),
  .tb_read_sched_gen_1_sched_addr_gen_strides_5(tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5),
  .tb_write_addr_gen_0_starting_addr(tb_only_tb_write_addr_gen_0_starting_addr),
  .tb_write_addr_gen_0_strides_0(tb_only_tb_write_addr_gen_0_strides_0),
  .tb_write_addr_gen_0_strides_1(tb_only_tb_write_addr_gen_0_strides_1),
  .tb_write_addr_gen_0_strides_2(tb_only_tb_write_addr_gen_0_strides_2),
  .tb_write_addr_gen_0_strides_3(tb_only_tb_write_addr_gen_0_strides_3),
  .tb_write_addr_gen_0_strides_4(tb_only_tb_write_addr_gen_0_strides_4),
  .tb_write_addr_gen_0_strides_5(tb_only_tb_write_addr_gen_0_strides_5),
  .tb_write_addr_gen_1_starting_addr(tb_only_tb_write_addr_gen_1_starting_addr),
  .tb_write_addr_gen_1_strides_0(tb_only_tb_write_addr_gen_1_strides_0),
  .tb_write_addr_gen_1_strides_1(tb_only_tb_write_addr_gen_1_strides_1),
  .tb_write_addr_gen_1_strides_2(tb_only_tb_write_addr_gen_1_strides_2),
  .tb_write_addr_gen_1_strides_3(tb_only_tb_write_addr_gen_1_strides_3),
  .tb_write_addr_gen_1_strides_4(tb_only_tb_write_addr_gen_1_strides_4),
  .tb_write_addr_gen_1_strides_5(tb_only_tb_write_addr_gen_1_strides_5),
  .accessor_output(accessor_output_int),
  .data_out(data_out_int),
  .tb_read_addr_d_out(agg_only_tb_read_addr_d_in),
  .tb_read_d_out(agg_only_tb_read_d_in)
);

Chain_2_16 chain (
  .accessor_output(accessor_output_int),
  .chain_data_in(chain_data_in_thin),
  .chain_en(chain_chain_en),
  .clk_en(clk_en),
  .curr_tile_data_out(data_out_int),
  .flush(flush),
  .data_out_tile(data_out_int_thin)
);

endmodule   // strg_ub_vec

module strg_ub_vec_flat (
  input logic [0:0] [16:0] chain_data_in_f_0,
  input logic [0:0] [16:0] chain_data_in_f_1,
  input logic clk,
  input logic clk_en,
  input logic [0:0] [16:0] data_in_f_0,
  input logic [0:0] [16:0] data_in_f_1,
  input logic flush,
  input logic rst_n,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_starting_addr,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_0,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_1,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_2,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_starting_addr,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_0,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_1,
  input logic [2:0] strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_2,
  input logic strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_enable,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2,
  input logic strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_enable,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2,
  input logic [2:0] strg_ub_vec_inst_agg_only_loops_in2buf_0_dimensionality,
  input logic [10:0] strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_0,
  input logic [10:0] strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_1,
  input logic [10:0] strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_2,
  input logic [2:0] strg_ub_vec_inst_agg_only_loops_in2buf_1_dimensionality,
  input logic [10:0] strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_0,
  input logic [10:0] strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_1,
  input logic [10:0] strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_2,
  input logic [7:0] strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_0_agg_read_padding,
  input logic [7:0] strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_1_agg_read_padding,
  input logic [8:0] strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr,
  input logic [8:0] strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr,
  input logic [1:0] strg_ub_vec_inst_agg_sram_shared_mode_0,
  input logic [1:0] strg_ub_vec_inst_agg_sram_shared_mode_1,
  input logic strg_ub_vec_inst_chain_chain_en,
  input logic [3:0] [15:0] strg_ub_vec_inst_data_from_strg_lifted,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_starting_addr,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_0,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_1,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_2,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_3,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_4,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_5,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_starting_addr,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_0,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_1,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_2,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_3,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_4,
  input logic [8:0] strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_5,
  input logic [3:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5,
  input logic [3:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4,
  input logic [10:0] strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5,
  input logic strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_enable,
  input logic [9:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5,
  input logic strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_enable,
  input logic [9:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4,
  input logic [15:0] strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5,
  input logic [3:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_dimensionality,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_0,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_1,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_2,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_3,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_4,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_5,
  input logic [3:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_dimensionality,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_0,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_1,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_2,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_3,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_4,
  input logic [10:0] strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_5,
  input logic strg_ub_vec_inst_tb_only_shared_tb_0,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_starting_addr,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_0,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_1,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_2,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_3,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_4,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_5,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_starting_addr,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_0,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_1,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_2,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_3,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_4,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_5,
  input logic strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_enable,
  input logic [9:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_delay,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5,
  input logic strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_enable,
  input logic [9:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_delay,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4,
  input logic [15:0] strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_starting_addr,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_0,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_1,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_2,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_3,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_4,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_5,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_starting_addr,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_0,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_1,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_2,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_3,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_4,
  input logic [3:0] strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_5,
  output logic accessor_output_f_b_0,
  output logic accessor_output_f_b_1,
  output logic [0:0] [16:0] data_out_f_0,
  output logic [0:0] [16:0] data_out_f_1,
  output logic [8:0] strg_ub_vec_inst_addr_out_lifted,
  output logic [3:0] [15:0] strg_ub_vec_inst_data_to_strg_lifted,
  output logic strg_ub_vec_inst_ren_to_strg_lifted,
  output logic strg_ub_vec_inst_wen_to_strg_lifted
);

logic [1:0] strg_ub_vec_inst_accessor_output;
logic [1:0][16:0] strg_ub_vec_inst_chain_data_in;
logic [1:0][16:0] strg_ub_vec_inst_data_in;
logic [1:0][16:0] strg_ub_vec_inst_data_out;
assign strg_ub_vec_inst_data_in[0] = data_in_f_0;
assign strg_ub_vec_inst_data_in[1] = data_in_f_1;
assign strg_ub_vec_inst_chain_data_in[0] = chain_data_in_f_0;
assign strg_ub_vec_inst_chain_data_in[1] = chain_data_in_f_1;
assign data_out_f_0 = strg_ub_vec_inst_data_out[0];
assign data_out_f_1 = strg_ub_vec_inst_data_out[1];
assign accessor_output_f_b_0 = strg_ub_vec_inst_accessor_output[0];
assign accessor_output_f_b_1 = strg_ub_vec_inst_accessor_output[1];
strg_ub_vec strg_ub_vec_inst (
  .agg_only_agg_write_addr_gen_0_starting_addr(strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_starting_addr),
  .agg_only_agg_write_addr_gen_0_strides_0(strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_0),
  .agg_only_agg_write_addr_gen_0_strides_1(strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_1),
  .agg_only_agg_write_addr_gen_0_strides_2(strg_ub_vec_inst_agg_only_agg_write_addr_gen_0_strides_2),
  .agg_only_agg_write_addr_gen_1_starting_addr(strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_starting_addr),
  .agg_only_agg_write_addr_gen_1_strides_0(strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_0),
  .agg_only_agg_write_addr_gen_1_strides_1(strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_1),
  .agg_only_agg_write_addr_gen_1_strides_2(strg_ub_vec_inst_agg_only_agg_write_addr_gen_1_strides_2),
  .agg_only_agg_write_sched_gen_0_enable(strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_enable),
  .agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr(strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_starting_addr),
  .agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0(strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_0),
  .agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1(strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_1),
  .agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2(strg_ub_vec_inst_agg_only_agg_write_sched_gen_0_sched_addr_gen_strides_2),
  .agg_only_agg_write_sched_gen_1_enable(strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_enable),
  .agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr(strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_starting_addr),
  .agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0(strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_0),
  .agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1(strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_1),
  .agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2(strg_ub_vec_inst_agg_only_agg_write_sched_gen_1_sched_addr_gen_strides_2),
  .agg_only_loops_in2buf_0_dimensionality(strg_ub_vec_inst_agg_only_loops_in2buf_0_dimensionality),
  .agg_only_loops_in2buf_0_ranges_0(strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_0),
  .agg_only_loops_in2buf_0_ranges_1(strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_1),
  .agg_only_loops_in2buf_0_ranges_2(strg_ub_vec_inst_agg_only_loops_in2buf_0_ranges_2),
  .agg_only_loops_in2buf_1_dimensionality(strg_ub_vec_inst_agg_only_loops_in2buf_1_dimensionality),
  .agg_only_loops_in2buf_1_ranges_0(strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_0),
  .agg_only_loops_in2buf_1_ranges_1(strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_1),
  .agg_only_loops_in2buf_1_ranges_2(strg_ub_vec_inst_agg_only_loops_in2buf_1_ranges_2),
  .agg_sram_shared_agg_read_sched_gen_0_agg_read_padding(strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_0_agg_read_padding),
  .agg_sram_shared_agg_read_sched_gen_1_agg_read_padding(strg_ub_vec_inst_agg_sram_shared_agg_read_sched_gen_1_agg_read_padding),
  .agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr(strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_0_starting_addr),
  .agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr(strg_ub_vec_inst_agg_sram_shared_agg_sram_shared_addr_gen_1_starting_addr),
  .agg_sram_shared_mode_0(strg_ub_vec_inst_agg_sram_shared_mode_0),
  .agg_sram_shared_mode_1(strg_ub_vec_inst_agg_sram_shared_mode_1),
  .chain_chain_en(strg_ub_vec_inst_chain_chain_en),
  .chain_data_in(strg_ub_vec_inst_chain_data_in),
  .clk(clk),
  .clk_en(clk_en),
  .data_from_strg(strg_ub_vec_inst_data_from_strg_lifted),
  .data_in(strg_ub_vec_inst_data_in),
  .flush(flush),
  .rst_n(rst_n),
  .sram_only_output_addr_gen_0_starting_addr(strg_ub_vec_inst_sram_only_output_addr_gen_0_starting_addr),
  .sram_only_output_addr_gen_0_strides_0(strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_0),
  .sram_only_output_addr_gen_0_strides_1(strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_1),
  .sram_only_output_addr_gen_0_strides_2(strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_2),
  .sram_only_output_addr_gen_0_strides_3(strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_3),
  .sram_only_output_addr_gen_0_strides_4(strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_4),
  .sram_only_output_addr_gen_0_strides_5(strg_ub_vec_inst_sram_only_output_addr_gen_0_strides_5),
  .sram_only_output_addr_gen_1_starting_addr(strg_ub_vec_inst_sram_only_output_addr_gen_1_starting_addr),
  .sram_only_output_addr_gen_1_strides_0(strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_0),
  .sram_only_output_addr_gen_1_strides_1(strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_1),
  .sram_only_output_addr_gen_1_strides_2(strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_2),
  .sram_only_output_addr_gen_1_strides_3(strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_3),
  .sram_only_output_addr_gen_1_strides_4(strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_4),
  .sram_only_output_addr_gen_1_strides_5(strg_ub_vec_inst_sram_only_output_addr_gen_1_strides_5),
  .sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_dimensionality),
  .sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_0),
  .sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_1),
  .sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_2),
  .sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_3),
  .sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_4),
  .sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_0_ranges_5),
  .sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_dimensionality),
  .sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_0),
  .sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_1),
  .sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_2),
  .sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_3),
  .sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_4),
  .sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5(strg_ub_vec_inst_sram_tb_shared_loops_buf2out_autovec_read_1_ranges_5),
  .sram_tb_shared_output_sched_gen_0_enable(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_enable),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_delay),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_starting_addr),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_0),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_1),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_2),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_3),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_4),
  .sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_0_sched_addr_gen_strides_5),
  .sram_tb_shared_output_sched_gen_1_enable(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_enable),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_delay),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_starting_addr),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_0),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_1),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_2),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_3),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_4),
  .sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5(strg_ub_vec_inst_sram_tb_shared_output_sched_gen_1_sched_addr_gen_strides_5),
  .tb_only_loops_buf2out_read_0_dimensionality(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_dimensionality),
  .tb_only_loops_buf2out_read_0_ranges_0(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_0),
  .tb_only_loops_buf2out_read_0_ranges_1(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_1),
  .tb_only_loops_buf2out_read_0_ranges_2(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_2),
  .tb_only_loops_buf2out_read_0_ranges_3(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_3),
  .tb_only_loops_buf2out_read_0_ranges_4(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_4),
  .tb_only_loops_buf2out_read_0_ranges_5(strg_ub_vec_inst_tb_only_loops_buf2out_read_0_ranges_5),
  .tb_only_loops_buf2out_read_1_dimensionality(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_dimensionality),
  .tb_only_loops_buf2out_read_1_ranges_0(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_0),
  .tb_only_loops_buf2out_read_1_ranges_1(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_1),
  .tb_only_loops_buf2out_read_1_ranges_2(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_2),
  .tb_only_loops_buf2out_read_1_ranges_3(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_3),
  .tb_only_loops_buf2out_read_1_ranges_4(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_4),
  .tb_only_loops_buf2out_read_1_ranges_5(strg_ub_vec_inst_tb_only_loops_buf2out_read_1_ranges_5),
  .tb_only_shared_tb_0(strg_ub_vec_inst_tb_only_shared_tb_0),
  .tb_only_tb_read_addr_gen_0_starting_addr(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_starting_addr),
  .tb_only_tb_read_addr_gen_0_strides_0(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_0),
  .tb_only_tb_read_addr_gen_0_strides_1(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_1),
  .tb_only_tb_read_addr_gen_0_strides_2(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_2),
  .tb_only_tb_read_addr_gen_0_strides_3(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_3),
  .tb_only_tb_read_addr_gen_0_strides_4(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_4),
  .tb_only_tb_read_addr_gen_0_strides_5(strg_ub_vec_inst_tb_only_tb_read_addr_gen_0_strides_5),
  .tb_only_tb_read_addr_gen_1_starting_addr(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_starting_addr),
  .tb_only_tb_read_addr_gen_1_strides_0(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_0),
  .tb_only_tb_read_addr_gen_1_strides_1(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_1),
  .tb_only_tb_read_addr_gen_1_strides_2(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_2),
  .tb_only_tb_read_addr_gen_1_strides_3(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_3),
  .tb_only_tb_read_addr_gen_1_strides_4(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_4),
  .tb_only_tb_read_addr_gen_1_strides_5(strg_ub_vec_inst_tb_only_tb_read_addr_gen_1_strides_5),
  .tb_only_tb_read_sched_gen_0_enable(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_enable),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_delay(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_delay),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_starting_addr),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_0),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_1),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_2),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_3),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_4),
  .tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5(strg_ub_vec_inst_tb_only_tb_read_sched_gen_0_sched_addr_gen_strides_5),
  .tb_only_tb_read_sched_gen_1_enable(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_enable),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_delay(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_delay),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_starting_addr),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_0),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_1),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_2),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_3),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_4),
  .tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5(strg_ub_vec_inst_tb_only_tb_read_sched_gen_1_sched_addr_gen_strides_5),
  .tb_only_tb_write_addr_gen_0_starting_addr(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_starting_addr),
  .tb_only_tb_write_addr_gen_0_strides_0(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_0),
  .tb_only_tb_write_addr_gen_0_strides_1(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_1),
  .tb_only_tb_write_addr_gen_0_strides_2(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_2),
  .tb_only_tb_write_addr_gen_0_strides_3(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_3),
  .tb_only_tb_write_addr_gen_0_strides_4(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_4),
  .tb_only_tb_write_addr_gen_0_strides_5(strg_ub_vec_inst_tb_only_tb_write_addr_gen_0_strides_5),
  .tb_only_tb_write_addr_gen_1_starting_addr(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_starting_addr),
  .tb_only_tb_write_addr_gen_1_strides_0(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_0),
  .tb_only_tb_write_addr_gen_1_strides_1(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_1),
  .tb_only_tb_write_addr_gen_1_strides_2(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_2),
  .tb_only_tb_write_addr_gen_1_strides_3(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_3),
  .tb_only_tb_write_addr_gen_1_strides_4(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_4),
  .tb_only_tb_write_addr_gen_1_strides_5(strg_ub_vec_inst_tb_only_tb_write_addr_gen_1_strides_5),
  .accessor_output(strg_ub_vec_inst_accessor_output),
  .addr_out(strg_ub_vec_inst_addr_out_lifted),
  .data_out(strg_ub_vec_inst_data_out),
  .data_to_strg(strg_ub_vec_inst_data_to_strg_lifted),
  .ren_to_strg(strg_ub_vec_inst_ren_to_strg_lifted),
  .wen_to_strg(strg_ub_vec_inst_wen_to_strg_lifted)
);

endmodule   // strg_ub_vec_flat

module write_scanner (
  input logic ID_out_ready,
  input logic [16:0] addr_in,
  input logic addr_in_valid,
  input logic addr_out_ready,
  input logic block_mode,
  input logic [16:0] block_wr_in,
  input logic block_wr_in_valid,
  input logic clk,
  input logic clk_en,
  input logic compressed,
  input logic [16:0] data_in,
  input logic data_in_valid,
  input logic data_out_ready,
  input logic flush,
  input logic init_blank,
  input logic lowest_level,
  input logic rst_n,
  input logic [15:0] stop_lvl,
  input logic tile_en,
  input logic vector_reduce_mode,
  output logic [16:0] ID_out,
  output logic ID_out_valid,
  output logic addr_in_ready,
  output logic [16:0] addr_out,
  output logic addr_out_valid,
  output logic block_wr_in_ready,
  output logic data_in_ready,
  output logic [16:0] data_out,
  output logic data_out_valid
);

typedef enum logic[3:0] {
  ALLOCATE1 = 4'h0,
  ALLOCATE2 = 4'h1,
  BLOCK_1_SZ = 4'h2,
  BLOCK_1_WR = 4'h3,
  BLOCK_2_SZ = 4'h4,
  BLOCK_2_WR = 4'h5,
  ComLL = 4'h6,
  DONE = 4'h7,
  FINALIZE1 = 4'h8,
  FINALIZE2 = 4'h9,
  LL = 4'hA,
  START = 4'hB,
  UL = 4'hC,
  UL_EMIT = 4'hD,
  UL_WZ = 4'hE,
  UnLL = 4'hF
} scan_seq_state;
logic [0:0][16:0] ID_out_fifo_data_in;
logic ID_out_fifo_empty;
logic ID_out_fifo_full;
logic ID_out_fifo_push;
logic [15:0] ID_to_fifo;
logic IN_DONE;
logic addr_done_in;
logic [15:0] addr_infifo_data_in;
logic addr_infifo_eos_in;
logic [16:0] addr_infifo_in_packed;
logic [16:0] addr_infifo_out_packed;
logic addr_infifo_valid_in;
logic addr_input_fifo_empty;
logic addr_input_fifo_full;
logic [0:0][16:0] addr_out_fifo_data_in;
logic addr_out_fifo_empty;
logic addr_out_fifo_full;
logic addr_out_fifo_push;
logic [15:0] addr_to_fifo;
logic blank_done_stick_sticky;
logic blank_done_stick_was_high;
logic [15:0] block_size;
logic block_wr_fifo_valid;
logic [0:0][15:0] block_wr_input_fifo_data_out;
logic block_wr_input_fifo_empty;
logic block_wr_input_fifo_full;
logic [15:0] block_write_count;
logic clr_blank_done;
logic clr_block_write;
logic clr_coord_addr;
logic clr_curr_coord;
logic clr_seg_addr;
logic clr_seg_ctr;
logic clr_wen_made;
logic [15:0] coord_addr;
logic data_done_in;
logic [15:0] data_infifo_data_in;
logic [15:0] data_infifo_data_in_d1;
logic data_infifo_eos_in;
logic [16:0] data_infifo_in_packed;
logic [16:0] data_infifo_out_packed;
logic data_infifo_valid_in;
logic data_input_fifo_empty;
logic data_input_fifo_full;
logic [0:0][16:0] data_out_fifo_data_in;
logic data_out_fifo_empty;
logic data_out_fifo_full;
logic data_out_fifo_push;
logic [15:0] data_to_fifo;
logic [16:0] done_token;
logic gclk;
logic inc_block_write;
logic inc_coord_addr;
logic inc_seg_addr;
logic inc_seg_ctr;
logic [1:0] infifo_pop;
logic new_coord;
logic op_to_fifo;
logic pop_block_wr;
logic push_to_outs;
scan_seq_state scan_seq_current_state;
scan_seq_state scan_seq_next_state;
logic [15:0] segment_addr;
logic [15:0] segment_counter;
logic [16:0] semi_done_token;
logic set_blank_done;
logic set_block_size;
logic set_curr_coord;
logic stop_in;
logic valid_coord_sticky_sticky;
logic valid_coord_sticky_was_high;
logic [16:0] vr_fsm_wr_scan_data_in;
logic wen_made_sticky;
logic wen_made_was_high;
assign gclk = clk & tile_en;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    blank_done_stick_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      blank_done_stick_was_high <= 1'h0;
    end
    else if (clr_blank_done) begin
      blank_done_stick_was_high <= 1'h0;
    end
    else if (set_blank_done) begin
      blank_done_stick_was_high <= 1'h1;
    end
  end
end
assign blank_done_stick_sticky = blank_done_stick_was_high;
assign done_token = {1'h1, 7'h0, 1'h1, 8'h0};
assign semi_done_token = {1'h1, 11'h0, 1'h1, 4'h0};
assign vr_fsm_wr_scan_data_in = (data_in == semi_done_token) ? done_token: data_in;
assign data_infifo_in_packed = vector_reduce_mode ? vr_fsm_wr_scan_data_in: data_in;
assign data_infifo_eos_in = data_infifo_out_packed[16];
assign data_infifo_data_in = data_infifo_out_packed[15:0];
assign data_in_ready = ~data_input_fifo_full;
assign data_infifo_valid_in = ~data_input_fifo_empty;
assign addr_infifo_in_packed[16] = addr_in[16];
assign addr_infifo_in_packed[15:0] = addr_in[15:0];
assign addr_infifo_eos_in = addr_infifo_out_packed[16];
assign addr_infifo_data_in = addr_infifo_out_packed[15:0];
assign addr_in_ready = ~addr_input_fifo_full;
assign addr_infifo_valid_in = ~addr_input_fifo_empty;
assign block_wr_in_ready = ~block_wr_input_fifo_full;
assign block_wr_fifo_valid = ~block_wr_input_fifo_empty;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    block_size <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      block_size <= 16'h0;
    end
    else if (1'h0) begin
      block_size <= 16'h0;
    end
    else if (set_block_size) begin
      block_size <= block_wr_input_fifo_data_out;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    block_write_count <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      block_write_count <= 16'h0;
    end
    else if (clr_block_write) begin
      block_write_count <= 16'h0;
    end
    else if (inc_block_write) begin
      block_write_count <= block_write_count + 16'h1;
    end
  end
end
assign data_out_fifo_data_in = {op_to_fifo, data_to_fifo};
assign data_out_valid = ~data_out_fifo_empty;
assign addr_out_fifo_data_in = {1'h0, addr_to_fifo};
assign addr_out_valid = ~addr_out_fifo_empty;
assign ID_out_fifo_data_in = {1'h0, ID_to_fifo};
assign ID_out_valid = ~ID_out_fifo_empty;
assign {data_out_fifo_push, addr_out_fifo_push, ID_out_fifo_push} = {push_to_outs, push_to_outs, push_to_outs};
assign data_done_in = data_infifo_valid_in & data_infifo_eos_in & (data_infifo_data_in[9:8] == 2'h1);
assign addr_done_in = addr_infifo_valid_in & addr_infifo_eos_in & (addr_infifo_data_in[9:8] == 2'h1);

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    segment_addr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      segment_addr <= 16'h0;
    end
    else if (clr_seg_addr) begin
      segment_addr <= 16'h0;
    end
    else if (inc_seg_addr) begin
      segment_addr <= segment_addr + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    coord_addr <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      coord_addr <= 16'h0;
    end
    else if (clr_coord_addr) begin
      coord_addr <= 16'h0;
    end
    else if (inc_coord_addr) begin
      coord_addr <= coord_addr + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    segment_counter <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      segment_counter <= 16'h0;
    end
    else if (clr_seg_ctr) begin
      segment_counter <= 16'h0;
    end
    else if (inc_seg_ctr) begin
      segment_counter <= segment_counter + 16'h1;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    data_infifo_data_in_d1 <= 16'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      data_infifo_data_in_d1 <= 16'h0;
    end
    else if (1'h0) begin
      data_infifo_data_in_d1 <= 16'h0;
    end
    else if (set_curr_coord) begin
      data_infifo_data_in_d1 <= data_infifo_data_in;
    end
  end
end

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    valid_coord_sticky_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      valid_coord_sticky_was_high <= 1'h0;
    end
    else if (clr_curr_coord) begin
      valid_coord_sticky_was_high <= 1'h0;
    end
    else if (set_curr_coord) begin
      valid_coord_sticky_was_high <= 1'h1;
    end
  end
end
assign valid_coord_sticky_sticky = valid_coord_sticky_was_high;
assign new_coord = data_infifo_valid_in & (~data_infifo_eos_in) & ((~valid_coord_sticky_sticky) |
    (data_infifo_data_in != data_infifo_data_in_d1));
assign stop_in = data_infifo_valid_in & data_infifo_eos_in;

always_ff @(posedge clk, negedge rst_n) begin
  if (~rst_n) begin
    wen_made_was_high <= 1'h0;
  end
  else if (clk_en) begin
    if (flush) begin
      wen_made_was_high <= 1'h0;
    end
    else if (clr_wen_made) begin
      wen_made_was_high <= 1'h0;
    end
    else if (push_to_outs) begin
      wen_made_was_high <= 1'h1;
    end
  end
end
assign wen_made_sticky = wen_made_was_high;

always_ff @(posedge clk, negedge rst_n) begin
  if (!rst_n) begin
    scan_seq_current_state <= START;
  end
  else if (clk_en) begin
    if (flush) begin
      scan_seq_current_state <= START;
    end
    else scan_seq_current_state <= scan_seq_next_state;
  end
end
always_comb begin
  scan_seq_next_state = scan_seq_current_state;
  unique case (scan_seq_current_state)
    ALLOCATE1: begin
        if (~(&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = ALLOCATE1;
        end
        else if ((~lowest_level) & (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = ALLOCATE2;
        end
        else if (lowest_level & block_mode & (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = BLOCK_1_SZ;
        end
        else if (lowest_level & (~block_mode) & (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = LL;
        end
      end
    ALLOCATE2: begin
        if (~(&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = ALLOCATE2;
        end
        else if (block_mode & (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = BLOCK_1_SZ;
        end
        else if ((~block_mode) & (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = UL_WZ;
        end
      end
    BLOCK_1_SZ: begin
        if (block_wr_fifo_valid) begin
          scan_seq_next_state = BLOCK_1_WR;
        end
        else scan_seq_next_state = BLOCK_1_SZ;
      end
    BLOCK_1_WR: begin
        if ((block_write_count == block_size) & (~lowest_level)) begin
          scan_seq_next_state = BLOCK_2_SZ;
        end
        else if ((block_write_count == block_size) & lowest_level) begin
          scan_seq_next_state = FINALIZE2;
        end
        else scan_seq_next_state = BLOCK_1_WR;
      end
    BLOCK_2_SZ: begin
        if (block_wr_fifo_valid) begin
          scan_seq_next_state = BLOCK_2_WR;
        end
        else scan_seq_next_state = BLOCK_2_SZ;
      end
    BLOCK_2_WR: begin
        if (block_write_count == block_size) begin
          scan_seq_next_state = FINALIZE1;
        end
        else scan_seq_next_state = BLOCK_2_WR;
      end
    ComLL: begin
        if (data_done_in) begin
          scan_seq_next_state = FINALIZE2;
        end
        else scan_seq_next_state = ComLL;
      end
    DONE: scan_seq_next_state = START;
    FINALIZE1: begin
        if (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full})) begin
          scan_seq_next_state = FINALIZE2;
        end
        else scan_seq_next_state = FINALIZE1;
      end
    FINALIZE2: begin
        if (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full})) begin
          scan_seq_next_state = DONE;
        end
        else scan_seq_next_state = FINALIZE2;
      end
    LL: begin
        if (init_blank & (~blank_done_stick_sticky)) begin
          scan_seq_next_state = FINALIZE2;
        end
        else if (compressed & ((~init_blank) | blank_done_stick_sticky)) begin
          scan_seq_next_state = ComLL;
        end
        else if ((~compressed) & ((~init_blank) | blank_done_stick_sticky)) begin
          scan_seq_next_state = UnLL;
        end
      end
    START: begin
        if (tile_en) begin
          scan_seq_next_state = ALLOCATE1;
        end
        else scan_seq_next_state = START;
      end
    UL: begin
        if (data_infifo_valid_in) begin
          scan_seq_next_state = UL_EMIT;
        end
        else scan_seq_next_state = UL;
      end
    UL_EMIT: begin
        if (data_done_in) begin
          scan_seq_next_state = FINALIZE1;
        end
        else scan_seq_next_state = UL_EMIT;
      end
    UL_WZ: begin
        if (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full})) begin
          scan_seq_next_state = UL;
        end
        else if (~(&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}))) begin
          scan_seq_next_state = UL_WZ;
        end
      end
    UnLL: begin
        if (data_done_in & addr_done_in) begin
          scan_seq_next_state = FINALIZE2;
        end
        else scan_seq_next_state = UnLL;
      end
    default: begin end
  endcase
end
always_comb begin
  unique case (scan_seq_current_state)
    ALLOCATE1: begin :scan_seq_ALLOCATE1_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h1;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_ALLOCATE1_Output
    ALLOCATE2: begin :scan_seq_ALLOCATE2_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h1;
        push_to_outs = 1'h1;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_ALLOCATE2_Output
    BLOCK_1_SZ: begin :scan_seq_BLOCK_1_SZ_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h1;
        push_to_outs = 1'h0;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = block_wr_fifo_valid;
        inc_block_write = 1'h0;
        clr_block_write = 1'h1;
        pop_block_wr = block_wr_fifo_valid;
        IN_DONE = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_BLOCK_1_SZ_Output
    BLOCK_1_WR: begin :scan_seq_BLOCK_1_WR_Output
        data_to_fifo = block_wr_input_fifo_data_out;
        op_to_fifo = 1'h1;
        addr_to_fifo = block_write_count;
        ID_to_fifo = 16'h0;
        push_to_outs = block_wr_fifo_valid & (block_write_count < block_size);
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = block_wr_fifo_valid & (block_write_count < block_size) &
            (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}));
        clr_block_write = 1'h0;
        pop_block_wr = block_wr_fifo_valid & (block_write_count < block_size) &
            (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}));
        IN_DONE = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_BLOCK_1_WR_Output
    BLOCK_2_SZ: begin :scan_seq_BLOCK_2_SZ_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h0;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = block_wr_fifo_valid;
        inc_block_write = 1'h0;
        clr_block_write = 1'h1;
        pop_block_wr = block_wr_fifo_valid;
        IN_DONE = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_BLOCK_2_SZ_Output
    BLOCK_2_WR: begin :scan_seq_BLOCK_2_WR_Output
        data_to_fifo = block_wr_input_fifo_data_out;
        op_to_fifo = 1'h1;
        addr_to_fifo = block_write_count;
        ID_to_fifo = 16'h1;
        push_to_outs = block_wr_fifo_valid & (block_write_count < block_size);
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = block_wr_fifo_valid & (block_write_count < block_size) &
            (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}));
        clr_block_write = 1'h0;
        pop_block_wr = block_wr_fifo_valid & (block_write_count < block_size) &
            (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full}));
        IN_DONE = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_BLOCK_2_WR_Output
    ComLL: begin :scan_seq_ComLL_Output
        data_to_fifo = data_infifo_data_in;
        op_to_fifo = 1'h1;
        addr_to_fifo = segment_addr;
        ID_to_fifo = 16'h0;
        push_to_outs = data_infifo_valid_in & (~data_infifo_eos_in);
        inc_seg_addr = data_infifo_valid_in & (~data_infifo_eos_in) & (&(~{data_out_fifo_full,
            addr_out_fifo_full, ID_out_fifo_full}));
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = data_infifo_valid_in & (data_infifo_eos_in | (&(~{data_out_fifo_full,
            addr_out_fifo_full, ID_out_fifo_full})));
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_ComLL_Output
    DONE: begin :scan_seq_DONE_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h0;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = data_done_in;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
        IN_DONE = 1'h1;
        pop_block_wr = 1'h0;
      end :scan_seq_DONE_Output
    FINALIZE1: begin :scan_seq_FINALIZE1_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h1;
        push_to_outs = 1'h1;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_FINALIZE1_Output
    FINALIZE2: begin :scan_seq_FINALIZE2_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h1;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_FINALIZE2_Output
    LL: begin :scan_seq_LL_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h0;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_LL_Output
    START: begin :scan_seq_START_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h0;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h0;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h1;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h1;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h1;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h1;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h1;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h1;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_START_Output
    UL: begin :scan_seq_UL_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h1;
        addr_to_fifo = 16'h0;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h0;
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = new_coord;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h1;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_UL_Output
    UL_EMIT: begin :scan_seq_UL_EMIT_Output
        data_to_fifo = stop_in ? segment_counter: data_infifo_data_in;
        op_to_fifo = 1'h1;
        addr_to_fifo = stop_in ? segment_addr: coord_addr;
        ID_to_fifo = stop_in ? 16'h0: 16'h1;
        push_to_outs = data_infifo_valid_in & (&(~{data_out_fifo_full, addr_out_fifo_full,
            ID_out_fifo_full})) & (~data_done_in);
        inc_seg_addr = stop_in & (&(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full})) &
            (~data_done_in);
        clr_seg_addr = 1'h0;
        inc_coord_addr = (~data_infifo_eos_in) & data_infifo_valid_in & (&(~{data_out_fifo_full,
            addr_out_fifo_full, ID_out_fifo_full}));
        clr_coord_addr = 1'h0;
        inc_seg_ctr = (~data_infifo_eos_in) & data_infifo_valid_in & (&(~{data_out_fifo_full,
            addr_out_fifo_full, ID_out_fifo_full}));
        clr_seg_ctr = 1'h0;
        set_curr_coord = new_coord;
        clr_curr_coord = ~wen_made_sticky;
        infifo_pop[0] = data_infifo_valid_in & (&(~{data_out_fifo_full, addr_out_fifo_full,
            ID_out_fifo_full})) & (~data_done_in);
        infifo_pop[1] = 1'h0;
        clr_wen_made = wen_made_sticky & data_infifo_valid_in;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_UL_EMIT_Output
    UL_WZ: begin :scan_seq_UL_WZ_Output
        data_to_fifo = 16'h0;
        op_to_fifo = 1'h1;
        addr_to_fifo = segment_addr;
        ID_to_fifo = 16'h0;
        push_to_outs = 1'h1;
        inc_seg_addr = &(~{data_out_fifo_full, addr_out_fifo_full, ID_out_fifo_full});
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = 1'h0;
        infifo_pop[1] = 1'h0;
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_UL_WZ_Output
    UnLL: begin :scan_seq_UnLL_Output
        data_to_fifo = data_infifo_data_in;
        op_to_fifo = 1'h1;
        addr_to_fifo = addr_infifo_data_in;
        ID_to_fifo = 16'h0;
        push_to_outs = data_infifo_valid_in & addr_infifo_valid_in & (~(data_infifo_eos_in |
            addr_infifo_eos_in));
        inc_seg_addr = 1'h0;
        clr_seg_addr = 1'h0;
        inc_coord_addr = 1'h0;
        clr_coord_addr = 1'h0;
        inc_seg_ctr = 1'h0;
        clr_seg_ctr = 1'h0;
        set_curr_coord = 1'h0;
        clr_curr_coord = 1'h0;
        infifo_pop[0] = data_infifo_valid_in & addr_infifo_valid_in & ((data_infifo_eos_in &
            addr_infifo_eos_in) | (&(~{data_out_fifo_full, addr_out_fifo_full,
            ID_out_fifo_full})));
        infifo_pop[1] = data_infifo_valid_in & addr_infifo_valid_in & ((data_infifo_eos_in &
            addr_infifo_eos_in) | (&(~{data_out_fifo_full, addr_out_fifo_full,
            ID_out_fifo_full})));
        clr_wen_made = 1'h0;
        set_block_size = 1'h0;
        inc_block_write = 1'h0;
        clr_block_write = 1'h0;
        IN_DONE = 1'h0;
        pop_block_wr = 1'h0;
        set_blank_done = 1'h0;
        clr_blank_done = 1'h0;
      end :scan_seq_UnLL_Output
    default: begin end
  endcase
end
reg_fifo_depth_2_w_17_afd_2 data_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(data_infifo_in_packed),
  .flush(flush),
  .pop(infifo_pop[0]),
  .push(data_in_valid),
  .rst_n(rst_n),
  .data_out(data_infifo_out_packed),
  .empty(data_input_fifo_empty),
  .full(data_input_fifo_full)
);

reg_fifo_depth_0_w_17_afd_2 addr_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(addr_infifo_in_packed),
  .flush(flush),
  .pop(infifo_pop[1]),
  .push(addr_in_valid),
  .rst_n(rst_n),
  .data_out(addr_infifo_out_packed),
  .empty(addr_input_fifo_empty),
  .full(addr_input_fifo_full)
);

reg_fifo_depth_0_w_16_afd_2 block_wr_input_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(block_wr_in[15:0]),
  .flush(flush),
  .pop(pop_block_wr),
  .push(block_wr_in_valid),
  .rst_n(rst_n),
  .data_out(block_wr_input_fifo_data_out),
  .empty(block_wr_input_fifo_empty),
  .full(block_wr_input_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 data_out_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(data_out_fifo_data_in),
  .flush(flush),
  .pop(data_out_ready),
  .push(data_out_fifo_push),
  .rst_n(rst_n),
  .data_out(data_out),
  .empty(data_out_fifo_empty),
  .full(data_out_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 addr_out_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(addr_out_fifo_data_in),
  .flush(flush),
  .pop(addr_out_ready),
  .push(addr_out_fifo_push),
  .rst_n(rst_n),
  .data_out(addr_out),
  .empty(addr_out_fifo_empty),
  .full(addr_out_fifo_full)
);

reg_fifo_depth_2_w_17_afd_2 ID_out_fifo (
  .clk(gclk),
  .clk_en(clk_en),
  .data_in(ID_out_fifo_data_in),
  .flush(flush),
  .pop(ID_out_ready),
  .push(ID_out_fifo_push),
  .rst_n(rst_n),
  .data_out(ID_out),
  .empty(ID_out_fifo_empty),
  .full(ID_out_fifo_full)
);

endmodule   // write_scanner

