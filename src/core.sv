`default_nettype none
`timescale 1ns/1ns

// Compute Core
// One block at a time. One fetcher/decoder/scheduler shared across THREADS_PER_BLOCK
// ALU/LSU/registers/PC lanes. Per-block shared memory (Phase 7) is instantiated here.
module core #(
    parameter DATA_MEM_ADDR_BITS    = 8,
    parameter DATA_MEM_DATA_BITS    = 8,
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16,
    parameter THREADS_PER_BLOCK     = 4,
    parameter ICACHE_LINES          = 0,
    parameter SMEM_SIZE             = 64,
    parameter PIPELINED             = 0
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              telemetry_reset,
    input  wire                              start,
    output wire                              done,

    input  wire [7:0]                        block_id,
    input  wire [$clog2(THREADS_PER_BLOCK):0] thread_count,

    output wire                              program_mem_read_valid,
    output wire [PROGRAM_MEM_ADDR_BITS-1:0]  program_mem_read_address,
    input  wire                              program_mem_read_ready,
    input  wire [PROGRAM_MEM_DATA_BITS-1:0]  program_mem_read_data,

    output wire [THREADS_PER_BLOCK-1:0]      data_mem_read_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]     data_mem_read_address  [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]      data_mem_read_ready,
    input  wire [DATA_MEM_DATA_BITS-1:0]     data_mem_read_data     [THREADS_PER_BLOCK-1:0],
    output wire [THREADS_PER_BLOCK-1:0]      data_mem_write_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]     data_mem_write_address [THREADS_PER_BLOCK-1:0],
    output wire [DATA_MEM_DATA_BITS-1:0]     data_mem_write_data    [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]      data_mem_write_ready,

    output wire [31:0]                       icache_hits,
    output wire [31:0]                       icache_misses
);
    wire [2:0]  core_state;
    wire [2:0]  fetcher_state;
    wire [15:0] instruction;

    wire [7:0]  current_pc;
    wire [7:0]  next_pc [THREADS_PER_BLOCK-1:0];
    wire [7:0]  rs      [THREADS_PER_BLOCK-1:0];
    wire [7:0]  rt      [THREADS_PER_BLOCK-1:0];
    wire [1:0]  lsu_state [THREADS_PER_BLOCK-1:0];
    wire [7:0]  lsu_out [THREADS_PER_BLOCK-1:0];
    wire [7:0]  alu_out [THREADS_PER_BLOCK-1:0];

    wire [3:0]  decoded_rd_address, decoded_rs_address, decoded_rt_address;
    wire [2:0]  decoded_nzp;
    wire [7:0]  decoded_immediate;
    wire        decoded_reg_write_enable, decoded_mem_read_enable, decoded_mem_write_enable;
    wire        decoded_nzp_write_enable;
    wire [1:0]  decoded_reg_input_mux, decoded_alu_arithmetic_mux;
    wire        decoded_alu_output_mux, decoded_pc_mux, decoded_ret;
    wire        decoded_smem_read_enable, decoded_smem_write_enable, decoded_bar;

    // Shared memory wires
    wire [THREADS_PER_BLOCK-1:0] smem_read_enable_v;
    wire [THREADS_PER_BLOCK-1:0] smem_write_enable_v;
    wire [7:0] smem_read_address  [THREADS_PER_BLOCK-1:0];
    wire [7:0] smem_read_data     [THREADS_PER_BLOCK-1:0];
    wire [7:0] smem_write_address [THREADS_PER_BLOCK-1:0];
    wire [7:0] smem_write_data    [THREADS_PER_BLOCK-1:0];

    fetcher #(
        .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
        .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS),
        .ICACHE_LINES(ICACHE_LINES)
    ) fetcher_instance (
        .clk(clk), .reset(reset), .telemetry_reset(telemetry_reset),
        .core_state(core_state), .current_pc(current_pc),
        .mem_read_valid(program_mem_read_valid),
        .mem_read_address(program_mem_read_address),
        .mem_read_ready(program_mem_read_ready),
        .mem_read_data(program_mem_read_data),
        .fetcher_state(fetcher_state),
        .instruction(instruction),
        .icache_hits(icache_hits),
        .icache_misses(icache_misses)
    );

    decoder decoder_instance (
        .clk(clk), .reset(reset),
        .core_state(core_state), .instruction(instruction),
        .decoded_rd_address(decoded_rd_address),
        .decoded_rs_address(decoded_rs_address),
        .decoded_rt_address(decoded_rt_address),
        .decoded_nzp(decoded_nzp),
        .decoded_immediate(decoded_immediate),
        .decoded_reg_write_enable(decoded_reg_write_enable),
        .decoded_mem_read_enable(decoded_mem_read_enable),
        .decoded_mem_write_enable(decoded_mem_write_enable),
        .decoded_nzp_write_enable(decoded_nzp_write_enable),
        .decoded_reg_input_mux(decoded_reg_input_mux),
        .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
        .decoded_alu_output_mux(decoded_alu_output_mux),
        .decoded_pc_mux(decoded_pc_mux),
        .decoded_ret(decoded_ret),
        .decoded_smem_read_enable(decoded_smem_read_enable),
        .decoded_smem_write_enable(decoded_smem_write_enable),
        .decoded_bar(decoded_bar)
    );

    scheduler #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
        .PIPELINED(PIPELINED)
    ) scheduler_instance (
        .clk(clk), .reset(reset), .start(start),
        .decoded_mem_read_enable(decoded_mem_read_enable),
        .decoded_mem_write_enable(decoded_mem_write_enable),
        .decoded_smem_read_enable(decoded_smem_read_enable),
        .decoded_smem_write_enable(decoded_smem_write_enable),
        .decoded_bar(decoded_bar),
        .decoded_ret(decoded_ret),
        .fetcher_state(fetcher_state),
        .lsu_state(lsu_state),
        .current_pc(current_pc), .next_pc(next_pc),
        .core_state(core_state), .done(done)
    );

    shared_memory #(
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
        .SIZE(SMEM_SIZE),
        .ADDR_BITS(8),
        .DATA_BITS(DATA_MEM_DATA_BITS)
    ) smem_instance (
        .clk(clk), .reset(reset),
        .lane_read_enable(smem_read_enable_v),
        .lane_read_addr(smem_read_address),
        .lane_read_data(smem_read_data),
        .lane_write_enable(smem_write_enable_v),
        .lane_write_addr(smem_write_address),
        .lane_write_data(smem_write_data)
    );

    genvar i;
    generate
        for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin : threads
            alu alu_instance (
                .clk(clk), .reset(reset), .enable(i < thread_count),
                .core_state(core_state),
                .decoded_alu_arithmetic_mux(decoded_alu_arithmetic_mux),
                .decoded_alu_output_mux(decoded_alu_output_mux),
                .rs(rs[i]), .rt(rt[i]),
                .alu_out(alu_out[i])
            );

            lsu lsu_instance (
                .clk(clk), .reset(reset), .enable(i < thread_count),
                .core_state(core_state),
                .decoded_mem_read_enable(decoded_mem_read_enable),
                .decoded_mem_write_enable(decoded_mem_write_enable),
                .decoded_smem_read_enable(decoded_smem_read_enable),
                .decoded_smem_write_enable(decoded_smem_write_enable),
                .rs(rs[i]), .rt(rt[i]),
                .mem_read_valid(data_mem_read_valid[i]),
                .mem_read_address(data_mem_read_address[i]),
                .mem_read_ready(data_mem_read_ready[i]),
                .mem_read_data(data_mem_read_data[i]),
                .mem_write_valid(data_mem_write_valid[i]),
                .mem_write_address(data_mem_write_address[i]),
                .mem_write_data(data_mem_write_data[i]),
                .mem_write_ready(data_mem_write_ready[i]),
                .smem_read_enable(smem_read_enable_v[i]),
                .smem_read_address(smem_read_address[i]),
                .smem_read_data(smem_read_data[i]),
                .smem_write_enable(smem_write_enable_v[i]),
                .smem_write_address(smem_write_address[i]),
                .smem_write_data(smem_write_data[i]),
                .lsu_state(lsu_state[i]),
                .lsu_out(lsu_out[i])
            );

            registers #(
                .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
                .THREAD_ID(i),
                .DATA_BITS(DATA_MEM_DATA_BITS)
            ) register_instance (
                .clk(clk), .reset(reset), .enable(i < thread_count),
                .block_id(block_id),
                .core_state(core_state),
                .decoded_reg_write_enable(decoded_reg_write_enable),
                .decoded_reg_input_mux(decoded_reg_input_mux),
                .decoded_rd_address(decoded_rd_address),
                .decoded_rs_address(decoded_rs_address),
                .decoded_rt_address(decoded_rt_address),
                .decoded_immediate(decoded_immediate),
                .alu_out(alu_out[i]),
                .lsu_out(lsu_out[i]),
                .rs(rs[i]), .rt(rt[i])
            );

            pc #(
                .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS),
                .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS)
            ) pc_instance (
                .clk(clk), .reset(reset), .enable(i < thread_count),
                .core_state(core_state),
                .decoded_nzp(decoded_nzp),
                .decoded_immediate(decoded_immediate),
                .decoded_nzp_write_enable(decoded_nzp_write_enable),
                .decoded_pc_mux(decoded_pc_mux),
                .alu_out(alu_out[i]),
                .current_pc(current_pc),
                .next_pc(next_pc[i])
            );
        end
    endgenerate
endmodule
