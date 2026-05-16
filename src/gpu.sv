`default_nettype none
`timescale 1ns/1ns

// GPU Top
// Wires the dispatcher, memory controllers, and an array of cores onto external memory.
// All Phase-3+ extensions are gated by parameters so the original Phase-1 baseline is the default.
module gpu #(
    parameter DATA_MEM_ADDR_BITS        = 8,
    parameter DATA_MEM_DATA_BITS        = 8,
    parameter DATA_MEM_NUM_CHANNELS     = 4,
    parameter PROGRAM_MEM_ADDR_BITS     = 8,
    parameter PROGRAM_MEM_DATA_BITS     = 16,
    parameter PROGRAM_MEM_NUM_CHANNELS  = 1,
    parameter NUM_CORES                 = 2,
    parameter THREADS_PER_BLOCK         = 4,
    parameter ICACHE_LINES              = 0,
    parameter COALESCE                  = 0
) (
    input  wire                              clk,
    input  wire                              reset,

    input  wire                              start,
    output wire                              done,

    input  wire                              device_control_write_enable,
    input  wire [7:0]                        device_control_data,

    output wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_valid,
    output wire [PROGRAM_MEM_ADDR_BITS-1:0]    program_mem_read_address [PROGRAM_MEM_NUM_CHANNELS-1:0],
    input  wire [PROGRAM_MEM_NUM_CHANNELS-1:0] program_mem_read_ready,
    input  wire [PROGRAM_MEM_DATA_BITS-1:0]    program_mem_read_data    [PROGRAM_MEM_NUM_CHANNELS-1:0],

    output wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_read_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]       data_mem_read_address    [DATA_MEM_NUM_CHANNELS-1:0],
    input  wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_read_ready,
    input  wire [DATA_MEM_DATA_BITS-1:0]       data_mem_read_data       [DATA_MEM_NUM_CHANNELS-1:0],
    output wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_write_valid,
    output wire [DATA_MEM_ADDR_BITS-1:0]       data_mem_write_address   [DATA_MEM_NUM_CHANNELS-1:0],
    output wire [DATA_MEM_DATA_BITS-1:0]       data_mem_write_data      [DATA_MEM_NUM_CHANNELS-1:0],
    input  wire [DATA_MEM_NUM_CHANNELS-1:0]    data_mem_write_ready,

    // Bench telemetry (Phase 2+)
    output wire [31:0]                         data_mem_requests,
    output wire [31:0]                         data_mem_requests_coalesced,
    output wire [31:0]                         icache_hits_total,
    output wire [31:0]                         icache_misses_total
);
    wire [7:0] thread_count;

    wire [NUM_CORES-1:0] core_start;
    wire [NUM_CORES-1:0] core_reset;
    wire [NUM_CORES-1:0] core_done;
    wire [7:0]           core_block_id     [NUM_CORES-1:0];
    wire [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0];

    localparam NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK;
    wire [NUM_LSUS-1:0] lsu_read_valid;
    wire [DATA_MEM_ADDR_BITS-1:0] lsu_read_address [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0] lsu_read_ready;
    wire [DATA_MEM_DATA_BITS-1:0] lsu_read_data    [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0] lsu_write_valid;
    wire [DATA_MEM_ADDR_BITS-1:0] lsu_write_address [NUM_LSUS-1:0];
    wire [DATA_MEM_DATA_BITS-1:0] lsu_write_data    [NUM_LSUS-1:0];
    wire [NUM_LSUS-1:0] lsu_write_ready;

    localparam NUM_FETCHERS = NUM_CORES;
    wire [NUM_FETCHERS-1:0]              fetcher_read_valid;
    wire [PROGRAM_MEM_ADDR_BITS-1:0]     fetcher_read_address [NUM_FETCHERS-1:0];
    wire [NUM_FETCHERS-1:0]              fetcher_read_ready;
    wire [PROGRAM_MEM_DATA_BITS-1:0]     fetcher_read_data    [NUM_FETCHERS-1:0];

    // Program controller doesn't write, but the controller wants a write port. Wire to 0.
    wire [NUM_FETCHERS-1:0] zero_write_valid = '0;
    wire [PROGRAM_MEM_ADDR_BITS-1:0] zero_write_addr [NUM_FETCHERS-1:0];
    wire [PROGRAM_MEM_DATA_BITS-1:0] zero_write_data [NUM_FETCHERS-1:0];
    wire [NUM_FETCHERS-1:0] discard_write_ready;
    wire [PROGRAM_MEM_NUM_CHANNELS-1:0] discard_pmem_write_valid;
    wire [PROGRAM_MEM_ADDR_BITS-1:0] discard_pmem_write_addr [PROGRAM_MEM_NUM_CHANNELS-1:0];
    wire [PROGRAM_MEM_DATA_BITS-1:0] discard_pmem_write_data [PROGRAM_MEM_NUM_CHANNELS-1:0];
    wire [PROGRAM_MEM_NUM_CHANNELS-1:0] zero_pmem_write_ready = '0;
    wire [31:0] prog_requests, prog_requests_coalesced;

    genvar gi;
    generate
        for (gi = 0; gi < NUM_FETCHERS; gi = gi + 1) begin : zwa
            assign zero_write_addr[gi] = '0;
            assign zero_write_data[gi] = '0;
        end
    endgenerate

    dcr dcr_instance (
        .clk(clk), .reset(reset),
        .device_control_write_enable(device_control_write_enable),
        .device_control_data(device_control_data),
        .thread_count(thread_count)
    );

    controller #(
        .ADDR_BITS(DATA_MEM_ADDR_BITS),
        .DATA_BITS(DATA_MEM_DATA_BITS),
        .NUM_CONSUMERS(NUM_LSUS),
        .NUM_CHANNELS(DATA_MEM_NUM_CHANNELS),
        .WRITE_ENABLE(1),
        .COALESCE(COALESCE)
    ) data_memory_controller (
        .clk(clk), .reset(reset),
        .consumer_read_valid(lsu_read_valid),
        .consumer_read_address(lsu_read_address),
        .consumer_read_ready(lsu_read_ready),
        .consumer_read_data(lsu_read_data),
        .consumer_write_valid(lsu_write_valid),
        .consumer_write_address(lsu_write_address),
        .consumer_write_data(lsu_write_data),
        .consumer_write_ready(lsu_write_ready),
        .mem_read_valid(data_mem_read_valid),
        .mem_read_address(data_mem_read_address),
        .mem_read_ready(data_mem_read_ready),
        .mem_read_data(data_mem_read_data),
        .mem_write_valid(data_mem_write_valid),
        .mem_write_address(data_mem_write_address),
        .mem_write_data(data_mem_write_data),
        .mem_write_ready(data_mem_write_ready),
        .mem_requests(data_mem_requests),
        .mem_requests_coalesced(data_mem_requests_coalesced)
    );

    controller #(
        .ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
        .DATA_BITS(PROGRAM_MEM_DATA_BITS),
        .NUM_CONSUMERS(NUM_FETCHERS),
        .NUM_CHANNELS(PROGRAM_MEM_NUM_CHANNELS),
        .WRITE_ENABLE(0),
        .COALESCE(0)
    ) program_memory_controller (
        .clk(clk), .reset(reset),
        .consumer_read_valid(fetcher_read_valid),
        .consumer_read_address(fetcher_read_address),
        .consumer_read_ready(fetcher_read_ready),
        .consumer_read_data(fetcher_read_data),
        .consumer_write_valid(zero_write_valid),
        .consumer_write_address(zero_write_addr),
        .consumer_write_data(zero_write_data),
        .consumer_write_ready(discard_write_ready),
        .mem_read_valid(program_mem_read_valid),
        .mem_read_address(program_mem_read_address),
        .mem_read_ready(program_mem_read_ready),
        .mem_read_data(program_mem_read_data),
        .mem_write_valid(discard_pmem_write_valid),
        .mem_write_address(discard_pmem_write_addr),
        .mem_write_data(discard_pmem_write_data),
        .mem_write_ready(zero_pmem_write_ready),
        .mem_requests(prog_requests),
        .mem_requests_coalesced(prog_requests_coalesced)
    );

    dispatch #(
        .NUM_CORES(NUM_CORES),
        .THREADS_PER_BLOCK(THREADS_PER_BLOCK)
    ) dispatch_instance (
        .clk(clk), .reset(reset), .start(start),
        .thread_count(thread_count),
        .core_done(core_done),
        .core_start(core_start),
        .core_reset(core_reset),
        .core_block_id(core_block_id),
        .core_thread_count(core_thread_count),
        .done(done)
    );

    // Aggregate I-cache telemetry across cores (packed for portability).
    wire [NUM_CORES*32-1:0] per_core_icache_hits;
    wire [NUM_CORES*32-1:0] per_core_icache_misses;

    genvar c;
    generate
        for (c = 0; c < NUM_CORES; c = c + 1) begin : cores
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_read_valid;
            wire [DATA_MEM_ADDR_BITS-1:0]     core_lsu_read_address  [THREADS_PER_BLOCK-1:0];
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_read_ready;
            wire [DATA_MEM_DATA_BITS-1:0]     core_lsu_read_data     [THREADS_PER_BLOCK-1:0];
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_write_valid;
            wire [DATA_MEM_ADDR_BITS-1:0]     core_lsu_write_address [THREADS_PER_BLOCK-1:0];
            wire [DATA_MEM_DATA_BITS-1:0]     core_lsu_write_data    [THREADS_PER_BLOCK-1:0];
            wire [THREADS_PER_BLOCK-1:0]      core_lsu_write_ready;

            genvar t;
            for (t = 0; t < THREADS_PER_BLOCK; t = t + 1) begin : lsu_xbar
                localparam integer LIDX = c * THREADS_PER_BLOCK + t;
                assign lsu_read_valid[LIDX]      = core_lsu_read_valid[t];
                assign lsu_read_address[LIDX]    = core_lsu_read_address[t];
                assign core_lsu_read_ready[t]    = lsu_read_ready[LIDX];
                assign core_lsu_read_data[t]     = lsu_read_data[LIDX];
                assign lsu_write_valid[LIDX]     = core_lsu_write_valid[t];
                assign lsu_write_address[LIDX]   = core_lsu_write_address[t];
                assign lsu_write_data[LIDX]      = core_lsu_write_data[t];
                assign core_lsu_write_ready[t]   = lsu_write_ready[LIDX];
            end

            core #(
                .DATA_MEM_ADDR_BITS(DATA_MEM_ADDR_BITS),
                .DATA_MEM_DATA_BITS(DATA_MEM_DATA_BITS),
                .PROGRAM_MEM_ADDR_BITS(PROGRAM_MEM_ADDR_BITS),
                .PROGRAM_MEM_DATA_BITS(PROGRAM_MEM_DATA_BITS),
                .THREADS_PER_BLOCK(THREADS_PER_BLOCK),
                .ICACHE_LINES(ICACHE_LINES)
            ) core_instance (
                .clk(clk),
                .reset(core_reset[c]),
                .telemetry_reset(reset),
                .start(core_start[c]),
                .done(core_done[c]),
                .block_id(core_block_id[c]),
                .thread_count(core_thread_count[c]),
                .program_mem_read_valid(fetcher_read_valid[c]),
                .program_mem_read_address(fetcher_read_address[c]),
                .program_mem_read_ready(fetcher_read_ready[c]),
                .program_mem_read_data(fetcher_read_data[c]),
                .data_mem_read_valid(core_lsu_read_valid),
                .data_mem_read_address(core_lsu_read_address),
                .data_mem_read_ready(core_lsu_read_ready),
                .data_mem_read_data(core_lsu_read_data),
                .data_mem_write_valid(core_lsu_write_valid),
                .data_mem_write_address(core_lsu_write_address),
                .data_mem_write_data(core_lsu_write_data),
                .data_mem_write_ready(core_lsu_write_ready),
                .icache_hits(per_core_icache_hits[c*32 +: 32]),
                .icache_misses(per_core_icache_misses[c*32 +: 32])
            );
        end
    endgenerate

    // Reduction tree (only NUM_CORES=2 supported here; trivially extensible).
    wire [31:0] icache_hits_c0 = per_core_icache_hits[31:0];
    wire [31:0] icache_miss_c0 = per_core_icache_misses[31:0];
    wire [31:0] icache_hits_c1 = (NUM_CORES > 1) ? per_core_icache_hits  [63:32] : 32'b0;
    wire [31:0] icache_miss_c1 = (NUM_CORES > 1) ? per_core_icache_misses[63:32] : 32'b0;
    assign icache_hits_total   = icache_hits_c0 + icache_hits_c1;
    assign icache_misses_total = icache_miss_c0 + icache_miss_c1;
endmodule
