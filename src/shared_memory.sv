`default_nettype none
`timescale 1ns/1ns

// Per-block shared memory (Phase 7 design — not wired into the core yet).
// Single-cycle, banked by THREADS_PER_BLOCK. One R/W port per lane.
// SIZE entries × DATA_BITS bits.
module shared_memory #(
    parameter THREADS_PER_BLOCK = 4,
    parameter SIZE              = 64,
    parameter ADDR_BITS         = 6,
    parameter DATA_BITS         = 8
) (
    input  wire                          clk,
    input  wire                          reset,
    // Per-lane R/W ports
    input  wire [THREADS_PER_BLOCK-1:0]  lane_read_enable,
    input  wire [ADDR_BITS-1:0]          lane_read_addr  [THREADS_PER_BLOCK-1:0],
    output reg  [DATA_BITS-1:0]          lane_read_data  [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]  lane_write_enable,
    input  wire [ADDR_BITS-1:0]          lane_write_addr [THREADS_PER_BLOCK-1:0],
    input  wire [DATA_BITS-1:0]          lane_write_data [THREADS_PER_BLOCK-1:0]
);
    reg [DATA_BITS-1:0] mem [0:SIZE-1];

    integer i;
    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < SIZE; i = i + 1) mem[i] <= {DATA_BITS{1'b0}};
            for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) lane_read_data[i] <= {DATA_BITS{1'b0}};
        end else begin
            for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                if (lane_write_enable[i]) mem[lane_write_addr[i]] <= lane_write_data[i];
                if (lane_read_enable[i])  lane_read_data[i]       <= mem[lane_read_addr[i]];
            end
        end
    end
endmodule
