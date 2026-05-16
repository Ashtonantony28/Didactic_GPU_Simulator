`default_nettype none
`timescale 1ns/1ns

// Per-block shared memory (Phase 7).
// Combinational read, synchronous write. SIZE entries × DATA_BITS bits.
// Addresses wider than $clog2(SIZE) are masked.
module shared_memory #(
    parameter THREADS_PER_BLOCK = 4,
    parameter SIZE              = 64,
    parameter ADDR_BITS         = 8,
    parameter DATA_BITS         = 8
) (
    input  wire                          clk,
    input  wire                          reset,
    // Per-lane R/W ports
    input  wire [THREADS_PER_BLOCK-1:0]  lane_read_enable,
    input  wire [ADDR_BITS-1:0]          lane_read_addr  [THREADS_PER_BLOCK-1:0],
    output wire [DATA_BITS-1:0]          lane_read_data  [THREADS_PER_BLOCK-1:0],
    input  wire [THREADS_PER_BLOCK-1:0]  lane_write_enable,
    input  wire [ADDR_BITS-1:0]          lane_write_addr [THREADS_PER_BLOCK-1:0],
    input  wire [DATA_BITS-1:0]          lane_write_data [THREADS_PER_BLOCK-1:0]
);
    localparam IDX_BITS = (SIZE <= 1) ? 1 : $clog2(SIZE);

    reg [DATA_BITS-1:0] mem [0:SIZE-1];

    // Combinational reads
    genvar g;
    generate
        for (g = 0; g < THREADS_PER_BLOCK; g = g + 1) begin : rports
            assign lane_read_data[g] = mem[lane_read_addr[g][IDX_BITS-1:0]];
        end
    endgenerate

    // Synchronous writes
    integer i;
    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < SIZE; i = i + 1) mem[i] <= {DATA_BITS{1'b0}};
        end else begin
            for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                if (lane_write_enable[i]) mem[lane_write_addr[i][IDX_BITS-1:0]] <= lane_write_data[i];
            end
        end
    end
endmodule
