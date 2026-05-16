`default_nettype none
`timescale 1ns/1ns

// Block Dispatcher
// Hands blocks to free cores until total_blocks is reached, then asserts done.
module dispatch #(
    parameter NUM_CORES         = 2,
    parameter THREADS_PER_BLOCK = 4
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              start,
    input  wire [7:0]                        thread_count,

    input  wire [NUM_CORES-1:0]              core_done,
    output reg  [NUM_CORES-1:0]              core_start,
    output reg  [NUM_CORES-1:0]              core_reset,
    output reg  [7:0]                        core_block_id   [NUM_CORES-1:0],
    output reg  [$clog2(THREADS_PER_BLOCK):0] core_thread_count [NUM_CORES-1:0],

    output reg                               done
);
    wire [7:0] total_blocks;
    assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    reg [7:0] blocks_dispatched;
    reg [7:0] blocks_done;
    reg       start_seen;
    integer i;

    always @(posedge clk) begin
        if (reset) begin
            done              <= 0;
            blocks_dispatched  = 0;
            blocks_done        = 0;
            start_seen        <= 0;
            for (i = 0; i < NUM_CORES; i = i + 1) begin
                core_start[i]        <= 0;
                core_reset[i]        <= 1;
                core_block_id[i]     <= 0;
                core_thread_count[i] <= THREADS_PER_BLOCK;
            end
        end else if (start) begin
            if (!start_seen) begin
                start_seen <= 1;
                for (i = 0; i < NUM_CORES; i = i + 1) core_reset[i] <= 1;
            end

            if (blocks_done == total_blocks) done <= 1;

            for (i = 0; i < NUM_CORES; i = i + 1) begin
                if (core_reset[i]) begin
                    core_reset[i] <= 0;
                    if (blocks_dispatched < total_blocks) begin
                        core_start[i]    <= 1;
                        core_block_id[i] <= blocks_dispatched;
                        core_thread_count[i] <= (blocks_dispatched == total_blocks - 1)
                            ? (thread_count - blocks_dispatched * THREADS_PER_BLOCK)
                            : THREADS_PER_BLOCK;
                        blocks_dispatched = blocks_dispatched + 1;
                    end
                end
            end

            for (i = 0; i < NUM_CORES; i = i + 1) begin
                if (core_start[i] && core_done[i]) begin
                    core_reset[i] <= 1;
                    core_start[i] <= 0;
                    blocks_done    = blocks_done + 1;
                end
            end
        end
    end
endmodule
