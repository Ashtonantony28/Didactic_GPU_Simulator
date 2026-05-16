`default_nettype none
`timescale 1ns/1ns

// Scheduler (control-flow FSM for one block on one core)
// Stages: IDLE → FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE → (loop or DONE).
// Per-thread next_pc reduces to a single PC under the no-divergence assumption;
// divergence handling is layered on by the per-core warp_scheduler in later phases.
module scheduler #(
    parameter THREADS_PER_BLOCK = 4
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire        decoded_mem_read_enable,
    input  wire        decoded_mem_write_enable,
    input  wire        decoded_ret,
    input  wire [2:0]  fetcher_state,
    input  wire [1:0]  lsu_state [THREADS_PER_BLOCK-1:0],
    output reg  [7:0]  current_pc,
    input  wire [7:0]  next_pc [THREADS_PER_BLOCK-1:0],
    output reg  [2:0]  core_state,
    output reg         done
);
    localparam [2:0]
        IDLE    = 3'b000,
        FETCH   = 3'b001,
        DECODE  = 3'b010,
        REQUEST = 3'b011,
        WAIT    = 3'b100,
        EXECUTE = 3'b101,
        UPDATE  = 3'b110,
        DONE    = 3'b111;

    integer i;
    reg any_lsu_waiting;

    always @(posedge clk) begin
        if (reset) begin
            current_pc <= 0;
            core_state <= IDLE;
            done       <= 0;
        end else begin
            case (core_state)
                IDLE: if (start) core_state <= FETCH;
                FETCH: if (fetcher_state == 3'b010) core_state <= DECODE;
                DECODE: core_state <= REQUEST;
                REQUEST: core_state <= WAIT;
                WAIT: begin
                    any_lsu_waiting = 1'b0;
                    for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                        if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10)
                            any_lsu_waiting = 1'b1;
                    end
                    if (!any_lsu_waiting) core_state <= EXECUTE;
                end
                EXECUTE: core_state <= UPDATE;
                UPDATE: begin
                    if (decoded_ret) begin
                        done       <= 1;
                        core_state <= DONE;
                    end else begin
                        // No divergence: all threads converge on the same next PC.
                        // Use lane 0 — guaranteed enabled (thread_count >= 1).
                        // Upstream uses THREADS_PER_BLOCK-1 which freezes when the last
                        // block has fewer threads than THREADS_PER_BLOCK.
                        current_pc <= next_pc[0];
                        core_state <= FETCH;
                    end
                end
                DONE: ;
                default: ;
            endcase
        end
    end
endmodule
