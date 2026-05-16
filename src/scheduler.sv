`default_nettype none
`timescale 1ns/1ns

// Scheduler (control-flow FSM for one block on one core)
// Stages: IDLE → FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE → (loop or DONE).
// PIPELINED=1 collapses the dead DECODE state (combinational decoder makes
// it redundant) — one cycle saved per instruction.
module scheduler #(
    parameter THREADS_PER_BLOCK = 4,
    parameter PIPELINED         = 0
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        start,
    input  wire        decoded_mem_read_enable,
    input  wire        decoded_mem_write_enable,
    input  wire        decoded_smem_read_enable,
    input  wire        decoded_smem_write_enable,
    input  wire        decoded_bar,
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
                FETCH: if (fetcher_state == 3'b010) begin
                    core_state <= (PIPELINED != 0) ? REQUEST : DECODE;
                end
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
