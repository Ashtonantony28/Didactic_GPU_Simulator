`default_nettype none
`timescale 1ns/1ns

// Scheduler (control-flow FSM for one block on one core)
// Stages: IDLE → FETCH → DECODE → REQUEST → WAIT → EXECUTE → UPDATE → (loop or DONE).
// PIPELINED=1 collapses the dead DECODE state (combinational decoder makes
// it redundant) — one cycle saved per instruction.
// WARPS_PER_CORE > 1 (ADR-012, ADR-014): IDLE is the issue slot where warp_scheduler picks a warp and
// UPDATE returns to IDLE. A warp whose global load or store is still in flight in WAIT yields and re-issues
// the same instruction once its LSUs are done; SMEM accesses (two cycles) do not yield. BAR holds a warp
// until every unfinished warp has reached a BAR.
module scheduler #(
    parameter THREADS_PER_BLOCK = 4,
    parameter PIPELINED         = 0,
    parameter WARPS_PER_CORE    = 1,
    parameter WARP_POLICY       = 0
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        telemetry_reset,
    input  wire        start,
    input  wire [$clog2(THREADS_PER_BLOCK):0] thread_count,
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
    output reg         done,
    output reg  [$clog2(WARPS_PER_CORE+1)-1:0] current_warp,
    output reg  [31:0] warp_issues,
    output reg  [31:0] bar_stall_cycles
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
    localparam W     = WARPS_PER_CORE;
    localparam LANES = THREADS_PER_BLOCK / W;
    localparam WB    = $clog2(W + 1);

    integer i;
    reg any_lsu_waiting;

    // Per-warp state, used when W > 1.
    reg  [7:0]    warp_pc  [W-1:0];
    reg  [7:0]    warp_age [W-1:0];
    reg  [W-1:0]  warp_done, warp_bar, warp_pending;
    wire [W-1:0]  warp_live, warp_busy, warp_ready;
    wire          bar_release, issue_valid;
    wire [WB-1:0] selected_warp;
    wire [31:0]   bar_waiting;

    generate
        if (W > 1) begin : warps
            reg [W-1:0] busy;
            reg [31:0]  waiting;
            integer     t, w;
            wire [W-1:0] last_issued = 1 << current_warp;
            always @(*) begin
                busy = 0;
                for (t = 0; t < THREADS_PER_BLOCK; t = t + 1)
                    if (lsu_state[t] == 2'b01 || lsu_state[t] == 2'b10) busy[t / LANES] = 1'b1;
            end
            genvar g;
            for (g = 0; g < W; g = g + 1) begin : live
                assign warp_live[g] = g * LANES < thread_count && !warp_done[g];
            end
            assign warp_busy   = busy;
            assign bar_release = |warp_live && &(warp_bar | ~warp_live);
            assign warp_ready  = (core_state == IDLE && start)
                ? warp_live & ~(warp_bar & ~{W{bar_release}}) & ~(warp_pending & warp_busy) : {W{1'b0}};
            always @(*) begin
                waiting = 0;
                if (!bar_release)
                    for (w = 0; w < W; w = w + 1) waiting = waiting + warp_bar[w];
            end
            assign bar_waiting = waiting;
            warp_scheduler #(.WARPS_PER_CORE(W), .POLICY(WARP_POLICY)) warp_scheduler_instance (
                .clk(clk), .reset(reset),
                .warp_ready(warp_ready),
                .warp_progressing(warp_ready & last_issued),
                .warp_age(warp_age),
                .selected_warp(selected_warp),
                .issue_valid(issue_valid)
            );
        end else begin : one_warp
            assign warp_live     = 1'b1;
            assign warp_busy     = 1'b0;
            assign warp_ready    = 1'b0;
            assign bar_release   = 1'b0;
            assign bar_waiting   = 0;
            assign selected_warp = 0;
            assign issue_valid   = 1'b0;
        end
    endgenerate

    // Telemetry (ADR-012): one issue per instruction a warp starts, and per cycle the warps held at a BAR.
    // Reset with the GPU, not with the per-block core reset.
    wire issue = (W == 1) ? ((core_state == IDLE && start) || (core_state == UPDATE && !decoded_ret))
                          : (issue_valid && !warp_pending[selected_warp]);
    always @(posedge clk) begin
        if (telemetry_reset) begin
            warp_issues      <= 0;
            bar_stall_cycles <= 0;
        end else begin
            if (issue) warp_issues <= warp_issues + 1;
            if (W > 1) bar_stall_cycles <= bar_stall_cycles + bar_waiting;
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            current_pc   <= 0;
            core_state   <= IDLE;
            done         <= 0;
            current_warp <= 0;
            warp_done    <= 0;
            warp_bar     <= 0;
            warp_pending <= 0;
            for (i = 0; i < W; i = i + 1) begin
                warp_pc[i]  <= 0;
                warp_age[i] <= 0;
            end
        end else begin
            if (bar_release) warp_bar <= 0;
            case (core_state)
                IDLE: if (start) begin
                    if (W == 1) core_state <= FETCH;
                    else if (issue_valid) begin
                        current_warp <= selected_warp;
                        current_pc   <= warp_pc[selected_warp];
                        warp_pending[selected_warp] <= 1'b0;
                        // Saturate below 8'hFF: warp_scheduler's GTO never picks a warp of age 8'hFF.
                        if (!warp_pending[selected_warp] && warp_age[selected_warp] != 8'hFE)
                            warp_age[selected_warp] <= warp_age[selected_warp] + 1;
                        core_state   <= FETCH;
                    end else if (!warp_live) begin
                        done       <= 1;
                        core_state <= DONE;
                    end
                end
                FETCH: if (fetcher_state == 3'b010) begin
                    core_state <= (PIPELINED != 0) ? REQUEST : DECODE;
                end
                DECODE: core_state <= REQUEST;
                REQUEST: core_state <= WAIT;
                WAIT: if (W == 1) begin
                    any_lsu_waiting = 1'b0;
                    for (i = 0; i < THREADS_PER_BLOCK; i = i + 1) begin
                        if (lsu_state[i] == 2'b01 || lsu_state[i] == 2'b10)
                            any_lsu_waiting = 1'b1;
                    end
                    if (!any_lsu_waiting) core_state <= EXECUTE;
                end else if (warp_busy[current_warp] && (decoded_mem_read_enable || decoded_mem_write_enable)) begin
                    warp_pending[current_warp] <= 1'b1;
                    core_state <= IDLE;
                end else if (!warp_busy[current_warp]) core_state <= EXECUTE;
                EXECUTE: core_state <= UPDATE;
                UPDATE: begin
                    if (W > 1) begin
                        warp_pc[current_warp] <= next_pc[current_warp * LANES];
                        if (decoded_ret) warp_done[current_warp] <= 1'b1;
                        if (decoded_bar) warp_bar[current_warp]  <= 1'b1;
                        core_state <= IDLE;
                    end else if (decoded_ret) begin
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
