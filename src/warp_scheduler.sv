`default_nettype none
`timescale 1ns/1ns

// Warp Scheduler (Phase 8)
// Selects which warp gets the issue slot each cycle, per a parameterised policy.
//
// Policies (POLICY parameter):
//   0 — LRR    (loose round-robin): rotating priority pointer; pick next ready.
//   1 — GTO    (greedy-then-oldest): stick with current warp while it's ready
//              and progressing; on stall pick the oldest ready warp.
//   2 — GTLRR  (Crouzet, Carle, Rochange 2025): separate fetch and issue
//              rotation pointers, locked to bound the worst-case execution time.
//              Implementation: round-robin among ready+progressing warps;
//              fall back to LRR otherwise. The lockstep policy lets the
//              issue order be statically analyzable.
//
// All three are stateless w.r.t. data dependencies — they only look at
// readiness, age, and progress signals provided by the surrounding core.
module warp_scheduler #(
    parameter WARPS_PER_CORE = 4,
    parameter POLICY         = 0
) (
    input  wire                           clk,
    input  wire                           reset,
    // Per-warp readiness (can it issue this cycle?)
    input  wire [WARPS_PER_CORE-1:0]      warp_ready,
    // Per-warp progress: 1 if this warp issued / advanced since last cycle.
    // Used by GTO to detect stalls.
    input  wire [WARPS_PER_CORE-1:0]      warp_progressing,
    // Per-warp age (instructions issued so far). Used by GTO and GTLRR.
    input  wire [7:0]                     warp_age [WARPS_PER_CORE-1:0],

    output reg  [$clog2(WARPS_PER_CORE+1)-1:0] selected_warp,
    output reg                            issue_valid
);
    // Rotation pointers
    reg [$clog2(WARPS_PER_CORE+1)-1:0] lrr_ptr;     // LRR / GTLRR rotation
    reg [$clog2(WARPS_PER_CORE+1)-1:0] gto_current; // GTO sticky warp

    localparam PTR_BITS = $clog2(WARPS_PER_CORE+1);

    integer i, j;
    reg [PTR_BITS-1:0] candidate;
    reg                found;
    reg [7:0]          oldest_age;
    reg [PTR_BITS-1:0] oldest_idx;

    always @(*) begin
        // Default
        selected_warp = 0;
        issue_valid   = 1'b0;
        candidate     = 0;
        found         = 1'b0;
        oldest_age    = 8'hFF;
        oldest_idx    = 0;

        case (POLICY)
            // ----- LRR: rotate -----
            0: begin
                for (i = 0; i < WARPS_PER_CORE; i = i + 1) begin
                    j = (lrr_ptr + i) % WARPS_PER_CORE;
                    if (!found && warp_ready[j]) begin
                        candidate     = j[PTR_BITS-1:0];
                        selected_warp = candidate;
                        issue_valid   = 1'b1;
                        found         = 1'b1;
                    end
                end
            end

            // ----- GTO: greedy then oldest -----
            1: begin
                if (warp_ready[gto_current] && warp_progressing[gto_current]) begin
                    selected_warp = gto_current;
                    issue_valid   = 1'b1;
                end else begin
                    // Stalled — find oldest ready warp
                    for (i = 0; i < WARPS_PER_CORE; i = i + 1) begin
                        if (warp_ready[i] && warp_age[i] < oldest_age) begin
                            oldest_age = warp_age[i];
                            oldest_idx = i[PTR_BITS-1:0];
                            found      = 1'b1;
                        end
                    end
                    if (found) begin
                        selected_warp = oldest_idx;
                        issue_valid   = 1'b1;
                    end
                end
            end

            // ----- GTLRR: round-robin among ready+progressing, else LRR -----
            2: begin
                for (i = 0; i < WARPS_PER_CORE; i = i + 1) begin
                    j = (lrr_ptr + i) % WARPS_PER_CORE;
                    if (!found && warp_ready[j] && warp_progressing[j]) begin
                        candidate     = j[PTR_BITS-1:0];
                        selected_warp = candidate;
                        issue_valid   = 1'b1;
                        found         = 1'b1;
                    end
                end
                if (!found) begin
                    for (i = 0; i < WARPS_PER_CORE; i = i + 1) begin
                        j = (lrr_ptr + i) % WARPS_PER_CORE;
                        if (!found && warp_ready[j]) begin
                            candidate     = j[PTR_BITS-1:0];
                            selected_warp = candidate;
                            issue_valid   = 1'b1;
                            found         = 1'b1;
                        end
                    end
                end
            end

            default: begin
                selected_warp = 0;
                issue_valid   = 1'b0;
            end
        endcase
    end

    // Sequential: update rotation pointers / GTO sticky.
    always @(posedge clk) begin
        if (reset) begin
            lrr_ptr     <= 0;
            gto_current <= 0;
        end else if (issue_valid) begin
            // LRR: advance past selected warp
            lrr_ptr <= (selected_warp + 1) % WARPS_PER_CORE;
            // GTO: latch selected as sticky
            gto_current <= selected_warp;
        end
    end
endmodule
