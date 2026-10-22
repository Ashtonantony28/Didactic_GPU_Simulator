`default_nettype none
`timescale 1ns/1ns

// SIMT reconvergence stack of one warp (ADR-011, ADR-015). Entry 0 is the bottom; entry sp runs under its mask.
// The state after the warp's current instruction is combinational (next_pc, next_done); `update` commits it.
module simt_stack #(
    parameter LANES = 4,
    parameter DEPTH = 8
) (
    input  wire             clk,
    input  wire             reset,
    input  wire             update,
    input  wire [LANES-1:0] exists,     // lanes holding a thread of the block
    input  wire             branch,     // BR, nzp != 000
    input  wire             sync,       // SYNC (BR, nzp = 000)
    input  wire             ret,
    input  wire [7:0]       immediate,
    input  wire [LANES-1:0] taken,      // lanes whose NZP matches the branch
    output wire [LANES-1:0] mask,
    output reg  [7:0]       next_pc,
    output reg              next_done
);
    localparam N  = DEPTH + 3;          // bottom, DEPTH pushed entries, and room for the push that overflows
    localparam SB = $clog2(N);

    reg [7:0]       e_pc   [N-1:0];
    reg [7:0]       e_rpc  [N-1:0];
    reg [LANES-1:0] e_mask [N-1:0];
    reg [SB-1:0]    sp;
    reg             sync_valid;
    reg [7:0]       sync_pc;

    reg [7:0]       n_pc   [N-1:0];
    reg [7:0]       n_rpc  [N-1:0];
    reg [LANES-1:0] n_mask [N-1:0];
    reg [SB-1:0]    n_sp;
    reg             n_sync_valid, diverge;
    reg [7:0]       n_sync_pc;
    wire [LANES-1:0] t = taken & mask;
    integer k;

    assign mask = e_mask[sp] & exists;

    always @(*) begin
        for (k = 0; k < N; k = k + 1) begin
            n_pc[k]   = e_pc[k];
            n_rpc[k]  = e_rpc[k];
            n_mask[k] = e_mask[k];
        end
        n_sp         = sp;
        n_sync_valid = sync_valid;
        n_sync_pc    = sync_pc;
        diverge      = 1'b0;
        n_pc[sp]     = e_pc[sp] + 1;
        if (ret) begin
            // Emptied entries are always a suffix of the stack (ADR-015).
            n_sp = 0;
            for (k = 0; k < N; k = k + 1) begin
                n_mask[k] = e_mask[k] & ~mask;
                if (k <= sp && (n_mask[k] & exists) != 0) n_sp = k;
            end
        end else if (sync) begin
            n_sync_valid = 1'b1;
            n_sync_pc    = immediate;
        end else if (branch && t == mask) begin
            n_pc[sp] = immediate;
        end else if (branch && t != 0) begin
            diverge          = 1'b1;
            n_sync_valid     = 1'b0;
            n_pc[sp]         = sync_pc;
            n_pc[sp + 1]     = e_pc[sp] + 1;
            n_rpc[sp + 1]    = sync_pc;
            n_mask[sp + 1]   = mask & ~t;
            n_pc[sp + 2]     = immediate;
            n_rpc[sp + 2]    = sync_pc;
            n_mask[sp + 2]   = t;
            n_sp             = sp + 2;
        end
        for (k = N - 1; k > 0; k = k - 1)
            if (k == n_sp && n_pc[k] == n_rpc[k]) n_sp = k - 1;
        next_pc   = n_pc[n_sp];
        next_done = ret && (e_mask[0] & exists & ~mask) == 0;
    end

    always @(posedge clk) begin
        if (reset) begin
            sp         <= 0;
            sync_valid <= 1'b0;
            sync_pc    <= 0;
            e_pc[0]    <= 0;
            e_rpc[0]   <= 0;
            e_mask[0]  <= {LANES{1'b1}};
        end else if (update) begin
            if (diverge && !sync_valid) $fatal(1, "simt_stack: divergent branch at pc %0d without SYNC", e_pc[sp]);
            if (diverge && sp + 2 > DEPTH) $fatal(1, "simt_stack: deeper than %0d at pc %0d", DEPTH, e_pc[sp]);
            for (k = 0; k < N; k = k + 1) begin
                e_pc[k]   <= n_pc[k];
                e_rpc[k]  <= n_rpc[k];
                e_mask[k] <= n_mask[k];
            end
            sp         <= n_sp;
            sync_valid <= n_sync_valid;
            sync_pc    <= n_sync_pc;
        end
    end
endmodule
