`default_nettype none
`timescale 1ns/1ns

// Testbench wrapper that exposes the warp scheduler under a simple cocotb
// interface. We model a workload as N warps that each take T cycles of
// execution and stall for S cycles after every K issues. The scheduler picks
// who issues each cycle; we measure total cycles to drain all warps.
module warp_scheduler_tb #(
    parameter WARPS_PER_CORE = 4,
    parameter POLICY         = 0
) (
    input  wire                        clk,
    input  wire                        reset,
    input  wire [WARPS_PER_CORE-1:0]   warp_ready,
    input  wire [WARPS_PER_CORE-1:0]   warp_progressing,
    input  wire [7:0]                  warp_age0,
    input  wire [7:0]                  warp_age1,
    input  wire [7:0]                  warp_age2,
    input  wire [7:0]                  warp_age3,

    output wire [$clog2(WARPS_PER_CORE+1)-1:0] selected_warp,
    output wire                        issue_valid
);
    wire [7:0] warp_age [WARPS_PER_CORE-1:0];
    assign warp_age[0] = warp_age0;
    assign warp_age[1] = warp_age1;
    assign warp_age[2] = warp_age2;
    assign warp_age[3] = warp_age3;

    warp_scheduler #(
        .WARPS_PER_CORE(WARPS_PER_CORE),
        .POLICY(POLICY)
    ) dut (
        .clk(clk), .reset(reset),
        .warp_ready(warp_ready),
        .warp_progressing(warp_progressing),
        .warp_age(warp_age),
        .selected_warp(selected_warp),
        .issue_valid(issue_valid)
    );
endmodule
