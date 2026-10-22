`default_nettype none
`timescale 1ns/1ns

// Program Counter
// Per-thread next-PC compute. NZP latched on CMP UPDATE; BRnzp consults it in EXECUTE.
module pc #(
    parameter WORD_BITS             = 8,
    parameter PROGRAM_MEM_ADDR_BITS = 8
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              enable,
    input  wire [2:0]                        core_state,
    input  wire [2:0]                        decoded_nzp,
    input  wire [7:0]                        decoded_immediate,
    input  wire                              decoded_nzp_write_enable,
    input  wire                              decoded_pc_mux,
    input  wire [WORD_BITS-1:0]              alu_out,
    input  wire [PROGRAM_MEM_ADDR_BITS-1:0]  current_pc,
    output reg  [PROGRAM_MEM_ADDR_BITS-1:0]  next_pc,
    output wire                              taken
);
    localparam [2:0] EXECUTE = 3'b101, UPDATE = 3'b110;

    reg [2:0] nzp;
    assign taken = (nzp & decoded_nzp) != 3'b0;

    always @(posedge clk) begin
        if (reset) begin
            nzp     <= 3'b0;
            next_pc <= 0;
        end else if (enable) begin
            if (core_state == EXECUTE) begin
                if (decoded_pc_mux && ((nzp & decoded_nzp) != 3'b0))
                    next_pc <= decoded_immediate;
                else
                    next_pc <= current_pc + 1;
            end
            if (core_state == UPDATE && decoded_nzp_write_enable)
                nzp <= alu_out[2:0];
        end
    end
endmodule
