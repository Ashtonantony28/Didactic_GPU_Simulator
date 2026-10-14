`default_nettype none
`timescale 1ns/1ns

// Arithmetic-Logic Unit
// Each thread has its own ALU. Executes ADD/SUB/MUL/DIV/SRA/MAX/CMP in EXECUTE state.
// DIV is unsigned; SRA and MAX treat operands as WORD_BITS two's complement.
module alu #(
    parameter WORD_BITS = 8
) (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [2:0]  core_state,
    input  wire [2:0]  decoded_alu_arithmetic_mux,
    input  wire        decoded_alu_output_mux,
    input  wire [WORD_BITS-1:0] rs,
    input  wire [WORD_BITS-1:0] rt,
    output wire [WORD_BITS-1:0] alu_out
);
    localparam [2:0] ADD = 3'b000, SUB = 3'b001, MUL = 3'b010, DIV = 3'b011, SRA = 3'b100, MAX = 3'b101;
    localparam [2:0] EXECUTE = 3'b101;

    reg [WORD_BITS-1:0] alu_out_reg;
    assign alu_out = alu_out_reg;

    always @(posedge clk) begin
        if (reset) begin
            alu_out_reg <= 0;
        end else if (enable && core_state == EXECUTE) begin
            if (decoded_alu_output_mux) begin
                // NZP comparison flags in alu_out[2:0]; bit-for-bit compatible with upstream.
                // Upstream uses (rs - rt > 0) under unsigned arithmetic, which collapses N+P
                // into a single "differs" flag. Kernels (matmul) depend on that quirk.
                alu_out_reg <= {{(WORD_BITS-3){1'b0}}, (rs - rt > 0), (rs - rt == 0), (rs - rt < 0)};
            end else begin
                case (decoded_alu_arithmetic_mux)
                    ADD: alu_out_reg <= rs + rt;
                    SUB: alu_out_reg <= rs - rt;
                    MUL: alu_out_reg <= rs * rt;
                    DIV: alu_out_reg <= (rt == 0) ? 0 : (rs / rt);
                    SRA: alu_out_reg <= $signed(rs) >>> rt;
                    MAX: alu_out_reg <= ($signed(rs) > $signed(rt)) ? rs : rt;
                    default: alu_out_reg <= 0;
                endcase
            end
        end
    end
endmodule
