`default_nettype none
`timescale 1ns/1ns

// Arithmetic-Logic Unit
// Each thread has its own ALU. Executes ADD/SUB/MUL/DIV/CMP in EXECUTE state.
module alu (
    input  wire        clk,
    input  wire        reset,
    input  wire        enable,
    input  wire [2:0]  core_state,
    input  wire [1:0]  decoded_alu_arithmetic_mux,
    input  wire        decoded_alu_output_mux,
    input  wire [7:0]  rs,
    input  wire [7:0]  rt,
    output wire [7:0]  alu_out
);
    localparam [1:0] ADD = 2'b00, SUB = 2'b01, MUL = 2'b10, DIV = 2'b11;
    localparam [2:0] EXECUTE = 3'b101;

    reg [7:0] alu_out_reg;
    assign alu_out = alu_out_reg;

    always @(posedge clk) begin
        if (reset) begin
            alu_out_reg <= 8'b0;
        end else if (enable && core_state == EXECUTE) begin
            if (decoded_alu_output_mux) begin
                // NZP comparison flags in alu_out[2:0]; bit-for-bit compatible with upstream.
                // Upstream uses (rs - rt > 0) under unsigned arithmetic, which collapses N+P
                // into a single "differs" flag. Kernels (matmul) depend on that quirk.
                alu_out_reg <= {5'b0, (rs - rt > 0), (rs - rt == 0), (rs - rt < 0)};
            end else begin
                case (decoded_alu_arithmetic_mux)
                    ADD: alu_out_reg <= rs + rt;
                    SUB: alu_out_reg <= rs - rt;
                    MUL: alu_out_reg <= rs * rt;
                    DIV: alu_out_reg <= (rt == 8'b0) ? 8'b0 : (rs / rt);
                endcase
            end
        end
    end
endmodule
