`default_nettype none
`timescale 1ns/1ns

// Register File (per thread)
// R0..R12 general-purpose, R13=%blockIdx, R14=%blockDim, R15=%threadIdx (read-only).
module registers #(
    parameter THREADS_PER_BLOCK = 4,
    parameter THREAD_ID         = 0,
    parameter WORD_BITS         = 8
) (
    input  wire                  clk,
    input  wire                  reset,
    input  wire                  enable,
    input  wire [7:0]            block_id,
    input  wire [2:0]            core_state,
    input  wire [3:0]            decoded_rd_address,
    input  wire [3:0]            decoded_rs_address,
    input  wire [3:0]            decoded_rt_address,
    input  wire                  decoded_reg_write_enable,
    input  wire [1:0]            decoded_reg_input_mux,
    input  wire [7:0]            decoded_immediate,
    input  wire [WORD_BITS-1:0]  alu_out,
    input  wire [WORD_BITS-1:0]  lsu_out,
    output reg  [WORD_BITS-1:0]  rs,
    output reg  [WORD_BITS-1:0]  rt
);
    localparam [1:0] ARITH = 2'b00, MEM = 2'b01, CONSTANT = 2'b10, SHARED = 2'b11;
    localparam [2:0] REQUEST = 3'b011, UPDATE = 3'b110;

    reg [WORD_BITS-1:0] registers [15:0];
    integer i;

    always @(posedge clk) begin
        if (reset) begin
            rs <= 0;
            rt <= 0;
            for (i = 0; i < 13; i = i + 1) registers[i] <= 0;
            registers[13] <= 0;
            registers[14] <= THREADS_PER_BLOCK;
            registers[15] <= THREAD_ID;
        end else if (enable) begin
            registers[13] <= block_id;
            if (core_state == REQUEST) begin
                rs <= registers[decoded_rs_address];
                rt <= registers[decoded_rt_address];
            end
            if (core_state == UPDATE && decoded_reg_write_enable && decoded_rd_address < 13) begin
                case (decoded_reg_input_mux)
                    ARITH:    registers[decoded_rd_address] <= alu_out;
                    MEM:      registers[decoded_rd_address] <= lsu_out;
                    CONSTANT: registers[decoded_rd_address] <= decoded_immediate;
                    SHARED:   registers[decoded_rd_address] <= lsu_out;
                endcase
            end
        end
    end
endmodule
