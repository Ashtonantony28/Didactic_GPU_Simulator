`default_nettype none
`timescale 1ns/1ns

// Instruction Decoder
// Combinational from the 16-bit instruction. The original tiny-gpu registered
// these on `core_state == DECODE`; making decode combinational lets the
// pipelined scheduler skip the DECODE state without losing setup time.
// Behavior is identical for the non-pipelined scheduler — decoded outputs
// are stable as soon as the instruction is latched into the fetcher.
module decoder (
    input  wire        clk,
    input  wire        reset,
    input  wire [2:0]  core_state,
    input  wire [15:0] instruction,
    output reg  [3:0]  decoded_rd_address,
    output reg  [3:0]  decoded_rs_address,
    output reg  [3:0]  decoded_rt_address,
    output reg  [2:0]  decoded_nzp,
    output reg  [7:0]  decoded_immediate,
    output reg         decoded_reg_write_enable,
    output reg         decoded_mem_read_enable,
    output reg         decoded_mem_write_enable,
    output reg         decoded_nzp_write_enable,
    output reg  [1:0]  decoded_reg_input_mux,
    output reg  [1:0]  decoded_alu_arithmetic_mux,
    output reg         decoded_alu_output_mux,
    output reg         decoded_pc_mux,
    output reg         decoded_ret,
    output reg         decoded_smem_read_enable,
    output reg         decoded_smem_write_enable,
    output reg         decoded_bar
);
    localparam [3:0]
        OP_NOP   = 4'b0000,
        OP_BRnzp = 4'b0001,
        OP_CMP   = 4'b0010,
        OP_ADD   = 4'b0011,
        OP_SUB   = 4'b0100,
        OP_MUL   = 4'b0101,
        OP_DIV   = 4'b0110,
        OP_LDR   = 4'b0111,
        OP_STR   = 4'b1000,
        OP_CONST = 4'b1001,
        OP_LDS   = 4'b1010,
        OP_STS   = 4'b1011,
        OP_BAR   = 4'b1100,
        OP_RET   = 4'b1111;

    always @(*) begin
        decoded_rd_address         = instruction[11:8];
        decoded_rs_address         = instruction[7:4];
        decoded_rt_address         = instruction[3:0];
        decoded_immediate          = instruction[7:0];
        decoded_nzp                = instruction[11:9];

        decoded_reg_write_enable   = 1'b0;
        decoded_mem_read_enable    = 1'b0;
        decoded_mem_write_enable   = 1'b0;
        decoded_nzp_write_enable   = 1'b0;
        decoded_reg_input_mux      = 2'b00;
        decoded_alu_arithmetic_mux = 2'b00;
        decoded_alu_output_mux     = 1'b0;
        decoded_pc_mux             = 1'b0;
        decoded_ret                = 1'b0;
        decoded_smem_read_enable   = 1'b0;
        decoded_smem_write_enable  = 1'b0;
        decoded_bar                = 1'b0;

        case (instruction[15:12])
            OP_NOP: ;
            OP_BRnzp: decoded_pc_mux = 1'b1;
            OP_CMP: begin
                decoded_alu_output_mux   = 1'b1;
                decoded_nzp_write_enable = 1'b1;
            end
            OP_ADD: begin
                decoded_reg_write_enable   = 1'b1;
                decoded_alu_arithmetic_mux = 2'b00;
            end
            OP_SUB: begin
                decoded_reg_write_enable   = 1'b1;
                decoded_alu_arithmetic_mux = 2'b01;
            end
            OP_MUL: begin
                decoded_reg_write_enable   = 1'b1;
                decoded_alu_arithmetic_mux = 2'b10;
            end
            OP_DIV: begin
                decoded_reg_write_enable   = 1'b1;
                decoded_alu_arithmetic_mux = 2'b11;
            end
            OP_LDR: begin
                decoded_reg_write_enable = 1'b1;
                decoded_reg_input_mux    = 2'b01;
                decoded_mem_read_enable  = 1'b1;
            end
            OP_STR: decoded_mem_write_enable = 1'b1;
            OP_CONST: begin
                decoded_reg_write_enable = 1'b1;
                decoded_reg_input_mux    = 2'b10;
            end
            OP_LDS: begin
                decoded_reg_write_enable  = 1'b1;
                decoded_reg_input_mux     = 2'b11;
                decoded_smem_read_enable  = 1'b1;
            end
            OP_STS: decoded_smem_write_enable = 1'b1;
            OP_BAR: decoded_bar = 1'b1;
            OP_RET: decoded_ret = 1'b1;
            default: ;
        endcase
    end

endmodule
