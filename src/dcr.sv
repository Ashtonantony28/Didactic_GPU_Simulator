`default_nettype none
`timescale 1ns/1ns

// Device Control Register
// Stores the kernel-wide thread count.
module dcr (
    input  wire       clk,
    input  wire       reset,
    input  wire       device_control_write_enable,
    input  wire [7:0] device_control_data,
    output wire [7:0] thread_count
);
    reg [7:0] dcr_reg;
    assign thread_count = dcr_reg;

    always @(posedge clk) begin
        if (reset) dcr_reg <= 8'b0;
        else if (device_control_write_enable) dcr_reg <= device_control_data;
    end
endmodule
