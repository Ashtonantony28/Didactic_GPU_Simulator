`default_nettype none
`timescale 1ns/1ns

// Memory Controller
// Routes consumer (LSU/fetcher) requests onto a smaller pool of memory channels.
// When COALESCE = 1, sequential reads to identical addresses are merged: a single
// channel transaction broadcasts its result to every consumer that requested it.
// (Memory coalescing — Phase 5.)
module controller #(
    parameter ADDR_BITS       = 8,
    parameter DATA_BITS       = 16,
    parameter NUM_CONSUMERS   = 4,
    parameter NUM_CHANNELS    = 1,
    parameter WRITE_ENABLE    = 1,
    parameter COALESCE        = 0,
    parameter PRIORITY_ENABLE = 0
) (
    input  wire                          clk,
    input  wire                          reset,

    input  wire [NUM_CONSUMERS-1:0]      consumer_read_valid,
    input  wire [ADDR_BITS-1:0]          consumer_read_address  [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]      consumer_read_ready,
    output reg  [DATA_BITS-1:0]          consumer_read_data     [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]      consumer_write_valid,
    input  wire [ADDR_BITS-1:0]          consumer_write_address [NUM_CONSUMERS-1:0],
    input  wire [DATA_BITS-1:0]          consumer_write_data    [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]      consumer_write_ready,

    // Per-consumer priority bit. Scanned first when PRIORITY_ENABLE=1.
    input  wire [NUM_CONSUMERS-1:0]      consumer_priority,

    output reg  [NUM_CHANNELS-1:0]       mem_read_valid,
    output reg  [ADDR_BITS-1:0]          mem_read_address       [NUM_CHANNELS-1:0],
    input  wire [NUM_CHANNELS-1:0]       mem_read_ready,
    input  wire [DATA_BITS-1:0]          mem_read_data          [NUM_CHANNELS-1:0],
    output reg  [NUM_CHANNELS-1:0]       mem_write_valid,
    output reg  [ADDR_BITS-1:0]          mem_write_address      [NUM_CHANNELS-1:0],
    output reg  [DATA_BITS-1:0]          mem_write_data         [NUM_CHANNELS-1:0],
    input  wire [NUM_CHANNELS-1:0]       mem_write_ready,

    // Telemetry
    output reg  [31:0]                   mem_requests,
    output reg  [31:0]                   mem_requests_coalesced
);
    localparam [2:0]
        IDLE           = 3'b000,
        READ_WAITING   = 3'b010,
        WRITE_WAITING  = 3'b011,
        READ_RELAYING  = 3'b100,
        WRITE_RELAYING = 3'b101;

    reg [2:0]                            controller_state [NUM_CHANNELS-1:0];
    reg [$clog2(NUM_CONSUMERS+1)-1:0]    current_consumer [NUM_CHANNELS-1:0];
    reg [NUM_CONSUMERS-1:0]              channel_serving_consumer;

    // Per-channel coalescing mask: which consumers are being served by this transaction.
    reg [NUM_CONSUMERS-1:0]              channel_coalesce_mask [NUM_CHANNELS-1:0];

    integer i, j;
    integer pass;
    reg     found;
    reg     can_consider;

    always @(posedge clk) begin
        if (reset) begin
            mem_read_valid       <= 0;
            mem_write_valid      <= 0;
            consumer_read_ready  <= 0;
            consumer_write_ready <= 0;
            channel_serving_consumer <= 0;
            mem_requests           <= 0;
            mem_requests_coalesced <= 0;
            for (i = 0; i < NUM_CHANNELS; i = i + 1) begin
                controller_state[i]      <= IDLE;
                current_consumer[i]      <= 0;
                mem_read_address[i]      <= 0;
                mem_write_address[i]     <= 0;
                mem_write_data[i]        <= 0;
                channel_coalesce_mask[i] <= 0;
            end
            for (i = 0; i < NUM_CONSUMERS; i = i + 1) consumer_read_data[i] <= 0;
        end else begin
            for (i = 0; i < NUM_CHANNELS; i = i + 1) begin
                case (controller_state[i])
                    IDLE: begin
                        found = 1'b0;
                        // Two-pass scan when priority is enabled: first only
                        // high-utility (priority) consumers, then everyone.
                        for (pass = 0; pass < (PRIORITY_ENABLE ? 2 : 1); pass = pass + 1) begin
                        for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                            can_consider = !PRIORITY_ENABLE || (pass == 1) || consumer_priority[j];
                            if (!found && can_consider && consumer_read_valid[j] && !channel_serving_consumer[j]) begin
                                channel_serving_consumer[j] <= 1'b1;
                                current_consumer[i]         <= j[$clog2(NUM_CONSUMERS+1)-1:0];
                                mem_read_valid[i]           <= 1'b1;
                                mem_read_address[i]         <= consumer_read_address[j];
                                controller_state[i]         <= READ_WAITING;
                                mem_requests                <= mem_requests + 1;

                                // Coalescing pass: any other consumer reading the same address
                                // in the same cycle rides this transaction.
                                channel_coalesce_mask[i] <= ({{(NUM_CONSUMERS-1){1'b0}}, 1'b1} << j);
                                if (COALESCE) begin : coalesce_block
                                    integer k;
                                    for (k = 0; k < NUM_CONSUMERS; k = k + 1) begin
                                        if (k != j
                                            && consumer_read_valid[k]
                                            && !channel_serving_consumer[k]
                                            && consumer_read_address[k] == consumer_read_address[j]) begin
                                            channel_serving_consumer[k] <= 1'b1;
                                            channel_coalesce_mask[i][k] <= 1'b1;
                                            mem_requests_coalesced <= mem_requests_coalesced + 1;
                                        end
                                    end
                                end
                                found = 1'b1;
                            end
                            if (WRITE_ENABLE && !found && can_consider
                                && consumer_write_valid[j] && !channel_serving_consumer[j]) begin
                                channel_serving_consumer[j] <= 1'b1;
                                current_consumer[i]         <= j[$clog2(NUM_CONSUMERS+1)-1:0];
                                mem_write_valid[i]          <= 1'b1;
                                mem_write_address[i]        <= consumer_write_address[j];
                                mem_write_data[i]           <= consumer_write_data[j];
                                controller_state[i]         <= WRITE_WAITING;
                                mem_requests                <= mem_requests + 1;
                                channel_coalesce_mask[i]    <= ({{(NUM_CONSUMERS-1){1'b0}}, 1'b1} << j);
                                found = 1'b1;
                            end
                        end
                        end
                    end
                    READ_WAITING: if (mem_read_ready[i]) begin
                        mem_read_valid[i] <= 0;
                        // Broadcast to all coalesced consumers.
                        for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                            if (channel_coalesce_mask[i][j]) begin
                                consumer_read_ready[j] <= 1'b1;
                                consumer_read_data[j]  <= mem_read_data[i];
                            end
                        end
                        controller_state[i] <= READ_RELAYING;
                    end
                    WRITE_WAITING: if (mem_write_ready[i]) begin
                        mem_write_valid[i] <= 0;
                        consumer_write_ready[current_consumer[i]] <= 1'b1;
                        controller_state[i] <= WRITE_RELAYING;
                    end
                    READ_RELAYING: begin
                        // Wait until every coalesced consumer has dropped its valid.
                        if (~|(consumer_read_valid & channel_coalesce_mask[i])) begin
                            for (j = 0; j < NUM_CONSUMERS; j = j + 1) begin
                                if (channel_coalesce_mask[i][j]) begin
                                    channel_serving_consumer[j] <= 1'b0;
                                    consumer_read_ready[j]      <= 1'b0;
                                end
                            end
                            channel_coalesce_mask[i] <= 0;
                            controller_state[i]      <= IDLE;
                        end
                    end
                    WRITE_RELAYING: if (!consumer_write_valid[current_consumer[i]]) begin
                        channel_serving_consumer[current_consumer[i]] <= 1'b0;
                        consumer_write_ready[current_consumer[i]]      <= 1'b0;
                        controller_state[i] <= IDLE;
                    end
                    default: ;
                endcase
            end
        end
    end
endmodule
