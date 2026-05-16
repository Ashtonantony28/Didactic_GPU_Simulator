`default_nettype none
`timescale 1ns/1ns

// Instruction Fetcher with optional I-cache (Phase 3).
// ICACHE_LINES = 0 → cache bypassed (behaves like upstream).
// telemetry_reset drives a separate global domain so the per-block `reset`
// doesn't wipe icache_hits / icache_misses.
module fetcher #(
    parameter PROGRAM_MEM_ADDR_BITS = 8,
    parameter PROGRAM_MEM_DATA_BITS = 16,
    parameter ICACHE_LINES = 0
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              telemetry_reset,
    input  wire [2:0]                        core_state,
    input  wire [PROGRAM_MEM_ADDR_BITS-1:0]  current_pc,

    output reg                               mem_read_valid,
    output reg  [PROGRAM_MEM_ADDR_BITS-1:0]  mem_read_address,
    input  wire                              mem_read_ready,
    input  wire [PROGRAM_MEM_DATA_BITS-1:0]  mem_read_data,

    output reg  [2:0]                        fetcher_state,
    output reg  [PROGRAM_MEM_DATA_BITS-1:0]  instruction,

    output reg  [31:0]                       icache_hits,
    output reg  [31:0]                       icache_misses
);
    localparam [2:0] IDLE = 3'b000, FETCHING = 3'b001, FETCHED = 3'b010;
    localparam [2:0] CORE_FETCH = 3'b001, CORE_DECODE = 3'b010, CORE_REQUEST = 3'b011;

    // 1-cycle pulse: a fetch was initiated this cycle.
    reg fetch_initiated;
    reg fetch_was_hit;

    always @(posedge clk) begin
        if (telemetry_reset) begin
            icache_hits   <= 0;
            icache_misses <= 0;
        end else if (fetch_initiated) begin
            if (fetch_was_hit) icache_hits   <= icache_hits   + 1;
            else               icache_misses <= icache_misses + 1;
        end
    end

    generate
        if (ICACHE_LINES == 0) begin : no_icache
            always @(posedge clk) begin
                if (reset) begin
                    fetcher_state    <= IDLE;
                    mem_read_valid   <= 0;
                    mem_read_address <= 0;
                    instruction      <= {PROGRAM_MEM_DATA_BITS{1'b0}};
                    fetch_initiated  <= 1'b0;
                    fetch_was_hit    <= 1'b0;
                end else begin
                    fetch_initiated <= 1'b0;
                    case (fetcher_state)
                        IDLE: if (core_state == CORE_FETCH) begin
                            fetcher_state    <= FETCHING;
                            mem_read_valid   <= 1;
                            mem_read_address <= current_pc;
                            fetch_initiated  <= 1'b1;
                            fetch_was_hit    <= 1'b0;
                        end
                        FETCHING: if (mem_read_ready) begin
                            fetcher_state  <= FETCHED;
                            instruction    <= mem_read_data;
                            mem_read_valid <= 0;
                        end
                        // Move out of FETCHED when the scheduler advances past FETCH.
                        // In pipelined mode the scheduler skips DECODE → REQUEST.
                        FETCHED: if (core_state == CORE_DECODE || core_state == CORE_REQUEST) fetcher_state <= IDLE;
                        default: ;
                    endcase
                end
            end
        end else begin : icache
            localparam INDEX_BITS = (ICACHE_LINES <= 1) ? 1 : $clog2(ICACHE_LINES);
            localparam TAG_BITS   = (PROGRAM_MEM_ADDR_BITS > INDEX_BITS)
                                    ? (PROGRAM_MEM_ADDR_BITS - INDEX_BITS) : 1;

            reg                              valid_arr [0:ICACHE_LINES-1];
            reg [TAG_BITS-1:0]               tag_arr   [0:ICACHE_LINES-1];
            reg [PROGRAM_MEM_DATA_BITS-1:0]  data_arr  [0:ICACHE_LINES-1];

            wire [INDEX_BITS-1:0] idx = current_pc[INDEX_BITS-1:0];
            wire [TAG_BITS-1:0]   tag = current_pc[PROGRAM_MEM_ADDR_BITS-1:INDEX_BITS];
            wire                  hit = valid_arr[idx] && (tag_arr[idx] == tag);

            integer i;
            always @(posedge clk) begin
                if (reset) begin
                    fetcher_state    <= IDLE;
                    mem_read_valid   <= 0;
                    mem_read_address <= 0;
                    instruction      <= {PROGRAM_MEM_DATA_BITS{1'b0}};
                    fetch_initiated  <= 1'b0;
                    fetch_was_hit    <= 1'b0;
                end else if (telemetry_reset) begin
                    // wipe cache state too on a fresh kernel launch
                    for (i = 0; i < ICACHE_LINES; i = i + 1) begin
                        valid_arr[i] <= 1'b0;
                        tag_arr[i]   <= {TAG_BITS{1'b0}};
                        data_arr[i]  <= {PROGRAM_MEM_DATA_BITS{1'b0}};
                    end
                end else begin
                    fetch_initiated <= 1'b0;
                    case (fetcher_state)
                        IDLE: if (core_state == CORE_FETCH) begin
                            if (hit) begin
                                instruction     <= data_arr[idx];
                                fetcher_state   <= FETCHED;
                                fetch_initiated <= 1'b1;
                                fetch_was_hit   <= 1'b1;
                            end else begin
                                fetcher_state    <= FETCHING;
                                mem_read_valid   <= 1;
                                mem_read_address <= current_pc;
                                fetch_initiated  <= 1'b1;
                                fetch_was_hit    <= 1'b0;
                            end
                        end
                        FETCHING: if (mem_read_ready) begin
                            fetcher_state  <= FETCHED;
                            instruction    <= mem_read_data;
                            mem_read_valid <= 0;
                            valid_arr[idx] <= 1'b1;
                            tag_arr[idx]   <= tag;
                            data_arr[idx]  <= mem_read_data;
                        end
                        // Move out of FETCHED when the scheduler advances past FETCH.
                        // In pipelined mode the scheduler skips DECODE → REQUEST.
                        FETCHED: if (core_state == CORE_DECODE || core_state == CORE_REQUEST) fetcher_state <= IDLE;
                        default: ;
                    endcase
                end
            end
        end
    endgenerate
endmodule
