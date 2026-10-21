`default_nettype none
`timescale 1ns/1ns

// L1-D cache (Phase 9 prerequisite)
// Direct-mapped, write-through, 1 word per line. Sits between the per-core
// LSU array (NUM_CONSUMERS lanes) and the downstream memory controller.
//
// Per-lane hit / miss telemetry feeds the MeDiC machinery:
//   - cache_hits[lane]    : counter, total hits since telemetry_reset.
//   - cache_misses[lane]  : counter, total misses.
//   - warp_low_utility[lane]: combinational; 1 when fewer than MEDIC_THRESHOLD
//     of the lane's last 4 lookups hit. Drives bypass.
//
// Windows reset to all hits, so every lane starts high-utility (ADR-009): it
// installs until its misses push it below the threshold. When MEDIC=1 and
// warp_low_utility[lane]=1, the lane's miss fill skips the cache install.
module l1d_cache #(
    parameter NUM_CONSUMERS    = 8,
    parameter ADDR_BITS        = 8,
    parameter DATA_BITS        = 8,
    parameter LINES            = 16,
    parameter MEDIC            = 0,
    parameter MEDIC_THRESHOLD  = 2    // hits in window < this → low utility
) (
    input  wire                              clk,
    input  wire                              reset,
    input  wire                              telemetry_reset,

    // Consumer side (one port per lane)
    input  wire [NUM_CONSUMERS-1:0]          consumer_read_valid,
    input  wire [ADDR_BITS-1:0]              consumer_read_address  [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]          consumer_read_ready,
    output reg  [DATA_BITS-1:0]              consumer_read_data     [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]          consumer_write_valid,
    input  wire [ADDR_BITS-1:0]              consumer_write_address [NUM_CONSUMERS-1:0],
    input  wire [DATA_BITS-1:0]              consumer_write_data    [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]          consumer_write_ready,

    // Downstream (controller) side
    output reg  [NUM_CONSUMERS-1:0]          mem_read_valid,
    output reg  [ADDR_BITS-1:0]              mem_read_address       [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]          mem_read_ready,
    input  wire [DATA_BITS-1:0]              mem_read_data          [NUM_CONSUMERS-1:0],
    output reg  [NUM_CONSUMERS-1:0]          mem_write_valid,
    output reg  [ADDR_BITS-1:0]              mem_write_address      [NUM_CONSUMERS-1:0],
    output reg  [DATA_BITS-1:0]              mem_write_data         [NUM_CONSUMERS-1:0],
    input  wire [NUM_CONSUMERS-1:0]          mem_write_ready,

    // Telemetry
    output wire [31:0]                       cache_hits_total,
    output wire [31:0]                       cache_misses_total,
    output wire [NUM_CONSUMERS-1:0]          warp_low_utility,

    // Per-lane utility (drives MeDiC priority logic upstream)
    output wire [NUM_CONSUMERS-1:0]          warp_high_utility
);
    localparam INDEX_BITS = (LINES <= 1) ? 1 : $clog2(LINES);
    localparam TAG_BITS   = (ADDR_BITS > INDEX_BITS) ? (ADDR_BITS - INDEX_BITS) : 1;

    // Cache storage
    reg                  valid_arr [0:LINES-1];
    reg [TAG_BITS-1:0]   tag_arr   [0:LINES-1];
    reg [DATA_BITS-1:0]  data_arr  [0:LINES-1];

    // Per-lane state. Modeled as a simple FSM:
    //   0 IDLE / hit_serve, 1 MISS_READ, 2 MISS_FILL, 3 WRITE_THROUGH
    reg [1:0] state_arr [NUM_CONSUMERS-1:0];
    localparam [1:0] L_IDLE = 2'b00, L_MISS = 2'b01, L_FILL = 2'b10, L_WT = 2'b11;

    // Per-lane rolling hit/miss window (last 4 lookups).
    reg [3:0] hit_window  [NUM_CONSUMERS-1:0];
    reg [3:0] miss_window [NUM_CONSUMERS-1:0];

    // Counters for telemetry
    reg [31:0] hits_total, misses_total;
    assign cache_hits_total   = hits_total;
    assign cache_misses_total = misses_total;

    // Combinational utility classification
    genvar lu;
    generate
        for (lu = 0; lu < NUM_CONSUMERS; lu = lu + 1) begin : util
            wire [4:0] h = {1'b0, hit_window[lu][3]} + {1'b0, hit_window[lu][2]} + {1'b0, hit_window[lu][1]} + {1'b0, hit_window[lu][0]};
            assign warp_low_utility[lu]  = (h < MEDIC_THRESHOLD);
            assign warp_high_utility[lu] = (h >= MEDIC_THRESHOLD);
        end
    endgenerate

    integer i, j;
    integer hit_inc, miss_inc;

    // Combinational hit detection for IDLE-state lanes
    function automatic [INDEX_BITS-1:0] idx_of;
        input [ADDR_BITS-1:0] a;
        idx_of = a[INDEX_BITS-1:0];
    endfunction
    function automatic [TAG_BITS-1:0] tag_of;
        input [ADDR_BITS-1:0] a;
        tag_of = a[ADDR_BITS-1:INDEX_BITS];
    endfunction

    always @(posedge clk) begin
        if (reset) begin
            for (i = 0; i < LINES; i = i + 1) begin
                valid_arr[i] <= 1'b0;
                tag_arr[i]   <= {TAG_BITS{1'b0}};
                data_arr[i]  <= {DATA_BITS{1'b0}};
            end
            for (i = 0; i < NUM_CONSUMERS; i = i + 1) begin
                state_arr[i]            <= L_IDLE;
                consumer_read_ready[i]  <= 1'b0;
                consumer_read_data[i]   <= {DATA_BITS{1'b0}};
                consumer_write_ready[i] <= 1'b0;
                mem_read_valid[i]       <= 1'b0;
                mem_read_address[i]     <= {ADDR_BITS{1'b0}};
                mem_write_valid[i]      <= 1'b0;
                mem_write_address[i]    <= {ADDR_BITS{1'b0}};
                mem_write_data[i]       <= {DATA_BITS{1'b0}};
                hit_window[i]           <= 4'b1111;
                miss_window[i]          <= 4'b0;
            end
            hits_total   <= 32'b0;
            misses_total <= 32'b0;
        end else begin
            if (telemetry_reset) begin
                hits_total   <= 32'b0;
                misses_total <= 32'b0;
                for (i = 0; i < NUM_CONSUMERS; i = i + 1) begin
                    hit_window[i]  <= 4'b1111;
                    miss_window[i] <= 4'b0;
                end
            end
            hit_inc  = 0;
            miss_inc = 0;
            for (i = 0; i < NUM_CONSUMERS; i = i + 1) begin
                case (state_arr[i])
                    L_IDLE: begin
                        // Read path
                        if (consumer_read_valid[i] && !consumer_read_ready[i]) begin
                            if (valid_arr[idx_of(consumer_read_address[i])]
                                && tag_arr[idx_of(consumer_read_address[i])] == tag_of(consumer_read_address[i])) begin
                                // HIT
                                consumer_read_data[i]  <= data_arr[idx_of(consumer_read_address[i])];
                                consumer_read_ready[i] <= 1'b1;
                                hit_inc                = hit_inc + 1;
                                hit_window[i]          <= {hit_window[i][2:0], 1'b1};
                                miss_window[i]         <= {miss_window[i][2:0], 1'b0};
                            end else begin
                                // MISS — issue downstream read
                                mem_read_valid[i]    <= 1'b1;
                                mem_read_address[i]  <= consumer_read_address[i];
                                state_arr[i]         <= L_MISS;
                                miss_inc             = miss_inc + 1;
                                hit_window[i]        <= {hit_window[i][2:0], 1'b0};
                                miss_window[i]       <= {miss_window[i][2:0], 1'b1};
                            end
                        end else if (consumer_read_ready[i] && !consumer_read_valid[i]) begin
                            consumer_read_ready[i] <= 1'b0;
                        end
                        // Write path — write-through with cache update
                        if (consumer_write_valid[i] && !consumer_write_ready[i]) begin
                            mem_write_valid[i]    <= 1'b1;
                            mem_write_address[i]  <= consumer_write_address[i];
                            mem_write_data[i]     <= consumer_write_data[i];
                            state_arr[i]          <= L_WT;
                        end else if (consumer_write_ready[i] && !consumer_write_valid[i]) begin
                            consumer_write_ready[i] <= 1'b0;
                        end
                    end
                    L_MISS: if (mem_read_ready[i]) begin
                        mem_read_valid[i]      <= 1'b0;
                        consumer_read_data[i]  <= mem_read_data[i];
                        consumer_read_ready[i] <= 1'b1;
                        // Install on miss — unless MeDiC says bypass.
                        if (!(MEDIC && warp_low_utility[i])) begin
                            valid_arr[idx_of(mem_read_address[i])] <= 1'b1;
                            tag_arr  [idx_of(mem_read_address[i])] <= tag_of(mem_read_address[i]);
                            data_arr [idx_of(mem_read_address[i])] <= mem_read_data[i];
                        end
                        state_arr[i] <= L_IDLE;
                    end
                    L_FILL: state_arr[i] <= L_IDLE;
                    L_WT: if (mem_write_ready[i]) begin
                        mem_write_valid[i]      <= 1'b0;
                        consumer_write_ready[i] <= 1'b1;
                        // Write-through: also update cache if hit
                        if (valid_arr[idx_of(mem_write_address[i])]
                            && tag_arr[idx_of(mem_write_address[i])] == tag_of(mem_write_address[i])) begin
                            data_arr[idx_of(mem_write_address[i])] <= mem_write_data[i];
                        end
                        state_arr[i] <= L_IDLE;
                    end
                endcase
            end
            // Apply aggregated counts (idiomatic non-blocking on accumulated value).
            hits_total   <= hits_total   + hit_inc[31:0];
            misses_total <= misses_total + miss_inc[31:0];
        end
    end
endmodule
