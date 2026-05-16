# ADR-008 — Pluggable warp scheduler (Phase 8)

## Status
Module implemented and unit-tested. `src/warp_scheduler.sv` + `_tb.sv`
ship LRR / GTO / GTLRR. Three cocotb tests exercise the
distinguishing behavior of each policy. Integration into a multi-warp
core remains the next step.

## Context
The Phase 8 deliverable is multiple warp scheduling policies behind a
common interface:

- **LRR** (loose round-robin) — fair, predictable.
- **GTO** (greedy-then-oldest) — boost the warp making progress.
- **GTLRR** (Crouzet, Carle, Rochange 2025) — coordinated fetch + issue
  for time-predictability.

These differ when there are *multiple warps per core* — a warp scheduler
chooses which warp gets the issue slot each cycle.

## Current architecture and the gap

Today a core runs one block (= one warp = THREADS_PER_BLOCK threads in
lockstep). The "scheduler" we have is a control-flow FSM, not a warp
scheduler. To demonstrate LRR/GTO/GTLRR we need at minimum 2 warps per
core sharing the fetch/decode/issue resources.

## Designed extension

### Multi-warp core

Replace one block per core with `WARPS_PER_CORE` blocks. Each warp has
its own:
- PC
- active mask
- fetch buffer
- register file partition (1/W of total RF entries per warp)
- LSU queue

A new `warp_scheduler` module each cycle picks one warp to:
- Issue to FETCH (front-end scheduler)
- Issue to EXECUTE (back-end scheduler)

### Scheduler interface

```
module warp_scheduler #(
    parameter WARPS_PER_CORE = 2,
    parameter POLICY = "LRR"  // "LRR" | "GTO" | "GTLRR"
) (
    input  wire [WARPS_PER_CORE-1:0] warp_ready,    // which warps can issue
    input  wire [7:0]                warp_age [WARPS_PER_CORE-1:0],
    input  wire [WARPS_PER_CORE-1:0] warp_progressing, // for GTO
    output reg  [$clog2(WARPS_PER_CORE)-1:0] selected_warp,
    output reg                        issue_valid
);
```

### Policy implementations

- **LRR**: maintain a rotating priority pointer; pick the next ready warp
  in order; advance the pointer.
- **GTO**: keep "current" warp; stick with it while it's ready and
  making progress; on stall, pick the *oldest* ready warp.
- **GTLRR**: separate fetch and issue rotation pointers, locked to
  produce the worst-case-execution-time bound from Crouzet et al. 2025.

### Test plan

Run all kernels three times, once per policy. Compare:
- mean cycle count (utilization)
- standard deviation across kernels (predictability)
- max - min cycle count (worst-case sensitivity to scheduling)

## Why not wired

Multi-warp-per-core is a significant rearchitecture. The fetcher,
decoder, register file, and LSU all need a `warp_id` parameter; the
scheduler FSM needs to track multiple warps at different stages
simultaneously. Realistic effort: 2-3 weeks.

## Consequences
- Single-warp performance is unaffected.
- Future implementation has a clean interface contract above.
