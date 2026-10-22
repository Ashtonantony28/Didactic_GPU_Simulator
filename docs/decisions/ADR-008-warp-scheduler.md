# ADR-008 — Pluggable warp scheduler (Phase 8)

## Status
Implemented. Phase 8 built `src/warp_scheduler.sv` (LRR / GTO / GTLRR) as a standalone module with a
testbench; since SG20 every core instantiates it at `WARPS_PER_CORE > 1`, selected by
`gpu.WARP_POLICY` (0 LRR, 1 GTO, 2 GTLRR). The multi-warp core is [ADR-012](ADR-012-multi-warp.md)
(contract) and [ADR-014](ADR-014-multi-warp-core.md) (implementation).

## Context
A warp scheduler picks which warp gets the issue slot. It only matters with several warps per core,
which SG20 added: a block of `THREADS_PER_BLOCK` threads runs as `WARPS_PER_CORE` warps that
time-share the core's lanes and its single fetch/decode front end.

## Interface (unchanged since Phase 8)

```
module warp_scheduler #(parameter WARPS_PER_CORE = 4, parameter POLICY = 0) (
    input  wire clk, reset,
    input  wire [WARPS_PER_CORE-1:0] warp_ready,        // can issue now
    input  wire [WARPS_PER_CORE-1:0] warp_progressing,  // ready and issued last
    input  wire [7:0]                warp_age [WARPS_PER_CORE-1:0],  // instructions issued
    output reg  [$clog2(WARPS_PER_CORE+1)-1:0] selected_warp,
    output reg                                 issue_valid);
```

The core drives the inputs only in its issue slot (ADR-014), so each pointer moves once per issue.

## Policies as implemented
- **LRR**: the first ready warp at or after a rotating pointer; the pointer moves past the warp issued.
- **GTO**: keep the last warp while it is ready and progressing; otherwise the ready warp with the
  smallest age (ties to the lowest index).
- **GTLRR**: round-robin over warps that are ready and progressing, else plain LRR. Since only the last
  warp issued can be progressing, this is greedy with a round-robin fallback. It is named after
  Crouzet, Carle and Rochange (2025) but does not implement their coordinated fetch and issue pointers:
  the core has one front end and one instruction in flight, so there is no separate fetch stage to
  coordinate.

## Results
On the phase-11 multi-warp kernels (`bench/phase-11-all.json`, sections `warps` and `warps_icache`;
`docs/final-report.md`): with data-memory latency only, GTO and GTLRR hide all of it and LRR most of
it; with fetch latency and a shared I-cache, LRR is faster. GTO and GTLRR give identical cycle counts in
every row; at 2 warps they coincide by construction, and they differ only in which warp follows a stall
(`test_multiwarp::test_policy_ages`).

## Tests
- `make test_warp_scheduler_{lrr,gto,gtlrr}`: the module alone, policy-specific issue orders.
- `make test_multiwarp`: the same orders inside the core, at `WARPS_PER_CORE = 4` under each policy and
  at 2.

## Consequences
- `WARPS_PER_CORE = 1` instantiates no scheduler and keeps every cycle count (ADR-014).
- GTO's age input saturates at 254 (ADR-014); the module's search starts from 255 with a strict `<`.
