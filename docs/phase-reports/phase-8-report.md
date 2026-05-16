# Phase 8 — Pluggable warp scheduler

## Status
Designed in [ADR-008](../decisions/ADR-008-warp-scheduler.md).

## Why it stops at design
The current architecture has *one warp per core* — LRR/GTO/GTLRR all
degenerate to the same single-warp schedule. Demonstrating the policies
requires a multi-warp-per-core rearchitecture: per-warp PCs, masks, RF
partitions, LSU queues, plus a real `warp_scheduler` module multiplexing
fetch and issue resources.

This is a multi-week change with cross-cutting impact on every per-thread
module. We document the policy implementations and the test plan so the
implementation can land as one contained future PR.

## Test plan (for future)
- Run every kernel under each of `POLICY="LRR" | "GTO" | "GTLRR"`.
- Bench dimensions: mean cycles, std dev across kernels, max-min
  cycles. GTLRR should show the lowest std-dev (predictability is its
  selling point), GTO should show the lowest mean (utilization is its
  selling point), LRR should sit in the middle.
- Reproduce the table in Crouzet et al. 2025 §4 on our tiny suite.

## Reproducibility
No code changes to run in this revision.
