# Phase 4 — Pipelining

## Status
Deferred. See [ADR-005](../decisions/ADR-005-phase-4-deferred.md) for the
honest descope reasoning.

## What this would have entailed
The 6-state sequential FSM in `scheduler.sv` would need to become a real
overlapping pipeline. Changes ripple through every module: fetcher
output queue, decoder per-stage registers, register file forwarding
paths, LSU per-stage queues, branch redirect on misprediction.

## What we have instead
- A measurable baseline (Phase 2/3/5 numbers) any future pipelined
  variant can be compared against.
- ADR-005 documents the architectural plan and a three-step
  incremental approach (fetch-decode overlap → early refill → full
  5-stage).

## Reproducibility
No changes to run. Same regression numbers as Phase 3 / 5.
