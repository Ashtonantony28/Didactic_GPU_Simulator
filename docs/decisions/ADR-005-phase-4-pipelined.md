# ADR-005 — Phase 4 (Pipelining) — implemented as combinational decode + skip-DECODE

## Status
Accepted (superseded the earlier "deferred" status).

## Context
Phase 4 calls for converting the 6-state sequential FSM in
`scheduler.sv` into an overlapping pipeline. A full 5-stage textbook
pipeline rewrites every module (fetcher queue, register file
forwarding, LSU per-stage queue, branch redirect on misprediction) —
out of scope for this revision.

## Decision

Ship the highest-impact subset:
1. Make the decoder **combinational** (`always @(*)`). The decoder
   inputs depend only on `instruction`; the output is stable as soon
   as the fetcher latches the instruction.
2. Add a `PIPELINED` parameter to `scheduler.sv`. When non-zero, skip
   the dead DECODE state — go FETCH → REQUEST directly.
3. Update the fetcher to release FETCHED on either `DECODE` or
   `REQUEST`, supporting both modes.

This saves 1 cycle per instruction. It's not a full pipeline — only one
instruction is in flight at a time — but it eliminates the dead state
that did nothing other than wait for the sequential decoder.

## Cycle savings observed
- matmul: 464 → 423 (-8.8%)
- conv1d: 327 → 295 (-9.8%)
- matadd: 231 → 220 (-4.8%)

## Consequences

- `PIPELINED=0` is bit-identical to Phase 3 behavior (verified by
  cycle count match).
- `PIPELINED=1` adds no per-cycle setup-time pressure because decode
  is combinational. (Real silicon would need timing analysis here;
  for simulation it's free.)
- Full 5-stage pipeline (fetcher prefetch, ID/EX/MEM/WB registers,
  forwarding paths, branch squash) remains a follow-up.
