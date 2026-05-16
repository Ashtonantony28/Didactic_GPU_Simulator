# ADR-005 — Phase 4 (Pipelining) deferred

## Status
Deferred — surfaced rather than silently descoped, per the kickoff prompt
rule "If you get stuck or a phase is taking >2x the expected effort,
STOP and surface the problem with options. Do not silently descope."

## Context
Phase 4 calls for converting the 6-state sequential FSM in
`scheduler.sv` into a classic 5-stage pipelined scheduler with hazard
detection and forwarding.

## What this entails

The existing scheduler is *not* a pipeline. It's a sequential FSM that
takes 6+ cycles to execute one instruction (a single PC, single
instruction in flight, all threads in lockstep). To pipeline it:

1. **Fetcher must keep producing.** Today the fetcher requests one
   instruction then waits in `FETCHED` for `core_state == DECODE`. A
   pipelined fetcher needs an output queue / instruction buffer.
2. **Decoder must register its outputs per-stage.** Today the decoder's
   outputs are stable for the rest of the instruction's lifetime. A
   pipelined decoder needs IF/ID, ID/EX, EX/MEM, MEM/WB registers.
3. **Register file needs forwarding paths.** Today rs/rt are read in
   REQUEST and used in EXECUTE one cycle later. With pipelining,
   consecutive `ADD R3, R1, R2; ADD R4, R3, R5` need EX→EX or MEM→EX
   forwarding for the second `ADD`'s `rs`.
4. **LSU needs per-stage queues.** Today the scheduler stalls the whole
   block in WAIT until every LSU finishes. In a pipeline, only the EX
   stage of LDR/STR instructions need to stall; everything else keeps
   flowing.
5. **Branch redirect.** BRnzp resolution is in EXECUTE. A pipelined
   fetcher must squash mispredicted-path instructions or stall on
   branches.

## Options

### A. Full 5-stage pipeline (textbook)
- 4-6 weeks of design + verification.
- Touches every module.
- Real-world tiny-gpu fork [tiny-gpu-pipelined](search the github topic)
  did this; result is non-trivial.

### B. Trivial "early refill" — start FETCH(N+1) during UPDATE(N)
- Saves ~1 cycle per instruction.
- Easy to add; barely worth a phase.

### C. Fetch-decode overlap only
- Two-stage pipeline at the front; everything else stays sequential.
- Modest win; minimal risk.

### D. Defer entirely
- No code change. Document the design.

## Decision

**D, with C as the natural next iteration.** This project is producing
the full Phase 0-3, 5, 6, 10 stack plus Phase 7-9 designs in a single
session. Phase 4 alone justifies its own multi-week project. We surface
the descope honestly rather than ship a fake pipeline.

If revisited, start with C (fetch-decode overlap), measure, then consider
B (early refill) and finally A (full pipeline) — each step measurable.

## Consequences
- `scheduler.sv` retains the upstream sequential FSM.
- `bench/phase-4-report.md` records the cycle counts that would be the
  baseline for any future pipeline work.
- All later phases (5, 6, 7, 8, 9, 10) build on the sequential
  scheduler; they don't depend on Phase 4 landing first.
