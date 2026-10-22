# ADR-015 — SIMT stack implementation

## Status
Accepted (PLAN SG21). Implements ADR-011's semantics in RTL behind `gpu.SIMT_STACK` (default 0).

## Context
ADR-011 fixes the encoding (`SYNC` = `BR` with `nzp = 000`) and the stack semantics, with
`xf/isasim.py` (`simt=True`) as the oracle. The core (ADR-014) has one control FSM for all warps and
per-thread registers, NZP/next-PC units and LSUs; a warp's PC lives in `scheduler.sv`.

## Decision
- **One stack per warp** (`src/simt_stack.sv`, instantiated `WARPS_PER_CORE` times in `core.sv`) with
  `LANES`-bit masks: entry 0 is the bottom, `sp` the top, plus the pending reconvergence PC and its
  valid bit. Entries hold `{pc, rpc, mask}` as ADR-011 states; the top entry's `pc` is the warp's PC.
- **Combinational next state, committed at `UPDATE`.** The stack computes, for the warp's current
  instruction, the state after it (push, top move, `RET` removal and the pop loop) and exposes the
  resulting top PC (`simt_next_pc`) and "no lanes left" (`simt_done`). The scheduler takes these in
  place of the lowest lane's `next_pc` and `decoded_ret`, so the FSM and its timing are unchanged:
  the stack adds no cycle, and a block-uniform kernel runs the same cycles with the stack on or off.
- **Branch outcome per lane.** `pc.sv` exports `taken = (NZP & nzp) != 0`; the stack ANDs it with the
  top mask.
- **Lane gating.** A lane outside the top mask has its register file and PC/NZP unit disabled and its
  LSU sees `core_state = IDLE`, so it reads and writes no register, NZP, memory or SMEM and issues no
  request. The mask cannot change while a warp's load or store is in flight (only `UPDATE` changes it),
  so a yielded access completes as in ADR-014.
- **Lanes without a thread.** The bottom mask resets to all ones and every mask is ANDed with
  `lane < thread_count`: `thread_count` arrives with `start`, after the per-block reset.
- **`RET` compaction.** After removing the active lanes, the emptied entries are always a suffix of the
  stack (entries above an entry are its descendants or its taken sibling), so the new top is the
  highest non-empty entry; the warp ends when the bottom is empty.
- **Errors.** Overflow (more than `DEPTH = 8` pushed entries) and a divergent branch with no pending
  `SYNC` are `$fatal` at the `UPDATE` that would commit them. Storage is `DEPTH + 3` entries so the
  overflowing push has somewhere to go before the `$fatal`.
- **BAR** in a divergent region flags the warp as without the stack, as `xf.isasim` does.

## Alternatives rejected
- Holding only non-top entries in the stack and the top in `scheduler.sv`: saves one entry per warp but
  splits each push and pop across two modules.
- A registered (next-cycle) stack update: an extra state or a cycle per branch, so block-uniform
  kernels would no longer keep their cycle counts with the stack on.
- A `SIMT_DEPTH` parameter: the masks of a stack form a laminar family of distinct lane sets, so a warp
  of `LANES` lanes never needs more than `2·LANES − 2` pushed entries; at the default 4 lanes the depth
  of 8 cannot overflow, and 8-lane warps (`THREADS_PER_BLOCK = 8`) reach it.

## Consequences
- `SIMT_STACK = 0` builds no stack (`lane_on` is constant 1) and keeps every cycle count; with
  `SIMT_STACK = 1`, kernels without divergent branches also keep them.
- `make test_divergence` (`test/test_divergence.py`) compares the RTL with `xf.isasim` on the AUDIT-4
  M1 program, nested divergence, per-lane loops, `RET` in a divergent region, a stack filled to exactly
  8 entries, and seeded random structured programs, at `WARPS_PER_CORE` 1/2/4, `NUM_CORES` 1/2/4,
  8-lane warps, and with the stack off; `test/simt_fatal.py` checks both `$fatal`s.
- `test_isa_diff` runs the model with `simt=True` on a `SIMT_STACK = 1` build.
