# ADR-006 — Branch divergence (Phase 6)

## Status
Partial — kernel-level data divergence implemented and tested; SIMT-stack
control divergence designed but not wired.

## Context
The upstream scheduler advances all threads in a block to the same
`next_pc`. Real divergence (different threads at different PCs) requires
per-thread PC plus an "active mask" that gates writes/memops.

## Phase 6 deliverable spectrum

| Level         | What it means                                                | Effort       |
|---------------|--------------------------------------------------------------|--------------|
| Data-divergent kernels | All threads execute the same instructions, but produce different data via mask-blend | Hours |
| SIMT stack     | Per-thread PC + active mask + immediate-post-dominator reconvergence (GPGPU-Sim style) | Weeks |
| Independent thread scheduling (Volta+) | Threads at same PC dynamically grouped | Months |

## Decision

Phase 6 ships the data-divergent kernel (`test/test_divergent_relu.py`),
which compiles to a no-branch program and proves the existing single-PC
SIMT correctly handles per-thread data divergence under masking.

The SIMT-stack design is documented below for a future phase.

## SIMT-stack design (for future implementation)

```
            ┌──────────────┐
            │   warp PC    │  (8 bits)
            │ active_mask  │  (THREADS_PER_BLOCK bits)
            │ reconv_pc    │  (8 bits, post-dominator)
            └──────────────┘
                  ▲
                  │ push on divergent branch
                  │ pop  when warp PC == reconv_pc
            ┌──────────────┐
            │ SIMT stack   │
            │  (8 entries) │
            └──────────────┘
```

`scheduler.sv` would:

- On `BRnzp` in EXECUTE: collect per-thread NZP-match into a `take_mask`.
  If `take_mask` is all-1 or all-0, no divergence — push nothing.
  Otherwise:
  1. Push `{not_take_pc = pc+1, not_take_mask = ~take_mask & active, reconv_pc = post_dominator}`.
  2. Continue with `{pc = imm, mask = take_mask & active}`.
- On UPDATE if `current_pc == top.reconv_pc`: pop the stack and OR the
  popped mask back into `active_mask`.

Per-thread PCs are then `current_pc` for all threads (we keep a single
warp PC + active mask, since all running threads are at the same PC).

Identifying the immediate post-dominator at runtime needs either
compiler markers in the ISA (e.g. `BRnzp.divergent` + paired
`RECONVERGE`) or a control-flow analyzer in the assembler. We'd add an
`RC` (reconverge) opcode and require the assembler/programmer to mark
reconvergence points.

## Consequences

- `test/test_divergent_relu.py` passes under the current single-PC
  scheduler.
- ISA reservation: a future opcode `1101` for `RC` (reconverge) is
  noted.
- Phase 6 bench numbers match Phase 5 (no scheduler change).
