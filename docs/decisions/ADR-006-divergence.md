# ADR-006 — Branch divergence (Phase 6)

## Status
Implemented in two steps. Phase 6: data-divergent kernels under one block-wide PC. SG21: SIMT-stack
control divergence behind `gpu.SIMT_STACK` (default 0), with the `SYNC` encoding of
[ADR-011](ADR-011-sync-simt-stack.md) and the RTL of [ADR-015](ADR-015-simt-stack-rtl.md).

## Context
The upstream scheduler advances all threads of a block to one `next_pc`. Threads whose branch
condition disagrees follow the chosen thread (today the warp's lowest thread; AUDIT-4 M1). Real
control divergence needs an active mask that gates writes and memory operations, and a way to
reconverge.

| Level | What it means |
|---|---|
| Data-divergent kernels | Every thread runs the same instructions; per-thread results come from a mask blend |
| SIMT stack | Per-warp active mask and stack of `{pc, rpc, mask}`, reconverging at a marked PC (GPGPU-Sim style) |
| Independent thread scheduling (Volta+) | Threads at the same PC regrouped dynamically; not planned |

## Decision
- **Phase 6** shipped the data-divergent kernel `test/test_divergent_relu.py`, a branch-free program
  that is correct under a single PC.
- **SG21** added the SIMT stack. The reconvergence point comes from the program: `SYNC imm` (`BR` with
  `nzp = 000`, a branch that could never be taken) sets the PC where the next divergent branch
  reconverges. A divergent branch pushes the not-taken and taken sides; the taken side runs first;
  entries pop when their PC reaches the reconvergence PC. ADR-011 gives the exact rules, ADR-015 the RTL.

The Phase 6 design reserved opcode `1101` for an `RC` (reconverge) instruction. That reservation is
void: ADR-010 assigned `1101` to `SRA` and no opcode is free, so the marker reuses the dead `BR nzp=000`
encoding instead.

## Consequences
- With `SIMT_STACK = 0` (default) branches must be uniform within a warp, as `docs/isa.md` states;
  `SYNC` is a no-op.
- With `SIMT_STACK = 1`, kernels without divergent branches keep their cycle counts, and a divergent
  if/else beats the mask blend on the same data (`bench/phase-11-all.json`, section `simt`;
  `docs/final-report.md`).
- `make test_divergence` checks the stack against `xf/isasim.py`.
