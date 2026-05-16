# Phase 3 — Instruction cache

## Status
Done. Parameterised, default-off, full regression green.

## What landed
- `src/fetcher.sv` gains a `generate if (ICACHE_LINES > 0)` block
  implementing a direct-mapped I-cache with 1 instruction per line.
- I-cache hit / miss counters per core; aggregated into
  `icache_hits_total` / `icache_misses_total` on the GPU top.
- Telemetry counters moved onto a separate reset domain
  ([ADR-003](../decisions/ADR-003-telemetry-reset.md)) so per-block
  `core_reset` doesn't wipe them.
- Design choice: [ADR-004](../decisions/ADR-004-icache.md).

## Comparison vs. Phase 2 (ICACHE_LINES=16)

| Kernel       | Phase 2 cycles | Phase 3 cycles | Δ      | icache hit/miss |
|--------------|----------------|----------------|--------|-----------------|
| matadd       | 231            | 231            | —      | 0/26            |
| **matmul**   | **556**        | **504**        | **-9.4%** | **13/28**    |
| vector_dot   | 231            | 231            | —      | 0/26            |
| relu         | 164            | 164            | —      | 0/18            |
| conv1d       | 327            | 327            | —      | 0/40            |
| transpose    | 244            | 244            | —      | 0/32            |

## Discussion
- The loop-heavy kernel (`matmul`) is the only one with hot fetch
  traffic; 13 hits / 28 misses = 32% hit rate.
- Straight-line kernels execute each instruction once per block; with
  two blocks on two cores each having a cold cache, hit rate is 0. To
  catch hits there we'd need either >1 block per core scheduled
  back-to-back (i.e. a kernel with more blocks than cores) or a shared
  cache across cores.

## Reproducibility
```
python3 bench/run.py --phase 3 --label phase-3-icache --params "ICACHE_LINES=16"
```
