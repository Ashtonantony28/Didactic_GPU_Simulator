# Phase 5 — Memory coalescing

## Status
Done.

## What landed
- `src/controller.sv` gains a `COALESCE` parameter. When set, the
  channel-idle scan collects every other consumer that wants the same
  address as the chosen one into a coalescing mask; one channel
  transaction services them all.
- Telemetry: `data_mem_requests_coalesced` counts the *additional*
  consumers piggybacking on another's transaction.

## Comparison vs. Phase 3 (ICACHE_LINES=16, COALESCE=1)

| Kernel       | Phase 3 cycles | Phase 5 cycles | mem_req | coal | Δ cycles |
|--------------|----------------|----------------|---------|------|----------|
| matadd       | 231            | 231            | 24      | 0    | —        |
| **matmul**   | **504**        | **464**        | **12**  | **8** | **-7.9%** |
| vector_dot   | 231            | 231            | 24      | 0    | —        |
| relu         | 164            | 164            | 16      | 0    | —        |
| conv1d       | 327            | 327            | 24      | 0    | —        |
| transpose    | 244            | 244            | 16      | 0    | —        |

## Discussion
- Coalescing fires only on **same-address** loads in a single cycle.
  `matmul` benefits because the inner-loop `LDR R10, R10` (loading
  `A[i]` for the row) is the same address across columns of a row, so
  threads in the same warp coalesce.
- The other kernels have unit-stride loads — threads load
  `&in[i]`, which are different addresses. To coalesce *strided*
  accesses we'd need either wider memory channels (with a single
  channel transaction returning multiple bytes) or a cache line concept.
- 12 effective requests + 8 coalesced = 20 total LSU requests (matches
  Phase 3 count of 20). The work is the same; we just do it faster.

## Reproducibility
```
python3 bench/run.py --phase 5 --label phase-5-coalesce --params "ICACHE_LINES=16 COALESCE=1"
```
