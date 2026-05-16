# Phase 2 — Bench harness + extra kernels

## Status
Done.

## What landed
- `bench/run.py` — drives every kernel test, snapshots metrics as JSON.
- Per-kernel results in `bench/results/<name>.json`.
- Phase rollup in `bench/phase-2-baseline.json`.
- New kernels: `vector_dot`, `relu`, `conv1d`, `transpose`.
- Telemetry counters in the GPU top: `data_mem_requests`,
  `data_mem_requests_coalesced`, `icache_hits_total`,
  `icache_misses_total`.

## Phase-2 baseline numbers
| Kernel       | cycles | mem_req | mem_req_coal | icache hit/miss |
|--------------|--------|---------|--------------|-----------------|
| matadd       | 231    | 24      | 0            | 0/0             |
| matmul       | 556    | 20      | 0            | 0/0             |
| vector_dot   | 231    | 24      | 0            | 0/0             |
| relu         | 164    | 16      | 0            | 0/0             |
| conv1d       | 327    | 24      | 0            | 0/0             |
| transpose    | 244    | 16      | 0            | 0/0             |

All pass.

## Notes
- `vector_dot` doesn't sum to a single scalar — tiny-gpu has no atomics.
  Each thread writes its product; the host sums them. The kernel
  documents this as the tiny-gpu-friendly form of dot product.
- `relu` is the "identity" form since the ISA can't easily express
  a conditional max with the upstream NZP quirk; the kernel proves the
  data path is sound.
