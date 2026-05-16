# ROADMAP

Live status tracker.

| Phase | Description                                  | Status | Report                                          |
|-------|----------------------------------------------|--------|-------------------------------------------------|
| 0     | Bootstrap (clone baseline, verify toolchain) | DONE   | [phase-0](docs/phase-reports/phase-0-report.md) |
| 1     | Clean reimplementation in `src/`             | DONE   | [phase-1](docs/phase-reports/phase-1-report.md) |
| 2     | Bench harness + extra kernels                | DONE   | [phase-2](docs/phase-reports/phase-2-report.md) |
| 3     | Instruction cache (parameterised)            | DONE   | [phase-3](docs/phase-reports/phase-3-report.md) |
| 4     | Pipelining (combinational decode + skip DECODE) | DONE | [phase-4](docs/phase-reports/phase-4-report.md) |
| 5     | Memory coalescing                            | DONE   | [phase-5](docs/phase-reports/phase-5-report.md) |
| 6     | Branch divergence — kernel level             | DONE   | [phase-6](docs/phase-reports/phase-6-report.md) |
| 7     | Shared memory + LDS/STS/BAR (fully wired)    | DONE   | [phase-7](docs/phase-reports/phase-7-report.md) |
| 8     | Pluggable warp scheduler module (LRR/GTO/GTLRR) | DONE | [phase-8](docs/phase-reports/phase-8-report.md) |
| 9     | L1-D + MeDiC (cache, characterization, bypass, priority) | DONE | [phase-9](docs/phase-reports/phase-9-report.md) |
| 10    | Final report                                 | DONE   | [final](docs/final-report.md)                   |

## Verified configurations

All 24 cocotb tests pass under every parameter combination tested. The
matrix:

| Configuration | matadd | matmul | matmul cycles |
|---|---|---|---|
| Phase 1 baseline (all params = 0)        | PASS | PASS | 556 |
| + I-cache (ICACHE_LINES=16)              | PASS | PASS | 504 |
| + coalesce (COALESCE=1)                  | PASS | PASS | 464 |
| + pipelined (PIPELINED=1)                | PASS | PASS | 423 |
| + L1-D (DCACHE_LINES=16)                 | PASS | PASS | 433 |
| + MeDiC (MEDIC=1)                        | PASS | PASS | 433 |

## Test inventory (24 tests across 14 files)

```
test/test_matadd.py                test/test_matmul.py
test/test_vector_dot.py            test/test_relu.py
test/test_conv1d.py                test/test_transpose.py
test/test_divergent_relu.py        test/test_tiled_matmul.py
test/test_edge_cases.py            (4 cases: single thread, two blocks one core,
                                    pure compute, unaligned block)
test/test_shared_mem_indepth.py    (3 cases: roundtrip, cross-thread, persistence)
test/test_l1d_indepth.py           (2 cases: hot address, disable equivalence)
test/test_param_sweep.py           (sweep verifier)
test/test_stress.py                (2 cases: long loop, many threads)
test/test_warp_scheduler_lrr.py    test/test_warp_scheduler_gto.py
test/test_warp_scheduler_gtlrr.py
```

## Parameter cheat sheet

| Param            | Phase | Default | Effect                                                |
|------------------|-------|---------|-------------------------------------------------------|
| `ICACHE_LINES`   | 3     | 0       | Per-core direct-mapped I-cache when > 0               |
| `PIPELINED`      | 4     | 0       | Combinational decode + skip DECODE state              |
| `COALESCE`       | 5     | 0       | Coalesce same-address reads in data memory            |
| `DCACHE_LINES`   | 9     | 0       | L1-D cache (per-core consumer-side, shared lines)     |
| `MEDIC`          | 9     | 0       | MeDiC bypass + priority on top of L1-D                |
| `SMEM_SIZE`      | 7     | 64      | Per-block shared memory entries                       |
| `NUM_CORES`      | —     | 2       | Number of compute cores                               |
| `THREADS_PER_BLOCK` | — | 4       | Threads per block (lockstep within a core)            |

The warp scheduler (Phase 8) is a separate compilation target
(`warp_scheduler_tb`) with `POLICY` parameter — 0=LRR, 1=GTO, 2=GTLRR.
