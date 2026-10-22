# ROADMAP

Live status tracker. Sub-goal (SG) detail and open issues are in `PLAN.md`.

| Phase | Description | Status | Report |
|---|---|---|---|
| 0 | Bootstrap (clone baseline, verify toolchain) | DONE | [phase-0](docs/phase-reports/phase-0-report.md) |
| 1 | Clean reimplementation in `src/` | DONE | [phase-1](docs/phase-reports/phase-1-report.md) |
| 2 | Bench harness + extra kernels | DONE | [phase-2](docs/phase-reports/phase-2-report.md) |
| 3 | Instruction cache (`ICACHE_LINES`) | DONE | [phase-3](docs/phase-reports/phase-3-report.md) |
| 4 | Combinational decode, DECODE skipped (`PIPELINED`) | DONE | [phase-4](docs/phase-reports/phase-4-report.md) |
| 5 | Memory coalescing (`COALESCE`) | DONE | [phase-5](docs/phase-reports/phase-5-report.md) |
| 6 | Branch divergence, kernel level | DONE | [phase-6](docs/phase-reports/phase-6-report.md) |
| 7 | Shared memory + LDS/STS/BAR | DONE | [phase-7](docs/phase-reports/phase-7-report.md) |
| 8 | Warp scheduler module (LRR/GTO/GTLRR) | DONE | [phase-8](docs/phase-reports/phase-8-report.md) |
| 9 | L1-D + MeDiC (`DCACHE_LINES`, `MEDIC`) | DONE | [phase-9](docs/phase-reports/phase-9-report.md) |
| 10 | Final report | DONE | [final](docs/final-report.md) |
| xf | Transformer kernels at `WORD_BITS=32` and audit fixes (SG0–SG16) | DONE | [xf-bench](docs/xf-bench.md) |
| R3 | SG17 ISA model and contracts; SG18 parallel channels and exact counters; SG19 MeDiC that caches; SG20 multi-warp core, warp scheduler and real `BAR`; SG21 SIMT stack; SG22 kernel and binder robustness | DONE | ADR-011 to ADR-015 |
| 11 | Benchmarks under latency and multi-warp; docs refresh (SG23) | DONE | [final](docs/final-report.md), `bench/phase-11-all.json` |

Phase reports 0–9 and `bench/phase-{2..10}-*.json` are history: their cycle counts predate SG18's
controller fix. Current numbers are in `bench/phase-11-all.json`.

## Verified configurations

`make -j4 matrix` runs the kernel regression, `test_l1d_indepth`, `test_isa_diff`, `test_controller`,
`test_multiwarp` (or `test_divergence` for `simt`) and the xf tests at the 8- and 32-bit widths under:

| Config | Parameters |
|---|---|
| default | — |
| icache | `ICACHE_LINES=16` |
| pipelined | `PIPELINED=1` |
| coalesce | `COALESCE=1` |
| dcache | `DCACHE_LINES=16` |
| medic | `DCACHE_LINES=16 MEDIC=1` (plus `test_medic`) |
| all | `ICACHE_LINES=16 PIPELINED=1 COALESCE=1 DCACHE_LINES=16 MEDIC=1` |
| cores1, cores4 | `NUM_CORES=1`, `NUM_CORES=4` |
| latency | `MEM_LATENCY=3` (test environment) |
| warps2 | `WARPS_PER_CORE=2` |
| warps4gto | `WARPS_PER_CORE=4 WARP_POLICY=1` |
| simt | `SIMT_STACK=1` |

`python3 bench/run.py --phase 11 --label phase-11-all` sweeps memory latency 0/3/10 over these
features, `WARPS_PER_CORE` 1/2/4 × `WARP_POLICY`, the SIMT stack and the MeDiC mixed workload.

## Tests

```
test/test_{matadd,matmul,vector_dot,relu,conv1d,transpose,divergent_relu,tiled_matmul}.py   kernel regression
test/test_edge_cases.py  test_stress.py  test_param_sweep.py  test_launches.py
test/test_shared_mem_indepth.py  test_l1d_indepth.py  test_medic.py  test_controller.py
test/test_multiwarp.py  test_warp_scheduler_{lrr,gto,gtlrr}.py  test_divergence.py  simt_fatal.py
test/test_isa_ext.py  test_isa_diff.py                    ISA extensions, RTL vs xf.isasim
test/test_xf_*.py                                         transformer kernels (32-bit build)
test/test_bench_sweep.py                                  phase-11 bench driver module
xf/tests/                                                 pytest: assembler, model, golden, export, audits, bench citations
```

## Parameters

| Param | Phase | Default | Effect |
|---|---|---|---|
| `ICACHE_LINES` | 3 | 0 | Per-core direct-mapped I-cache when > 0 |
| `PIPELINED` | 4 | 0 | Combinational decode + skip DECODE state |
| `COALESCE` | 5 | 0 | Same-address reads share one data transaction |
| `DCACHE_LINES` | 9 | 0 | L1-D (per GPU, shared lines, write-through) when > 0 |
| `MEDIC` | 9 | 0 | MeDiC bypass + controller priority on top of L1-D |
| `WORD_BITS` | xf | 8 | Register/ALU/SMEM width; must equal `DATA_MEM_DATA_BITS` |
| `NUM_CORES` | — | 2 | Number of compute cores |
| `THREADS_PER_BLOCK` | — | 4 | Threads per block |
| `WARPS_PER_CORE` | SG20 | 1 | Warps per block; must divide `THREADS_PER_BLOCK` |
| `WARP_POLICY` | SG20 | 0 | 0 LRR, 1 GTO, 2 GTLRR (used when `WARPS_PER_CORE > 1`) |
| `SIMT_STACK` | SG21 | 0 | Per-warp SIMT stack; `SYNC` marks reconvergence |

Shared memory is 64 words per core (`core.SMEM_SIZE`), not a `gpu` parameter.
