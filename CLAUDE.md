# CLAUDE.md — tiny-gpu-extended

This is project memory. Read on every turn.

You are implementing a research project that replicates and extends
[adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu), a minimal
SystemVerilog GPU.

## Repository layout

```
tiny-gpu-extended/
├── CLAUDE.md                  # this file (project memory)
├── ROADMAP.md                 # phase tracker
├── baseline/                  # untouched clone of upstream tiny-gpu (diff target only)
├── src/                       # reimplementation + extensions (the design)
├── test/                      # cocotb 2.x tests
├── kernels/                   # .asm kernel sources (documentation form)
├── bench/                     # bench harness + per-phase JSON results
│   ├── run.py                 # bench driver
│   ├── results/               # per-kernel metrics from the latest run
│   └── phase-N-*.json         # frozen per-phase result roll-ups
├── docs/
│   ├── architecture.md
│   ├── isa.md
│   ├── final-report.md        # written in Phase 10
│   ├── phase-reports/         # one per phase (phase-1-report.md etc.)
│   └── decisions/             # ADRs: one per design decision
└── Makefile
```

## Toolchain

- Icarus Verilog (iverilog) — `12.0`
- cocotb `2.0.1` (pinned; 2.1.x aborts with "No GPI_USERS specified")
- sv2v — SystemVerilog → Verilog conversion
- Python 3 for the bench harness

The Makefile runs `sv2v` on `src/*.sv`, prepends a `\`timescale`, then drives
iverilog and cocotb. See [docs/decisions/ADR-002-toolchain.md](docs/decisions/ADR-002-toolchain.md).

## Working rules

1. `baseline/` is read-only. It exists for diffing against `src/`.
2. Every new module gets a cocotb test (in `test/`) and a `kernels/`
   companion when relevant.
3. Every non-trivial design choice gets an ADR in `docs/decisions/`.
4. Conventional commit prefixes: `feat:`, `fix:`, `test:`, `docs:`,
   `refactor:`, `bench:`.
5. Never break a previously-passing test.
6. ISA changes require updating both `docs/isa.md` and an ADR in the same
   commit.

## Phase status

See `ROADMAP.md` for live status. The harness invocations are:

```bash
make test_matadd                                   # single test
make regression                                    # all kernels, default params
NANOGPT_DIR=... make test_all                      # every test module, harness_selftest, xf pytest, export --check
make -j4 matrix                                    # kernel, L1-D and xf tests per config
python3 bench/run.py --phase 3 \
    --label phase-3-icache --params "ICACHE_LINES=16"
python3 bench/run.py --phase 5 \
    --label phase-5-coalesce --params "ICACHE_LINES=16 COALESCE=1"
python3 bench/run.py --phase 11 --label phase-11-all   # latency / multi-warp / SIMT / MeDiC sweep
python3 bench/run.py --check-citations                 # docs/final-report.md cites existing bench rows
```

## Parameter switches

Everything Phase 3+ is gated by a parameter so the Phase 1 baseline is the
default. Pass them via `iverilog -Pgpu.NAME=VALUE`:

| Param           | Phase | Default | Effect                                          |
|-----------------|-------|---------|-------------------------------------------------|
| `ICACHE_LINES`  | 3     | `0`     | Per-core direct-mapped I-cache when > 0         |
| `PIPELINED`     | 4     | `0`     | Combinational decode + skip DECODE state        |
| `COALESCE`      | 5     | `0`     | Coalesce same-address reads in data memory      |
| `DCACHE_LINES`  | 9     | `0`     | L1-D cache (per-gpu, shared lines, write-through) |
| `MEDIC`         | 9     | `0`     | MeDiC bypass + priority on top of L1-D          |
| `NUM_CORES`     | —     | `2`     | Number of compute cores                         |
| `WORD_BITS`     | xf    | `8`     | Register/ALU/SMEM width; must equal `DATA_MEM_DATA_BITS` (ADR-010) |
| `THREADS_PER_BLOCK` | — | `4`     | Threads per block                               |
| `WARPS_PER_CORE` | SG20 | `1`    | Warps per block, time-sharing the core's lanes (ADR-014) |
| `WARP_POLICY`   | SG20  | `0`     | 0 LRR, 1 GTO, 2 GTLRR when `WARPS_PER_CORE > 1` |
| `SIMT_STACK`    | SG21  | `0`     | Per-warp SIMT stack; `SYNC` marks reconvergence (ADR-015) |

Shared memory is 64 words per core (`core.SMEM_SIZE`); it is not a `gpu` parameter.
| `WARPS_PER_CORE` | SG20 | `1`    | Warps per block, sharing `THREADS_PER_BLOCK / WARPS_PER_CORE` lanes (ADR-012, ADR-014) |
| `WARP_POLICY`   | SG20  | `0`     | Warp scheduler: 0 LRR, 1 GTO, 2 GTLRR (`make test_multiwarp`) |
| `SIMT_STACK`    | SG21  | `0`     | Per-warp SIMT stack (8 entries): `SYNC` + divergent `BR` split and reconverge (ADR-011, ADR-015; `make test_divergence`) |

The warp scheduler (Phase 8) is built as a separate compilation target:

```
make test_warp_scheduler_lrr     # POLICY=0
make test_warp_scheduler_gto     # POLICY=1
make test_warp_scheduler_gtlrr   # POLICY=2
```
