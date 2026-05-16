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
- cocotb 2.x
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
python3 bench/run.py --phase 3 \
    --label phase-3-icache --params "ICACHE_LINES=16"
python3 bench/run.py --phase 5 \
    --label phase-5-coalesce --params "ICACHE_LINES=16 COALESCE=1"
```

## Parameter switches

Everything Phase 3+ is gated by a parameter so the Phase 1 baseline is the
default. Pass them via `iverilog -Pgpu.NAME=VALUE`:

| Param           | Phase | Default | Effect                                          |
|-----------------|-------|---------|-------------------------------------------------|
| `ICACHE_LINES`  | 3     | `0`     | Per-core direct-mapped I-cache when > 0         |
| `COALESCE`      | 5     | `0`     | Coalesce same-address reads in data memory      |
| `NUM_CORES`     | —     | `2`     | Number of compute cores                         |
| `THREADS_PER_BLOCK` | — | `4`     | Threads per block (lockstep within a core)      |
