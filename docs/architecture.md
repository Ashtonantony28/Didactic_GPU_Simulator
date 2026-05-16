# Architecture

This document describes the tiny-gpu-extended design from first principles.
It mirrors the upstream architecture and adds the Phase 3+ extensions, all
gated by parameters so the Phase 1 baseline is the default.

## Block diagram

```
                                  ┌──────────────────┐
                                  │     dispatch     │
                                  │  block scheduler │
                                  └────────┬─────────┘
                                           │ core_start / core_reset
                                           │ core_block_id / core_thread_count
                              ┌────────────┴────────────┐
                              ▼                         ▼
                       ┌────────────┐            ┌────────────┐
                       │   core 0   │            │   core 1   │
                       │ ┌────────┐ │            │ ┌────────┐ │
                       │ │ sched  │ │            │ │ sched  │ │
                       │ └───┬────┘ │            │ └───┬────┘ │
                       │ ┌───▼────┐ │            │ ┌───▼────┐ │
                       │ │fetcher │ │            │ │fetcher │ │
                       │ │+icache │ │            │ │+icache │ │
                       │ └───┬────┘ │            │ └───┬────┘ │
                       │ ┌───▼────┐ │            │ ┌───▼────┐ │
                       │ │decoder │ │            │ │decoder │ │
                       │ └───┬────┘ │            │ └───┬────┘ │
                       │     │      │            │     │      │
                       │  THREADS_PER_BLOCK lanes │  THREADS_PER_BLOCK lanes
                       │ ┌──┴──┬──┬─┐ │            │ ┌──┴──┬──┬─┐ │
                       │ │ ALU │RF│PC│            │ │ ALU │RF│PC│
                       │ │ LSU │  │  │            │ │ LSU │  │  │
                       │ └─────┴──┴──┘            │ └─────┴──┴──┘
                       └──────┬───────┘            └──────┬───────┘
                              │                           │
                              ▼                           ▼
                          ┌────────────────────────────────┐
                          │    data memory controller       │
                          │ (NUM_CHANNELS, optional COALESCE)│
                          └────────────────┬───────────────┘
                                           ▼
                                    External Data Memory

                          ┌────────────────────────────────┐
                          │  program memory controller      │
                          └────────────────┬───────────────┘
                                           ▼
                                  External Program Memory
```

## Modules

| Module        | File                  | Role                                                                 |
|---------------|-----------------------|----------------------------------------------------------------------|
| `gpu`         | `src/gpu.sv`          | Top-level: wires dispatcher, controllers, cores onto external memory |
| `dispatch`    | `src/dispatch.sv`     | Issues blocks of `THREADS_PER_BLOCK` threads to free cores           |
| `dcr`         | `src/dcr.sv`          | Device Control Register — stores the kernel-wide `thread_count`      |
| `controller`  | `src/controller.sv`   | Bridges NUM_CONSUMERS LSUs/fetchers onto NUM_CHANNELS memory ports   |
| `core`        | `src/core.sv`         | One block at a time; instantiates lanes + scheduler                  |
| `scheduler`   | `src/scheduler.sv`    | 6-state FSM driving the block's control flow                         |
| `fetcher`     | `src/fetcher.sv`      | Fetches instructions; optional I-cache                               |
| `decoder`     | `src/decoder.sv`      | Maps a 16-bit instruction to control signals                         |
| `alu`         | `src/alu.sv`          | Per-thread ALU                                                       |
| `lsu`         | `src/lsu.sv`          | Per-thread load/store unit (async with REQUESTING/WAITING/DONE)      |
| `registers`   | `src/registers.sv`    | Per-thread 16-entry register file (R0..R12 free, R13/R14/R15 RO)     |
| `pc`          | `src/pc.sv`           | Per-thread next-PC and NZP register                                  |

## Control flow

A block executes in lockstep. The scheduler cycles through:

1. **IDLE** — wait for `start`.
2. **FETCH** — drive `current_pc` to the fetcher; wait for `fetcher_state==FETCHED`.
3. **DECODE** — decoder produces control signals.
4. **REQUEST** — register file produces `rs`/`rt`; LSUs leave IDLE if mem op.
5. **WAIT** — block until all LSUs are out of REQUESTING/WAITING.
6. **EXECUTE** — ALU computes; PC computes next.
7. **UPDATE** — register file writeback, NZP latch, PC advance, or `RET → DONE`.

Loop back to FETCH unless `RET`.

`scheduler.sv` picks the next PC from lane 0 (always enabled). Upstream picks
lane `THREADS_PER_BLOCK-1`, which can hang on the last block if it has fewer
threads — see [ADR-001](decisions/ADR-001-deviations.md).

## Memory model

- **Program memory** — 8-bit address, 16-bit data, read-only, 1 channel.
- **Data memory**    — 8-bit address, 8-bit data, R/W, 4 channels by default.

The data memory controller multiplexes `NUM_LSUS = NUM_CORES * THREADS_PER_BLOCK`
consumers onto `DATA_MEM_NUM_CHANNELS` external channels. With `COALESCE=1`
sequential reads to the same address are merged into a single transaction whose
response broadcasts to every coalesced consumer.

## ISA summary

11 instructions, 16-bit fixed width. See [docs/isa.md](isa.md) for full
details. Opcode in bits [15:12]; the remaining fields depend on the
instruction (`rd[11:8] rs[7:4] rt[3:0]`, or `imm[7:0]`, or `nzp[11:9] imm[7:0]`
for `BRnzp`).

## Telemetry

The GPU exposes counters consumed by `bench/run.py`:

- `data_mem_requests` — total transactions issued on data memory.
- `data_mem_requests_coalesced` — number of consumer requests merged into
  another's transaction by the coalescer.
- `icache_hits_total`, `icache_misses_total` — summed across cores.

Telemetry counters live on the **global** clock domain and reset only on
the top-level `reset`, not on per-block `core_reset`. See [ADR-003](decisions/ADR-003-telemetry-reset.md).
