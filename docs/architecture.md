# Architecture

tiny-gpu-extended keeps upstream tiny-gpu's structure (dispatcher, cores, memory controllers) and adds
the Phase 3+ and Round 3 extensions, each gated by a `gpu` parameter whose default gives the Phase 1
baseline. Measured effects are in [final-report.md](final-report.md).

## Block diagram

```
                         ┌──────────────────────────┐
                         │ dispatch (blocks → cores)│   dcr: thread_count
                         └────────────┬─────────────┘
                 core_start / core_reset / block_id / thread_count
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
   ┌───────────────────────────┐                 ┌───────────────────────────┐
   │ core 0                    │   ...  NUM_CORES │ core N-1                  │
   │  warp_scheduler (W > 1)   │                 │                           │
   │  scheduler FSM, per-warp  │                 │                           │
   │   PC (+ simt_stack)       │                 │                           │
   │  fetcher (+ I-cache)      │                 │                           │
   │  decoder                  │                 │                           │
   │  LANES = TPB / W  ALUs    │                 │                           │
   │  TPB × {regs, pc/NZP, LSU}│                 │                           │
   │  shared_memory (64 words) │                 │                           │
   └─────┬──────────────┬──────┘                 └─────┬──────────────┬──────┘
         │ fetch        │ TPB LSUs                     │              │
         │              ▼                              │              ▼
         │   ┌─────────────────────────────────────────────────────────────┐
         │   │ l1d_cache (DCACHE_LINES > 0; MEDIC bypass)                    │
         │   └──────────────────────────────┬──────────────────────────────┘
         │   ┌──────────────────────────────▼──────────────────────────────┐
         │   │ data controller: DATA_MEM_NUM_CHANNELS channels, COALESCE,    │
         │   │ MEDIC priority                                                │
         │   └──────────────────────────────┬──────────────────────────────┘
         ▼                                  ▼
   program controller (1 channel)     external data memory
         ▼
   external program memory
```

`TPB` = `THREADS_PER_BLOCK`, `W` = `WARPS_PER_CORE`.

## Modules

| Module | File | Role |
|---|---|---|
| `gpu` | `src/gpu.sv` | Top level: dispatcher, cores, L1-D, controllers, telemetry |
| `dispatch` | `src/dispatch.sv` | Hands blocks of `THREADS_PER_BLOCK` threads to free cores |
| `dcr` | `src/dcr.sv` | Device control register: the kernel's `thread_count` |
| `controller` | `src/controller.sv` | Arbitrates consumers onto memory channels; coalescing, priority, request counters |
| `l1d_cache` | `src/l1d_cache.sv` | Direct-mapped, write-through L1-D shared by all LSUs; MeDiC classification and bypass |
| `core` | `src/core.sv` | One block at a time, as `WARPS_PER_CORE` warps |
| `scheduler` | `src/scheduler.sv` | Control FSM and per-warp PCs; issue slot, yield, `BAR` |
| `warp_scheduler` | `src/warp_scheduler.sv` | Picks the issuing warp (LRR / GTO / GTLRR) when `WARPS_PER_CORE > 1` |
| `simt_stack` | `src/simt_stack.sv` | Per-warp reconvergence stack and lane mask when `SIMT_STACK = 1` |
| `fetcher` | `src/fetcher.sv` | Instruction fetch; I-cache when `ICACHE_LINES > 0` |
| `decoder` | `src/decoder.sv` | Combinational decode of a 16-bit instruction |
| `alu` | `src/alu.sv` | `LANES` per core, shared by the warps |
| `lsu` | `src/lsu.sv` | One per thread: global and shared loads and stores |
| `registers` | `src/registers.sv` | One per thread: 16 registers (R13–R15 read-only `%blockIdx`, `%blockDim`, `%threadIdx`) |
| `pc` | `src/pc.sv` | One per thread: NZP and branch outcome |
| `shared_memory` | `src/shared_memory.sv` | Per-core SMEM, 64 words, one read and one write port per lane |

## Control flow

The scheduler steps one instruction at a time through FETCH, DECODE, REQUEST, WAIT, EXECUTE, UPDATE.
`PIPELINED = 1` skips DECODE (the decoder is combinational). REQUEST starts LSU operations, WAIT holds
until they finish, EXECUTE computes ALU results and next PCs, UPDATE writes registers and NZP and
advances the PC, or ends the warp on `RET`.

- **`WARPS_PER_CORE = 1`**: UPDATE goes straight back to FETCH; the block is one warp in lockstep.
- **`WARPS_PER_CORE > 1`** (ADR-014): IDLE is the issue slot where `warp_scheduler` picks a ready warp;
  UPDATE returns to IDLE. A warp whose global load or store is still in flight in WAIT yields the core
  and re-issues the instruction once its LSUs are done. A warp at `BAR` waits until every live warp of
  the block is at a `BAR`.
- **Branches**: without the SIMT stack a warp follows its lowest thread's next PC, so branches must be
  uniform within a warp (ADR-001, AUDIT-4 M1). With `SIMT_STACK = 1` the stack supplies the next PC and
  the lane mask (ADR-011, ADR-015).

## Memory model

| Memory | Address bits | Data bits | Channels | Access |
|---|---|---|---|---|
| Program | `PROGRAM_MEM_ADDR_BITS` = 8 | 16 | `PROGRAM_MEM_NUM_CHANNELS` = 1 | read |
| Data, default build | `DATA_MEM_ADDR_BITS` = 8 | `DATA_MEM_DATA_BITS` = `WORD_BITS` = 8 | `DATA_MEM_NUM_CHANNELS` = 4 | read/write |
| Data, transformer build | 12 | 32 | 4 | read/write |
| Shared (per core) | register, masked to 6 bits | `WORD_BITS` | one port per lane | read/write |

`DATA_MEM_DATA_BITS` must equal `WORD_BITS` (ADR-010). The data controller serves
`NUM_CORES × THREADS_PER_BLOCK` LSUs (through the L1-D when present) on its channels; each channel takes
a different consumer in the same cycle (ADR-013). With `COALESCE = 1` reads of one address share one
transaction. With `MEDIC = 1` high-utility lanes are claimed first. The program controller serves one
fetcher per core.

## ISA

16 opcodes, 16-bit fixed width, `SYNC` encoded as `BR` with `nzp = 000`. See [isa.md](isa.md).

## Telemetry

| Output | Meaning |
|---|---|
| `data_mem_requests` | downstream data transactions |
| `data_mem_requests_coalesced` | consumer reads served by another consumer's transaction |
| `icache_hits_total`, `icache_misses_total` | summed over all cores |
| `dcache_hits_total`, `dcache_misses_total` | L1-D lookups |
| `warp_issues_total` | instructions issued, all warps and cores (equals `xf.isasim` steps) |
| `bar_stall_cycles_total` | cycles warps wait at `BAR` |

Exact definitions are in ADR-012; `test/helpers/runner.py::check_counters` checks the request and
I-cache counters on every kernel run. Counters reset with the GPU, not the per-block core reset
([ADR-003](decisions/ADR-003-telemetry-reset.md)).
