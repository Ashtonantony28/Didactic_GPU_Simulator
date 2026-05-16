# ADR-007 — Shared memory + barriers (Phase 7)

## Status
Implemented. RTL in `src/shared_memory.sv` is wired through the LSU,
register file, and decoder. LDS / STS / BAR opcodes are decoded and
executed. Demonstrated by `test_tiled_matmul` and three `smem_indepth`
unit tests.

## Context
Phase 7 adds per-block SRAM addressable by new opcodes `LDS` / `STS` and
synchronizes with `BAR`. The expected workload is tiled matrix multiply.

## Design

### Memory layout
```
shared_memory #(.SIZE = 64) per core
            ↑
    ┌───────┴───────┐
    │  THREADS_PER_BLOCK lanes
    │   LDS / STS port (1 read + 1 write per cycle, banked by lane)
```

64 entries × 8 bits per core, addressable by a 6-bit immediate in the
shared-mem opcodes.

### ISA encoding

```
| op  | rd  | rs  | rt  |
| 1010|     |     |     |   LDS rd, rs       rd ← SMEM[rs]
| 1011|     |     |     |   STS rs, rt       SMEM[rs] ← rt
| 1100| 0000| 0000| 0000|   BAR              implicit; serves as scheduler hint
```

### BAR semantics
In a single-warp-per-core architecture, all threads are already at the
same instruction (lockstep), so BAR is a free no-op. It exists as
forward-compatibility for the multi-warp scheduler in Phase 8 — there it
becomes a real wait.

### Decoder changes
Three new opcode constants. `LDS` / `STS` set new control signals
`decoded_smem_read_enable` / `decoded_smem_write_enable`, routed to the
LSU which steers between the data memory controller and the shared
memory.

### Module skeleton (provided)
`src/shared_memory.sv` is a single-cycle banked SRAM with one R/W port
per lane.

## How it's wired

- **LSU** (`src/lsu.sv`): the same FSM (IDLE→REQUESTING→WAITING→DONE)
  now handles four mutually-exclusive paths — `decoded_mem_read_enable`
  (global read), `decoded_mem_write_enable` (global write),
  `decoded_smem_read_enable` (shared read), `decoded_smem_write_enable`
  (shared write). The decoder guarantees at most one of these is set.
  Reads land in `lsu_out` regardless of which path; writeback through
  the register file looks the same.
- **Register file** (`src/registers.sv`): a new `SHARED = 2'b11`
  reg_input_mux value. LDS routes through `lsu_out` like LDR does.
- **Decoder** (`src/decoder.sv`): three new opcodes plus their
  control signals (also made fully combinational — see ADR-005).
- **Core** (`src/core.sv`): instantiates a per-block `shared_memory`
  with per-lane R/W ports.
- **`src/shared_memory.sv`**: combinational read, synchronous write
  (combinational read lets the LSU latch in WAITING state without an
  extra cycle).

BAR is a no-op in single-warp lockstep — when WARPS_PER_CORE > 1
(future) it becomes a real barrier wait. The opcode is decoded and
visible at the scheduler, ready to consume.

## Tiled matmul kernel (for future)

```asm
; Each thread loads A[r][k] and B[k][c] into SMEM, BAR, then accumulates
; using LDS reads. Cuts global-memory pressure by a factor of TILE_SIZE.
```

See `kernels/tiled_matmul.asm` for the documented kernel.

## Consequences
- `docs/isa.md` documents the reserved opcodes.
- The `shared_memory.sv` module is a known-good starting point.
- Phase 7 bench numbers are identical to Phase 5 (no kernel change).
