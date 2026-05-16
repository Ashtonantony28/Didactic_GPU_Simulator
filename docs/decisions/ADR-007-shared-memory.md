# ADR-007 — Shared memory + barriers (Phase 7)

## Status
Designed; RTL skeleton in `src/shared_memory.sv`; not wired into the
core in this revision.

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

## Why not wired

Wiring the shared-memory path means:
- Splitting the LSU into "global" and "shared" sub-LSUs (or adding a
  selector in the existing one).
- Adding a second data port to the register file's `rd` writeback.
- Updating every cycle the scheduler considers `lsu_state` to also
  consider the shared-memory LSU.

This is ~300 lines of RTL plus its own tests. We chose to ship the
design (this ADR, `shared_memory.sv`, ISA reservation) rather than a
half-finished implementation.

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
