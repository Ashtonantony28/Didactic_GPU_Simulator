# ADR-007 — Shared memory and barriers (Phase 7)

## Status
Implemented. `src/shared_memory.sv` is wired through the decoder, LSU and register file; `LDS`, `STS`
and `BAR` execute. `BAR` is a real barrier at `WARPS_PER_CORE > 1` since SG20 (ADR-012, ADR-014).

## Context
Phase 7 adds per-block SRAM for tiled kernels, reached by `LDS` / `STS` and synchronised by `BAR`.

## Design

### Memory
One `shared_memory` per core, used by the block running on it (all its warps). `core.SMEM_SIZE` = 64
words; it is a `core` parameter fixed at 64, not a `gpu` parameter. Words are `WORD_BITS` wide (8, or 32
in the transformer build). Contents are zeroed by the per-block reset.

### Encoding

```
| op   | rd | rs | rt |
| 1010 | rd | rs |    |   LDS rd, rs     rd ← SMEM[rs]
| 1011 |    | rs | rt |   STS rs, rt     SMEM[rs] ← rt
| 1100 |    |    |    |   BAR
```

The address is a register: the LSU passes `rs[7:0]` and the memory masks it to `$clog2(SMEM_SIZE)`
(6) bits.

### Datapath
- **Decoder** sets `decoded_smem_read_enable` / `decoded_smem_write_enable` / `decoded_bar`.
- **LSU** handles four exclusive paths in one FSM (global read, global write, shared read, shared write).
  A shared access takes two cycles and never yields the core (ADR-014).
- **Register file**: `LDS` writes back through `lsu_out`, as `LDR` does.
- **`shared_memory.sv`**: one read and one write port per lane, combinational read, synchronous write.

### BAR
At `WARPS_PER_CORE = 1` a block is one warp in lockstep, so `BAR` is a no-op. At `WARPS_PER_CORE > 1`
a warp that executes `BAR` waits until every unfinished warp of its block has reached a `BAR`
(ADR-012); `bar_stall_cycles_total` counts the wait.

## Kernels
- `kernels/tiled_matmul.asm`: 2x2 matmul through SMEM, the program of `test/test_tiled_matmul.py`
  (`xf/tests/test_asm.py` checks they match).
- `kernels/matmul_tiled.asm`: the transformer track's int8/int32 tiled matmul.
- `test/test_shared_mem_indepth.py` (three SMEM cases) and `test/test_multiwarp.py` (barriers that fail
  when `BAR` is a no-op).

## Consequences
- Programs that relied on lockstep for SMEM ordering need `BAR` once a block has several warps
  (ADR-012).
- On the 2x2 regression the tiled kernel issues 12 data requests against 20 for the naive matmul
  (`bench/phase-11-all.json`, `kernels/default/{tiled_matmul,matmul}/L0`).
