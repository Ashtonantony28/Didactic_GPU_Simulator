# Phase 7 — Shared memory + LDS/STS/BAR (fully wired)

## Status
Done. LDS / STS / BAR opcodes wired through decoder, LSU, register
file, and the new `src/shared_memory.sv` module. Demonstrated by
`test/test_tiled_matmul.py` (2×2 matmul using SMEM staging).

## What landed
- `src/shared_memory.sv`: per-block banked SRAM. Combinational read,
  synchronous write. Per-lane ports. SIZE entries × DATA_BITS bits.
- `src/decoder.sv`: opcodes `LDS` (1010), `STS` (1011), `BAR` (1100)
  added.
- `src/lsu.sv` extended: when `decoded_smem_read_enable` or
  `decoded_smem_write_enable` is set, the LSU uses the SMEM port (with
  the same FSM as global memory). Combinational SMEM reads let the
  LSU latch data in WAITING state with no extra cycle.
- `src/registers.sv`: a 4th reg_input_mux value (SHARED = 2'b11) is
  added, routing SMEM reads back into the register file via `lsu_out`.
- `src/scheduler.sv`: new ports for SMEM decoded signals (today
  consumed only by the LSU; BAR is a no-op in single-warp lockstep —
  designed-but-not-yet-functional barrier semantics are documented in
  [ADR-007](../decisions/ADR-007-shared-memory.md)).

## Test results

```
[PASS] tiled_matmul                       cycles=418  (PIPELINED=0)
[PASS] test_smem_roundtrip                STS → BAR → LDS, single-lane
[PASS] test_smem_cross_thread_read        threads read each other's writes
[PASS] test_smem_persistence              multiple LDS to same addr
```

The tiled-matmul kernel cuts global memory requests vs. an untiled
version: 12 mem_req (4 LDR + 4 STR + 4 STR for staging) vs. the naive
matmul's 20 mem_req.

## ISA additions
See [docs/isa.md](../isa.md):
```
1010 LDS rd, rs   ; rd ← SMEM[rs]
1011 STS rs, rt   ; SMEM[rs] ← rt
1100 BAR          ; barrier (no-op in single-warp; reserved for Phase 8 multi-warp)
```

## Reproducibility
```
make test_tiled_matmul
make test_shared_mem_indepth
```
