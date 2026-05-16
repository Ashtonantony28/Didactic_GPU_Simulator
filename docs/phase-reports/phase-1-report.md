# Phase 1 — Clean reimplementation

## Status
Done. Both upstream tests (`matadd`, `matmul`) pass against `src/`.

## What landed
- All 12 upstream modules reimplemented from scratch in `src/`:
  `alu`, `controller`, `core`, `dcr`, `decoder`, `dispatch`, `fetcher`,
  `gpu`, `lsu`, `pc`, `registers`, `scheduler`.
- Trailing commas, `break` statements, and other upstream
  iverilog-non-portable patterns removed (sv2v still required for the
  unpacked-array module ports — see [ADR-002](../decisions/ADR-002-toolchain.md)).
- cocotb 2.x test harness in `test/helpers/`. `test/test_matadd.py` and
  `test/test_matmul.py` mirror upstream's kernels bit-for-bit.

## Deviations from upstream
All intentional deviations are logged in
[ADR-001](../decisions/ADR-001-deviations.md). Highlights:
- `scheduler.sv` advances PC from `next_pc[0]` rather than
  `next_pc[THREADS_PER_BLOCK-1]` — fixes an upstream bug on short last
  blocks.
- ALU NZP formula kept bit-for-bit upstream-compatible (unsigned wrap
  quirk that matmul depends on).
- Dispatch counters kept as blocking assignments (necessary for
  same-cycle multi-core dispatch).

## Verification
```
$ make test_matadd
test.test_matadd.test_matadd   PASS   2330 ns

$ make test_matmul
test.test_matmul.test_matmul   PASS   5580 ns
```
