# Phase 4 — Pipelining (early refill via combinational decode + skip DECODE)

## Status
Done. `PIPELINED=1` skips the dead DECODE state of the scheduler FSM by
making decode fully combinational.

## What landed
- `src/decoder.sv` rewritten as combinational (`always @(*)`). Decoded
  outputs are now wires from `instruction`, stable while the fetcher
  holds the instruction register.
- `src/scheduler.sv` adds a `PIPELINED` parameter. When non-zero, the
  FETCH state transitions directly to REQUEST (skipping DECODE).
- `src/fetcher.sv` extended to leave FETCHED on either `core_state ==
  DECODE` (legacy path) or `core_state == REQUEST` (pipelined path), so
  the fetcher correctly recycles regardless of mode.
- `src/gpu.sv` and `src/core.sv` plumb the new parameter through.

## Why combinational decode is safe
- The decoder's outputs depend only on `instruction`.
- `instruction` is registered in the fetcher and is stable from the
  cycle after `FETCHED` until the next `FETCH` triggers a refetch.
- Register file, ALU, LSU, PC all clock-sample decoded signals on edges
  inside that stable window.

## Cycle savings (ICACHE_LINES=16, COALESCE=1)

| Kernel       | PIPELINED=0 | PIPELINED=1 | Δ            |
|--------------|-------------|-------------|--------------|
| matadd       | 231         | 220         | -4.8%        |
| **matmul**   | **464**     | **423**     | **-8.8%**    |
| vector_dot   | 231         | 220         | -4.8%        |
| relu         | 164         | 155         | -5.5%        |
| **conv1d**   | **327**     | **295**     | **-9.8%**    |
| transpose    | 244         | 230         | -5.7%        |
| divergent_relu | 288       | 270         | -6.2%        |
| **tiled_matmul** | **418** | **387**     | **-7.4%**    |

## Verification
- All 24 cocotb tests pass with `PIPELINED=1`.
- `test/test_param_sweep.py` runs the same kernel under every
  combination of `PIPELINED ∈ {0, 1}` with all other params; passes.
- Negative test: with `PIPELINED=0`, cycle counts identical to Phase 7
  numbers (no regression).

## Reproducibility
```
python3 bench/run.py --phase 4 --label phase-4-pipelined \
    --params "ICACHE_LINES=16 COALESCE=1 PIPELINED=1"
```
