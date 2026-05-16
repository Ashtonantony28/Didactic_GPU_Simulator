# ADR-001 — Intentional deviations from upstream tiny-gpu

## Status
Accepted

## Context
We reimplemented every module in `src/` from scratch (Phase 1). A few
places had to deviate from upstream to compile under iverilog 12 + cocotb
2.x or to fix real bugs we found. This ADR is the audit log.

## Deviations

### 1. scheduler picks `next_pc[0]` instead of `next_pc[THREADS_PER_BLOCK-1]`

Upstream `scheduler.sv` advances the PC to `next_pc[THREADS_PER_BLOCK-1]`.
Under the "no divergence" assumption, every lane's `next_pc` is the same,
so the choice doesn't matter — **except** that PCs of disabled lanes
(`enable=0` when the block has fewer threads than `THREADS_PER_BLOCK`)
never update. They stay at 0. So if the last block is short (e.g. 6
threads with `THREADS_PER_BLOCK=4` → second block has 2 threads, lanes 2
and 3 disabled), `next_pc[3]` is locked to 0 and the second block loops
on PC=0 forever.

We pick lane 0, which is guaranteed enabled (block always has ≥ 1 thread).

This affected our `conv1d` (6 threads) and `transpose` (8 threads with the
8-thread tail) kernels.

### 2. ALU NZP formula kept upstream's quirk

`CMP` sets `NZP = {(rs-rt > 0), (rs-rt == 0), (rs-rt < 0)}` under unsigned
arithmetic — `N=1` whenever `rs ≠ rt`, `P=0` always. We tried `(rs > rt) /
(rs == rt) / (rs < rt)` first; it broke `matmul` because the upstream
kernel relies on the wrap-around behavior of the `<` flag. Kept upstream's
formula to maintain binary compatibility with upstream's kernels.

### 3. Decoder: replaced trailing comma in `output reg [3:0] decoded_rs_address,` style port lists

iverilog with `-g2012` doesn't accept the trailing comma upstream's
`dcr.sv`, `fetcher.sv`, `gpu.sv`, `core.sv` have. sv2v fixes this when
compiling baseline. We just removed the trailing commas in our
reimplementation.

### 4. controller: removed `break` statements

iverilog 12 errors on `break` in `always` blocks. We replaced upstream's
`break;` with a `found` flag.

### 5. dispatch: blocking assignments preserved for `blocks_dispatched`/`blocks_done`

Upstream uses `=` (blocking) for these counters inside a `posedge clk`
always block, with `<=` for the rest. This is a common style violation
but is semantically necessary: the `for (i = 0; ...)` loop dispatches one
block per iteration, and a non-blocking write would make every core see
the same `blocks_dispatched` value and dispatch the same block.

We initially "fixed" this to `<=` and broke matadd's 2-block dispatch.
Reverted. Logged here so we don't make the same mistake again.

### 6. Telemetry counters on a separate reset domain

See [ADR-003](ADR-003-telemetry-reset.md).

## Consequences

`src/` is functionally a clean rewrite of `baseline/` with these
deviations. Both upstream test programs (`matadd`, `matmul`) pass
bit-identically. Several upstream-only edge cases that we don't trigger
(threads-not-aligned-to-THREADS_PER_BLOCK, telemetry across multiple
dispatch waves) actually behave better in our version.
