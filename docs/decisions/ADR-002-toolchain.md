# ADR-002 — Toolchain choices

## Status
Accepted

## Context
The kickoff prompt requires Icarus Verilog + cocotb + sv2v + Python 3
(match upstream). The environment has iverilog 12 and Python 3.11
available; cocotb and sv2v had to be installed.

## Decisions

### Icarus Verilog 12 with `-g2012`

iverilog 12 supports most of SystemVerilog-2012 directly, but it does
**not** accept module ports with unpacked array types (`reg [7:0] foo
[NUM-1:0]` in a port list) nor `break` inside `always` blocks. We use
`sv2v` to pre-process `src/*.sv` into a single Verilog-2005-ish
`build/gpu.v`, then run iverilog on that.

### cocotb 2.0.1

Cocotb 2.x changed two things that matter for porting upstream's tests:

1. `MODULE=...` envvar is gone. We use `COCOTB_TEST_MODULES=...`.
2. `Clock(dut.clk, 25, units="us")` is rejected by the iverilog timescale
   precision check. We use `Clock(dut.clk, 10, unit="ns")` with a
   prepended `\`timescale 1ns/1ns` on the generated Verilog. The Makefile
   does the prepending.
3. `LogicArray` no longer supports arithmetic operators — `format_cycle`
   in `baseline/test/helpers/format.py` is incompatible. We omitted the
   format helper from our `test/` and use a smaller `Memory.display`
   instead.
4. `vvp` exits 0 even when cocotb fails a test or never starts (2.1.x
   aborts with "No GPI_USERS specified"), so the version is pinned to
   2.0.1 and every make target checks `build/results.xml` with
   `test/helpers/check_results.py`: any failure, or zero tests run, fails
   the target.

### sv2v 0.0.13 (downloaded binary)

We download the prebuilt Linux binary into `/usr/local/bin/sv2v`. The
binary is small and self-contained.

## Consequences

- `make compile` runs `sv2v -w build/gpu.v src/*.sv` then `iverilog
  -g2012`. The Makefile prepends a timescale.
- `make test_<name>` sets `COCOTB_TEST_MODULES` and runs `vvp` with the
  cocotb VPI library.
- Parameter overrides go via `iverilog -Pgpu.<NAME>=<VALUE>`.
