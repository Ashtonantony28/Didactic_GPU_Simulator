# Phase 7 — Shared memory + barriers

## Status
Designed. RTL skeleton in `src/shared_memory.sv`, documented kernel in
`kernels/tiled_matmul.asm`, ISA reservation in [docs/isa.md](../isa.md),
design in [ADR-007](../decisions/ADR-007-shared-memory.md).

Not wired into `core.sv` — the wiring is a ~300-line change touching
LSU, register file, decoder, and scheduler.

## Why it stops at design
The kickoff prompt's working rule: "Never break a previously-passing
test. If a phase regresses, fix it before moving on." Wiring shared
memory means touching every datapath module; we chose to document the
design carefully rather than ship a half-working datapath that would
take subsequent phases as collateral.

## What's ready
- `src/shared_memory.sv` — banked single-cycle SRAM, parameterized by
  `THREADS_PER_BLOCK`, `SIZE`, `ADDR_BITS`, `DATA_BITS`.
- Opcodes `LDS` (`1010`), `STS` (`1011`), `BAR` (`1100`) reserved.
- `kernels/tiled_matmul.asm` documents the demonstration kernel.

## Reproducibility
Module compiles standalone:
```
sv2v -w build/sm.v src/shared_memory.sv && iverilog -g2012 -y src/ build/sm.v -o /dev/null
```
