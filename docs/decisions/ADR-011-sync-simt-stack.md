# ADR-011 — SYNC encoding and SIMT-stack semantics

## Status
Accepted (PLAN SG17, contract 10). Encoding and reference semantics only; the RTL stack is SG21.

## Context
Branches resolve to lane 0's target (AUDIT-4 M1), so kernels must keep branches block-uniform.
ADR-006 designed a SIMT stack with a reconvergence opcode, but SG2 used the last free opcodes
(SRA, MAX). A reconvergence point still has to come from the program.

## Decision
`BR` with `nzp = 000` is never taken today. It becomes `SYNC imm`: set the warp's pending
reconvergence PC to `imm`, consumed by the next divergent branch of that warp. The assembler spells
it `SYNC target` and rejects a bare `BR` (AUDIT-4 L3). Kernels without `SYNC` keep their semantics.

With the stack enabled, each warp keeps entries `{pc, rpc, mask}`; the top runs under `mask`.
- Uniform branch: move `top.pc`.
- Divergent branch: needs a pending `R` (else error). Set `top.pc = R`, push
  `{pc+1, R, not-taken}`, push `{imm, R, taken}`; the taken side runs first.
- After every instruction, pop while a non-bottom `top.pc == top.rpc`.
- Inactive lanes read and write nothing (registers, NZP, memory, SMEM).
- `RET` removes the active lanes from every entry and pops empty entries; the warp ends when no
  lanes remain.
- More than `stack_depth` (default 8) pushed entries is an error (`$fatal` in RTL).

Without the stack (today's RTL), `SYNC` is a no-op and the lowest enabled lane decides.
`xf/isasim.py` implements both; it is the oracle for SG21.

## Alternatives rejected
- A 17th opcode: needs a 5-bit opcode field and re-encodes every kernel.
- Assembler-computed post-dominators: needs a CFG pass over arbitrary branch targets; an explicit
  `SYNC` keeps the hardware and the assembler simple.
- Stealing `NOP` (`0000`) operand bits: NOP words appear as zeroed program memory past `RET`.
