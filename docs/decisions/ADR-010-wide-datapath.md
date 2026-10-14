# ADR-010 — Wide datapath, SRA and MAX

## Status
Implemented (PLAN.md SG2). Default build unchanged.

## Context
An int8 matmul with K=16 needs ≥20-bit accumulators and the SwiGLU MLP
needs ~400 words of weights; the 8-bit, 256-word machine fits neither.
Requantization needs an arithmetic right shift, and ReLU/clamping need a
signed comparison, which `CMP` cannot provide (ADR-001).

## Decision
- `gpu.WORD_BITS` (default 8) sets the width of registers, ALU, LSU data,
  shared memory and the NZP-carrying `alu_out`. It must equal
  `DATA_MEM_DATA_BITS`; elaboration stops with `$fatal` otherwise. The
  transformer build uses `WORD_BITS=32`, `DATA_MEM_DATA_BITS=32`,
  `DATA_MEM_ADDR_BITS=12` (`make compile_xf`).
- LSU global addresses are the low `DATA_MEM_ADDR_BITS` bits of `rs`
  (previously hard-wired `[7:0]`); SMEM addresses stay the low 8 bits.
- Instruction encoding, immediates (8-bit, zero-extended), PC width, the
  DCR thread count and `%blockIdx` (8-bit, zero-extended) are unchanged.
- `1101 SRA rd, rs, rt`: `rd ← $signed(rs) >>> rt`, `rt` unsigned over the
  full word, so shifts ≥ `WORD_BITS` give 0 or −1.
- `1110 MAX rd, rs, rt`: signed maximum.
- The ALU mux widens from 2 to 3 bits; `CMP` keeps its unsigned
  "differs" N flag at every width.
- `run_kernel` takes the data-memory width from the DUT, so the existing
  kernel tests run unchanged at 32 bits (`make regression_xf`).

## Alternatives rejected
- Multi-precision int32 on 8-bit registers: needs carry/MULH ops and
  roughly triples kernel length.
- Signed `CMP`: changes upstream's branch semantics that existing kernels
  rely on; `MAX` covers the kernels' needs without touching control flow.

## Verification
`test/test_isa_ext.py` (8-bit, `make test_isa_ext`) and
`test/test_xf_isa_ext.py` (32-bit, `make test_xf_isa_ext`): SRA by 0, 1,
W−1, W, 31, 255 and an all-ones shift on extreme and negative operands;
MAX at sign boundaries; CMP flags on all pairs of 0, 1, 2, −1, min, max.
