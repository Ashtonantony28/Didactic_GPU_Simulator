# ISA — tiny-gpu-extended

16-bit fixed-width instructions. 16 opcodes. Compatible with upstream
[adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu).

## Register file

16 registers per thread, `WORD_BITS` bits each (default 8; 32 for the
transformer build, see [ADR-010](decisions/ADR-010-wide-datapath.md)). Arithmetic wraps
modulo 2^`WORD_BITS`; `CONST` and `%blockIdx`/`%blockDim`/`%threadIdx` are
zero-extended. `LDR`/`STR` use the low `DATA_MEM_ADDR_BITS` bits of `rs` as the
address, `LDS`/`STS` the low 8.

| Index | Mnemonic     | Notes                      |
|-------|--------------|----------------------------|
| R0–R12| general-purpose | read/write              |
| R13   | %blockIdx    | read-only, set by hardware |
| R14   | %blockDim    | read-only, = THREADS_PER_BLOCK |
| R15   | %threadIdx   | read-only, set by hardware |

## Instruction encoding

```
 15  12 11  8 7   4 3   0
+------+-----+-----+-----+
| op   |  rd |  rs |  rt |    (3-register form: ADD, SUB, MUL, DIV, SRA, MAX)
+------+-----+-----+-----+

 15  12 11  8 7         0
+------+-----+-----------+
| op   |  rd | immediate |    (CONST)
+------+-----+-----------+

 15  12 11 9  8  7         0
+------+----+---+-----------+
|0001  | nzp| 0 | immediate |    (BRnzp; SYNC when nzp = 000)
+------+----+---+-----------+
```

## Opcode table

| Opcode  | Mnemonic | Semantics                                                       |
|---------|----------|-----------------------------------------------------------------|
| `0000`  | NOP      | no-op                                                           |
| `0001`  | BRnzp    | if `(NZP & nzp) != 0` then `PC ← imm`; the block follows thread 0 (below) |
| `0010`  | CMP      | set NZP from `rs - rt` (unsigned wrap)                          |
| `0011`  | ADD      | `rd ← rs + rt`                                                  |
| `0100`  | SUB      | `rd ← rs - rt`                                                  |
| `0101`  | MUL      | `rd ← rs * rt`                                                  |
| `0110`  | DIV      | `rd ← rs / rt`  (returns 0 if rt = 0)                           |
| `0111`  | LDR      | `rd ← MEM[rs]`                                                  |
| `1000`  | STR      | `MEM[rs] ← rt`                                                  |
| `1001`  | CONST    | `rd ← imm`                                                      |
| `1101`  | SRA      | `rd ← rs >>> rt`  (signed `rs`, unsigned `rt`; `rt ≥ WORD_BITS` fills with the sign) |
| `1110`  | MAX      | `rd ← max(rs, rt)`  (signed)                                    |
| `1111`  | RET      | finish current block                                            |

## Branches and SYNC

Every thread evaluates `BRnzp` with its own NZP, but the block (warp) has one PC: the next PC is
thread 0's target. Branch conditions must therefore be uniform across the block; a disagreeing
thread silently follows thread 0 (AUDIT-4 M1).

| Encoding | Mnemonic | Semantics |
|---|---|---|
| `0001`, nzp `000` | SYNC | set the reconvergence PC of the warp's next divergent branch to `imm`; no-op without the SIMT stack |

`SYNC` and the SIMT-stack semantics are specified in
[ADR-011](decisions/ADR-011-sync-simt-stack.md); `xf/isasim.py` is the reference model.

## NZP semantics (compatibility note)

`CMP rs, rt` sets `NZP = {N, Z, P}` where

```
N = (rs - rt > 0)    // unsigned: equivalent to "rs ≠ rt"
Z = (rs - rt == 0)
P = (rs - rt < 0)    // unsigned: always 0
```

The N flag collapses `>` and unsigned wrap-around (which is upstream's
matmul kernel's branch predicate). Kept bit-for-bit compatible with
upstream — see [ADR-001](decisions/ADR-001-deviations.md).

## Shared memory (Phase 7)

| Opcode  | Mnemonic | Semantics                                              |
|---------|----------|--------------------------------------------------------|
| `1010`  | LDS      | `rd ← SMEM[rs]`  (per-core shared memory load)         |
| `1011`  | STS      | `SMEM[rs] ← rt`                                        |
| `1100`  | BAR      | barrier: waits until every unfinished warp of the block reaches a BAR (ADR-012); a no-op at `WARPS_PER_CORE=1` |

SMEM is `SMEM_SIZE` (default 64) words per core, zeroed on reset; addresses
are masked to `$clog2(SMEM_SIZE)` bits.
See [ADR-007](decisions/ADR-007-shared-memory.md) for the design.

## Assembler

`xf/isa.py` is the single source for opcodes and field layout; `xf/asm.py`
assembles `kernels/*.asm` (`python3 -m xf.asm file.asm`). Branches are
written `BRn`, `BRzp`, `BRnzp`, … (a bare `BR` is rejected) or `SYNC`, with a label or instruction index
inside the program (a target past the last instruction is rejected);
immediates are `#5` or `#0x80` (0–255).

## Signed ops (ADR-010)

`SRA` and `MAX` are the only signed instructions; `ADD`/`SUB`/`MUL` are
sign-agnostic modulo 2^`WORD_BITS`, `DIV` and `CMP` stay unsigned. `SRA` is
floor division by 2^`rt` (requantization); `MAX` gives ReLU (`MAX rd, rs, R0`
with `R0 = 0`) and clamping, with `MIN(x, c) = -MAX(-x, -c)`.
