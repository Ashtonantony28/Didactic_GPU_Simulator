# ISA — tiny-gpu-extended

16-bit fixed-width instructions. 11 opcodes. Compatible with upstream
[adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu).

## Register file

16 registers per thread, 8 bits each.

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
| op   |  rd |  rs |  rt |    (3-register form: ADD, SUB, MUL, DIV)
+------+-----+-----+-----+

 15  12 11  8 7         0
+------+-----+-----------+
| op   |  rd | immediate |    (CONST)
+------+-----+-----------+

 15  12 11 9 8         0
+------+----+-----------+
|0001  | nzp| immediate |    (BRnzp)
+------+----+-----------+
```

## Opcode table

| Opcode  | Mnemonic | Semantics                                                       |
|---------|----------|-----------------------------------------------------------------|
| `0000`  | NOP      | no-op                                                           |
| `0001`  | BRnzp    | if `(NZP & nzp) != 0` then `PC ← imm`                           |
| `0010`  | CMP      | set NZP from `rs - rt` (unsigned wrap)                          |
| `0011`  | ADD      | `rd ← rs + rt`                                                  |
| `0100`  | SUB      | `rd ← rs - rt`                                                  |
| `0101`  | MUL      | `rd ← rs * rt`                                                  |
| `0110`  | DIV      | `rd ← rs / rt`  (returns 0 if rt = 0)                           |
| `0111`  | LDR      | `rd ← MEM[rs]`                                                  |
| `1000`  | STR      | `MEM[rs] ← rt`                                                  |
| `1001`  | CONST    | `rd ← imm`                                                      |
| `1111`  | RET      | finish current block                                            |

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

## Designed but not yet wired (Phase 7)

These are reserved opcodes documented for the shared-memory phase:

| Opcode  | Mnemonic | Semantics                                              |
|---------|----------|--------------------------------------------------------|
| `1010`  | LDS      | `rd ← SMEM[rs]`  (per-block shared memory load)        |
| `1011`  | STS      | `SMEM[rs] ← rt`                                        |
| `1100`  | BAR      | barrier: all threads in the block wait at this instr   |

See [ADR-007](decisions/ADR-007-shared-memory.md) for the design.
