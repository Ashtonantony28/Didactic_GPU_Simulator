"""SRA, MAX and CMP at the DUT's word width (8 here, 32 via test_xf_isa_ext)."""
import random

import cocotb

from xf.asm import assemble
from .helpers.runner import data_bus, run_kernel

SRA_MAX = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #{n}
    ADD R2, R0, R1
    LDR R3, R0          ; a[i]
    LDR R4, R2          ; b[i]
    ADD R5, R2, R1
    SRA R6, R3, R4
    STR R5, R6          ; sra[i] at 2n+i
    ADD R5, R5, R1
    MAX R6, R3, R4
    STR R5, R6          ; max[i] at 3n+i
    RET
"""

# One case per block: thread 0 owns the block PC, so the branches are uniform within a block.
CMP_FLAGS = """
    CONST R1, #{n}
    LDR R4, %blockIdx
    ADD R2, %blockIdx, R1
    LDR R5, R2
    ADD R3, R2, R1
    CONST R9, #7
    CONST R10, #1
    CONST R11, #2
    CONST R12, #4
    CMP R4, R5
    BRn has_n
    SUB R9, R9, R12
has_n:
    BRz has_z
    SUB R9, R9, R11
has_z:
    BRp has_p
    SUB R9, R9, R10
has_p:
    STR R3, R9          ; nzp code at 2n+block
    RET
"""


def _signed(v, w):
    v &= (1 << w) - 1
    return v - (1 << w) if v >> (w - 1) else v


def _operand_pairs(w):
    lo, hi, mask = -(1 << (w - 1)), (1 << (w - 1)) - 1, (1 << w) - 1
    rng = random.Random(w)
    shifts = [0, 1, w - 1, w, 31, 255, mask]
    vals = [0, 1, -1, hi, lo, hi - 1, lo + 1, -2]
    sra = [(a, s) for a in (hi, lo, -1, 1, 0, -96) for s in shifts]
    mx = [(lo, hi), (hi, lo), (-1, 0), (0, -1), (lo, lo + 1), (hi, hi - 1), (-1, -1), (lo, lo)]
    rnd = [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(8)]
    pairs = sra + mx + rnd + [(a, b) for a in vals[:3] for b in vals[3:5]]
    pairs += [(0, 0)] * (-len(pairs) % 4)
    return [(a & mask, b & mask) for a, b in pairs]


@cocotb.test()
async def test_sra_max(dut):
    w, _ = data_bus(dut)
    pairs = _operand_pairs(w)
    n = len(pairs)
    assert 4 * n <= 256
    data = [a for a, _ in pairs] + [b for _, b in pairs]
    _, mem = await run_kernel(dut, assemble(SRA_MAX.format(n=n)), data, n)
    mask = (1 << w) - 1
    for i, (a, b) in enumerate(pairs):
        want_sra = (_signed(a, w) >> b) & mask
        want_max = max(_signed(a, w), _signed(b, w)) & mask
        assert mem.memory[2 * n + i] == want_sra, f"SRA {a:#x}, {b}: got {mem.memory[2*n+i]:#x} want {want_sra:#x}"
        assert mem.memory[3 * n + i] == want_max, f"MAX {a:#x}, {b:#x}: got {mem.memory[3*n+i]:#x} want {want_max:#x}"


@cocotb.test()
async def test_cmp_unchanged(dut):
    w, _ = data_bus(dut)
    mask = (1 << w) - 1
    vals = [0, 1, 2, mask, 1 << (w - 1), (1 << (w - 1)) - 1]
    pairs = [(a, b) for a in vals for b in vals]
    n = len(pairs)
    data = [a for a, _ in pairs] + [b for _, b in pairs]
    _, mem = await run_kernel(dut, assemble(CMP_FLAGS.format(n=n)), data, 4 * n)
    for i, (a, b) in enumerate(pairs):
        want = 0b100 if a != b else 0b010  # N = "differs", Z = equal, P never (ADR-001)
        assert mem.memory[2 * n + i] == want, f"CMP {a:#x}, {b:#x}: got {mem.memory[2*n+i]:03b} want {want:03b}"
