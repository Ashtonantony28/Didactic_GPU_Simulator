"""Parameter-sweep / negative tests.

Each kernel runs under varying ICACHE_LINES / COALESCE / PIPELINED / DCACHE_LINES
parameter combinations and must produce identical functional results.

Only the smallest matadd-style kernel is exercised here; the bench rolls up
the full regression.
"""
import cocotb
from .helpers.runner import run_kernel


def encode(op, rd=0, rs=0, rt=0, imm=None):
    if imm is not None:
        return (op << 12) | (rd << 8) | (imm & 0xFF)
    return (op << 12) | (rd << 8) | (rs << 4) | rt


ADD = 0b0011
LDR = 0b0111
STR = 0b1000
CONST = 0b1001
RET = 0b1111
THREADIDX = 15


PROGRAM = [
    encode(CONST, rd=0, imm=0),
    encode(ADD,   rd=0, rs=0, rt=THREADIDX),
    encode(CONST, rd=1, imm=0),
    encode(CONST, rd=2, imm=8),
    encode(ADD,   rd=3, rs=1, rt=0),
    encode(LDR,   rd=4, rs=3, rt=0),
    encode(ADD,   rd=5, rs=2, rt=0),
    encode(STR,   rs=5, rt=4),
    encode(RET),
]
DATA = [11, 22, 33, 44, 0, 0, 0, 0]


@cocotb.test()
async def test_param_sweep(dut):
    """Run the same kernel under whatever params iverilog was built with;
    correctness must hold regardless of parameter combination."""
    metrics, mem = await run_kernel(dut, PROGRAM, DATA, threads=4)
    for i in range(4):
        assert mem.memory[8 + i] == DATA[i], (
            f"thread {i}: got {mem.memory[8+i]} want {DATA[i]}"
        )
