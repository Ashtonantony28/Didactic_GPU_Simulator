"""Stress tests — larger kernels and longer regression than the unit tests.

Each test runs to completion within a fixed cycle budget; this catches
regressions where a phase causes the scheduler to stall indefinitely.
"""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


def encode(op, rd=0, rs=0, rt=0, imm=None, nzp=None):
    if imm is not None and nzp is not None:
        return (op << 12) | (nzp << 9) | (imm & 0xFF)
    if imm is not None:
        return (op << 12) | (rd << 8) | (imm & 0xFF)
    return (op << 12) | (rd << 8) | (rs << 4) | rt


ADD = 0b0011
SUB = 0b0100
MUL = 0b0101
CMP = 0b0010
BR = 0b0001
LDR = 0b0111
STR = 0b1000
CONST = 0b1001
RET = 0b1111
THREADIDX = 15


@cocotb.test()
async def test_long_loop(dut):
    """A kernel that loops 8 times per thread, accumulating sum 0+1+...+7=28."""
    p = [
        # 0  R0 = threadIdx (kept unused for outputs)
        encode(CONST, rd=0, imm=0),
        encode(ADD,   rd=0, rs=0, rt=THREADIDX),
        # 2  R1 = 8 (loop limit)
        encode(CONST, rd=1, imm=8),
        # 3  R2 = 0 (counter)
        encode(CONST, rd=2, imm=0),
        # 4  R3 = 0 (accumulator)
        encode(CONST, rd=3, imm=0),
        # 5  R4 = 1 (increment)
        encode(CONST, rd=4, imm=1),
        # 6: LOOP
        encode(ADD,   rd=3, rs=3, rt=2),       # acc += counter
        encode(ADD,   rd=2, rs=2, rt=4),       # counter += 1
        encode(CMP,   rs=2, rt=1),             # compare counter to limit
        encode(BR,    imm=6, nzp=0b100),       # BRn LOOP — N=1 while counter != limit (NZP quirk)
        # 10 store acc
        encode(CONST, rd=5, imm=16),
        encode(ADD,   rd=6, rs=5, rt=THREADIDX),
        encode(STR,   rs=6, rt=3),
        encode(RET),
    ]
    metrics, mem = await run_kernel(dut, p, [0] * 32, threads=4, timeout_cycles=10_000)
    # 0+1+...+7 = 28
    for i in range(4):
        assert mem.memory[16 + i] == 28, f"thread {i}: got {mem.memory[16+i]} want 28"
    write_bench_result("stress_long_loop", metrics)


@cocotb.test()
async def test_many_threads(dut):
    """16 threads → 4 blocks; exercises core reuse."""
    p = [
        encode(CONST, rd=0, imm=0),
        encode(ADD,   rd=0, rs=0, rt=THREADIDX),
        # global id i = blockIdx * blockDim + threadIdx
        encode(MUL,   rd=1, rs=13, rt=14),    # blockIdx*blockDim
        encode(ADD,   rd=1, rs=1, rt=0),
        encode(CONST, rd=2, imm=64),
        encode(ADD,   rd=3, rs=2, rt=1),       # &out[i]
        # Compute i*2
        encode(ADD,   rd=4, rs=1, rt=1),
        encode(STR,   rs=3, rt=4),
        encode(RET),
    ]
    DATA = [0] * 128
    metrics, mem = await run_kernel(dut, p, DATA, threads=16, timeout_cycles=20_000)
    for i in range(16):
        assert mem.memory[64 + i] == i * 2 % 256, (
            f"thread {i}: got {mem.memory[64+i]} want {i*2 % 256}"
        )
    write_bench_result("stress_many_threads", metrics)
