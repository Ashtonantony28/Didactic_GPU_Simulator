"""In-depth tests for Phase 7 shared memory (LDS / STS / BAR)."""
import cocotb
from .helpers.runner import run_kernel, write_bench_result


def encode(op, rd=0, rs=0, rt=0, imm=None):
    if imm is not None:
        return (op << 12) | (rd << 8) | (imm & 0xFF)
    return (op << 12) | (rd << 8) | (rs << 4) | rt


ADD = 0b0011
SUB = 0b0100
MUL = 0b0101
LDR = 0b0111
STR = 0b1000
CONST = 0b1001
LDS = 0b1010
STS = 0b1011
BAR = 0b1100
RET = 0b1111
THREADIDX = 15


@cocotb.test()
async def test_smem_roundtrip(dut):
    """Each thread: STS smem[t] = t+1, then LDS r = smem[t], then STR mem[t] = r."""
    p = [
        encode(CONST, rd=0, imm=0),
        encode(ADD,   rd=0, rs=0, rt=THREADIDX),     # R0 = threadIdx
        encode(CONST, rd=1, imm=1),
        encode(ADD,   rd=2, rs=0, rt=1),             # R2 = t+1
        encode(STS,   rs=0, rt=2),                   # SMEM[t] = t+1
        encode(BAR),
        encode(LDS,   rd=3, rs=0),                   # R3 = SMEM[t]
        encode(STR,   rs=0, rt=3),                   # MEM[t] = R3
        encode(RET),
    ]
    metrics, mem = await run_kernel(dut, p, [0] * 16, threads=4)
    for i in range(4):
        assert mem.memory[i] == i + 1, f"thread {i}: got {mem.memory[i]} want {i+1}"
    write_bench_result("smem_roundtrip", metrics)


@cocotb.test()
async def test_smem_cross_thread_read(dut):
    """thread t writes SMEM[t]=t+10; then thread t reads SMEM[(t+1) mod 4] and stores it."""
    p = [
        encode(CONST, rd=0, imm=0),
        encode(ADD,   rd=0, rs=0, rt=THREADIDX),         # R0 = t
        encode(CONST, rd=1, imm=10),
        encode(ADD,   rd=2, rs=0, rt=1),                 # R2 = t + 10
        encode(STS,   rs=0, rt=2),                       # SMEM[t] = t+10
        encode(BAR),
        # Compute (t + 1) mod 4. R1 holds 10, so use R10 for the constant 1.
        encode(CONST, rd=10, imm=1),
        encode(CONST, rd=4, imm=4),
        encode(ADD,   rd=5, rs=0, rt=10),                # R5 = t + 1
        # R5 in 1..4; result = R5 - 4*(R5/4)
        encode(0b0110, rd=6, rs=5, rt=4),                # R6 = R5 / 4
        encode(MUL,   rd=6, rs=6, rt=4),                 # R6 = (R5/4)*4
        encode(SUB,   rd=5, rs=5, rt=6),                 # R5 = R5 - that
        encode(LDS,   rd=7, rs=5),                       # R7 = SMEM[(t+1)%4]
        encode(CONST, rd=8, imm=16),
        encode(ADD,   rd=9, rs=8, rt=0),                 # &out[t]
        encode(STR,   rs=9, rt=7),                       # mem[16+t] = SMEM[(t+1)%4]
        encode(RET),
    ]
    metrics, mem = await run_kernel(dut, p, [0] * 32, threads=4)
    expected = [(i + 1) % 4 + 10 for i in range(4)]
    for i, e in enumerate(expected):
        assert mem.memory[16 + i] == e, f"thread {i}: got {mem.memory[16+i]} want {e}"
    write_bench_result("smem_cross_thread", metrics)


@cocotb.test()
async def test_smem_persistence(dut):
    """Multiple LDS to the same address should return the same value."""
    p = [
        encode(CONST, rd=0, imm=42),
        encode(CONST, rd=1, imm=0),
        encode(STS,   rs=1, rt=0),               # SMEM[0] = 42
        encode(LDS,   rd=2, rs=1),               # R2 = SMEM[0]
        encode(LDS,   rd=3, rs=1),               # R3 = SMEM[0]
        encode(ADD,   rd=4, rs=2, rt=3),         # R4 = 84
        encode(STR,   rs=1, rt=4),               # mem[0] = 84
        encode(RET),
    ]
    metrics, mem = await run_kernel(dut, p, [0] * 16, threads=1)
    assert mem.memory[0] == 84, f"got {mem.memory[0]}"
    write_bench_result("smem_persistence", metrics)
