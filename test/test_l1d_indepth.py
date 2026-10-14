"""In-depth tests for the Phase 9 L1-D cache.

Verifies, at the build's DCACHE_LINES / MEDIC / word width:
- Exact hit/miss counts for repeated reads of one address (all misses under MeDiC).
- A read after a write by the same thread returns the written value.
- A write by one block is read by a later block on any core.
- Disabling the cache leaves the downstream request count unchanged.
"""
import cocotb
from .helpers.runner import run_kernel, data_bus, write_bench_result


def encode(op, rd=0, rs=0, rt=0, imm=None):
    if imm is not None:
        return (op << 12) | (rd << 8) | (imm & 0xFF)
    return (op << 12) | (rd << 8) | (rs << 4) | rt


ADD = 0b0011
MUL = 0b0101
LDR = 0b0111
STR = 0b1000
CONST = 0b1001
RET = 0b1111
BLOCKIDX = 13
BLOCKDIM = 14
THREADIDX = 15
OUT, KADDR = 8, 7


def params(dut):
    return int(dut.DCACHE_LINES.value), int(dut.MEDIC.value), int(dut.NUM_CORES.value)


def word_mask(dut):
    return (1 << data_bus(dut)[0]) - 1


@cocotb.test()
async def test_hot_address_hits(dut):
    """Every thread reads address 0 three times: 4 misses then 8 hits, or 12 misses under MeDiC."""
    p = [
        encode(CONST, rd=0, imm=0),          # addr = 0
        encode(LDR,   rd=1, rs=0, rt=0),     # R1 = mem[0]
        encode(LDR,   rd=2, rs=0, rt=0),     # R2 = mem[0]
        encode(LDR,   rd=3, rs=0, rt=0),     # R3 = mem[0]
        encode(ADD,   rd=4, rs=1, rt=2),
        encode(ADD,   rd=4, rs=4, rt=3),     # R4 = 3 * mem[0]
        encode(CONST, rd=5, imm=16),
        encode(ADD,   rd=6, rs=5, rt=THREADIDX),
        encode(STR,   rs=6, rt=4),
        encode(RET),
    ]
    DATA = [7] + [0] * 31
    metrics, mem = await run_kernel(dut, p, DATA, threads=4)
    for i in range(4):
        assert mem.memory[16 + i] == 21, f"thread {i}: got {mem.memory[16+i]}"
    write_bench_result("l1d_hot_address", metrics)
    lines, medic, _ = params(dut)
    # MeDiC starts every lane low-utility, so it never installs a line.
    expect = (0, 0) if lines == 0 else (0, 12) if medic else (8, 4)
    got = (metrics["dcache_hits"], metrics["dcache_misses"])
    assert got == expect, f"(hits, misses) = {got}, expected {expect}"


@cocotb.test()
async def test_read_after_write_same_thread(dut):
    """Each thread caches in[t], stores in[t] + 1 over it, reloads it and stores it to out[t]."""
    p = [
        encode(LDR,   rd=1, rs=THREADIDX),   # R1 = in[t] (miss, fill)
        encode(LDR,   rd=1, rs=THREADIDX),   # R1 = in[t] (hit)
        encode(CONST, rd=2, imm=1),
        encode(ADD,   rd=1, rs=1, rt=2),
        encode(STR,   rs=THREADIDX, rt=1),   # in[t] = in[t] + 1 (write hit)
        encode(LDR,   rd=3, rs=THREADIDX),   # R3 = in[t]
        encode(CONST, rd=4, imm=8),
        encode(ADD,   rd=4, rs=4, rt=THREADIDX),
        encode(STR,   rs=4, rt=3),           # out[t] = R3
        encode(RET),
    ]
    mask = word_mask(dut)
    DATA = [(0x7FFFFFF0 + i) & mask for i in range(4)] + [0] * 4
    metrics, mem = await run_kernel(dut, p, DATA, threads=4)
    for i in range(4):
        want = (DATA[i] + 1) & mask
        assert mem.memory[i] == want, f"in[{i}] = {mem.memory[i]:#x}, expected {want:#x}"
        assert mem.memory[8 + i] == want, f"thread {i}: stale read after write {mem.memory[8 + i]:#x}, expected {want:#x}"
    write_bench_result("l1d_read_after_write", metrics)


@cocotb.test()
async def test_write_read_by_later_block(dut):
    """Every block caches s[t], records it in out[b][t], then overwrites s[t] with k.

    Block NUM_CORES starts only after an earlier block has finished, so it must read k.
    """
    _, _, cores = params(dut)
    blocks = cores + 1
    p = [
        encode(LDR,   rd=1, rs=THREADIDX),   # R1 = s[t] (fills the line)
        encode(MUL,   rd=2, rs=BLOCKIDX, rt=BLOCKDIM),
        encode(ADD,   rd=2, rs=2, rt=THREADIDX),
        encode(CONST, rd=3, imm=OUT),
        encode(ADD,   rd=2, rs=2, rt=3),
        encode(STR,   rs=2, rt=1),           # out[4b + t] = s[t]
        encode(CONST, rd=4, imm=KADDR),
        encode(LDR,   rd=4, rs=4),           # R4 = k
        encode(STR,   rs=THREADIDX, rt=4),   # s[t] = k (write hit)
        encode(RET),
    ]
    mask = word_mask(dut)
    k = 0xA5C3E1F7 & mask
    s = [(0x5A3C1E10 + i) & mask for i in range(4)]
    DATA = s + [0] * 3 + [k] + [0] * (4 * blocks)
    metrics, mem = await run_kernel(dut, p, DATA, threads=4 * blocks)
    for i in range(4):
        assert mem.memory[i] == k, f"s[{i}] = {mem.memory[i]:#x}, expected {k:#x}"
        assert mem.memory[OUT + i] == s[i], f"block 0 thread {i}: {mem.memory[OUT + i]:#x}, expected {s[i]:#x}"
        got = mem.memory[OUT + 4 * cores + i]
        assert got == k, f"block {cores} thread {i}: stale read {got:#x}, expected {k:#x}"
    write_bench_result("l1d_cross_block", metrics)


@cocotb.test()
async def test_l1d_disable_equivalence(dut):
    """A copy with no reuse sends the same 8 requests downstream with or without the cache."""
    p = [
        encode(CONST, rd=0, imm=0),
        encode(ADD,   rd=0, rs=0, rt=THREADIDX),
        encode(CONST, rd=1, imm=0),                 # base in
        encode(CONST, rd=2, imm=8),                 # base out
        encode(ADD,   rd=3, rs=1, rt=0),            # &in[t]
        encode(LDR,   rd=4, rs=3, rt=0),
        encode(ADD,   rd=5, rs=2, rt=0),            # &out[t]
        encode(STR,   rs=5, rt=4),                  # out[t] = in[t]
        encode(RET),
    ]
    DATA = [10, 20, 30, 40, 0, 0, 0, 0]
    metrics, mem = await run_kernel(dut, p, DATA, threads=4)
    for i in range(4):
        assert mem.memory[8 + i] == DATA[i]
    assert metrics["data_mem_requests"] == 8, f"data_mem_requests = {metrics['data_mem_requests']}"
    write_bench_result("l1d_disable_equivalence", metrics)
