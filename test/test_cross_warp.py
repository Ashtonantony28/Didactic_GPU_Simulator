"""Cross-warp STRs under channel contention (SG28, AUDIT-18 H1, docs/isa.md): RTL vs xf.isasim replaying the RTL's
issue order, at the AUDIT-18 H1 build (`make matrix` config warps2lat: THREADS_PER_BLOCK=8, WARPS_PER_CORE=2,
I-cache, PIPELINED, MEM_LATENCY=10). Block 1 keeps the data channels busy while block 0's two warps store to word 100.

- Racing STRs (no BAR between the warps) leave word 100 unspecified: the model flags the race and the words that
  depend on it, and every other word must match.
- The same stores in BAR-separated rounds have one result, and every word must match.
"""
import cocotb

from xf.asm import assemble
from xf.isasim import run_trace
from xf.tests.test_audit18 import H1_DATA, H1_SRC
from .helpers.logger import logger
from .helpers.runner import data_bus, memory_diff, run_kernel

# H1_SRC with the two warps' stores in BAR-separated rounds: warp 1 stores t + 1 to word 100 (warp 0 to 120 + t), then
# warp 0 does (warp 1 to 120 + t). Word 100 ends with warp 0's highest lane, 4.
ROUNDS = H1_SRC[:H1_SRC.index("B0:")] + """
B0: CONST R2, #4
    DIV R3, %threadIdx, R2
    LDR R4, R3
D:  SUB R4, R4, R1
    CMP R4, R8
    BRn D
    CONST R9, #100
    ADD R11, %threadIdx, R1
    CONST R12, #20
    ADD R12, R12, %threadIdx
    SUB R10, R1, R3
    MUL R10, R10, R12
    ADD R10, R10, R9
    STR R10, R11
    BAR
    MUL R10, R3, R12
    ADD R10, R10, R9
    STR R10, R11
    BAR
    LDR R7, R9
    CONST R6, #140
    ADD R6, R6, %threadIdx
    STR R6, R7
    RET
"""
# (warp 0 delay, warp 1 delay, block 1 hog phase): AUDIT-18's two H1 hits first.
CASES = [(2, 1, 1), (4, 3, 3), (1, 2, 1), (3, 1, 2), (1, 1, 1), (5, 2, 4)]
RACED = {100, *range(140, 148)}


async def replay(dut, src, d0, d1, hog):
    assert (int(dut.THREADS_PER_BLOCK.value), int(dut.WARPS_PER_CORE.value)) == (8, 2), "needs TPB 8, 2 warps"
    data = [d0, d1, hog] + H1_DATA[3:]
    prog, trace, (bits, addr) = assemble(src), {}, data_bus(dut)
    _, mem = await run_kernel(dut, prog, data, 16, timeout_cycles=200_000, trace=trace)
    r = run_trace(trace, prog, data, 16, word_bits=bits, addr_bits=addr, threads_per_block=8, warps_per_core=2)
    issues = int(dut.warp_issues_total.value)
    assert issues == r.steps, f"warp_issues_total {issues} != model steps {r.steps}"
    return mem.memory, r


def check(name, rtl, r, raced=()):
    bad = memory_diff(name, rtl, r, raced)
    assert not bad, f"{name}: words {bad} differ: RTL {[rtl[a] for a in bad]}, model {[r.memory[a] for a in bad]}"


@cocotb.test()
async def test_race_flagged_and_excluded(dut):
    """H1's racing STRs: the race is flagged, words 100 and 140-147 are left out, every other word matches."""
    for case in CASES:
        rtl, r = await replay(dut, H1_SRC, *case)
        assert r.store_races == [(0, 35, 100)], r.store_races
        check(f"race {case}", rtl, r, RACED)
        logger.info(f"race {case}: word 100 RTL {rtl[100]}, model {r.memory[100]} (issue order)")


@cocotb.test()
async def test_bar_separated_rounds(dut):
    """The same stores in BAR-separated rounds: no race, every word matches, warp 0's lane 3 wins word 100."""
    for case in CASES:
        rtl, r = await replay(dut, ROUNDS, *case)
        check(f"rounds {case}", rtl, r)
        assert rtl[100] == 4 and rtl[140:148] == [4] * 8 and rtl[120:128] == list(range(1, 9)), rtl[100:148]
