"""RTL vs xf.isasim: every 8-bit ISA fixture (test/fixtures/isa) must leave data memory bit-identical.

The model runs with the DUT's WARPS_PER_CORE, and with simt=True at SIMT_STACK=1. At WARPS_PER_CORE > 1 the model
replays the warp issue order the RTL recorded (xf.isasim.run_trace), which must pick only ready warps, so fixtures whose
result depends on the interleaving (sts_conflict) are compared too (AUDIT-14 L1). warp_issues_total must
equal the model's steps. Words a cross-warp STR race leaves unspecified (docs/isa.md) are not compared; only
str_conflict's lane pairs race, at one lane per warp (RACED).
"""
from enum import Enum

import cocotb

from xf.isa_fixtures import FIXTURE_DIR, load
from xf.isasim import run, run_trace
from .helpers.runner import data_bus, memory_diff, run_kernel

# str_conflict at one lane per warp: each lane pair (words 8-11) is two warps' STRs; 24-31 read them back.
RACED = {f"str_conflict{w}": set(range(8, 12)) | set(range(24, 32)) for w in (8, 32)}


def diff_test(word_bits):
    fixtures = {p.stem: load(p) for p in sorted(FIXTURE_DIR.glob("*.json"))}
    fixtures = {k: fx for k, fx in fixtures.items() if fx["params"]["word_bits"] == word_bits}
    # An Enum so cocotb names each test after its fixture (it numbers long strings).
    cases = Enum("Case", [(k,) * 2 for k in sorted(fixtures)])

    @cocotb.test(name="isa_diff")
    @cocotb.parametrize(case=list(cases))
    async def isa_diff(dut, case):
        case = case.value
        fx = fixtures[case]
        p = fx["params"]
        warps, simt = int(dut.WARPS_PER_CORE.value), bool(int(dut.SIMT_STACK.value))
        assert data_bus(dut) == (p["word_bits"], p["addr_bits"]), f"{case}: DUT data bus is {data_bus(dut)}, fixture {p}"
        assert int(dut.THREADS_PER_BLOCK.value) == p.get("threads_per_block", 4), f"{case}: THREADS_PER_BLOCK differs"
        args, kw = (fx["program"], fx["data"], fx["threads"]), dict(p, warps_per_core=warps, simt=simt)
        trace = {} if warps > 1 else None
        _, mem = await run_kernel(dut, *args, timeout_cycles=200_000, trace=trace)
        r = run_trace(trace, *args, **kw) if warps > 1 else run(*args, **kw)
        if warps == 1:
            assert r.memory[:len(fx["memory"])] == fx["memory"], f"{case}: model no longer reproduces the fixture"
        lanes = int(dut.THREADS_PER_BLOCK.value) // warps
        bad = memory_diff(case, mem.memory, r, RACED.get(case, ()) if lanes == 1 else ())
        assert not bad, (f"{case}: {len(bad)} words differ, first at {bad[0]}: RTL {mem.memory[bad[0]]:#x}, "
                         f"model {r.memory[bad[0]]:#x}")
        issues = int(dut.warp_issues_total.value)
        assert issues == r.steps, f"{case}: warp_issues_total {issues} != model steps {r.steps}"
        if warps == 1:
            assert int(dut.bar_stall_cycles_total.value) == 0, f"{case}: bar_stall_cycles_total at one warp"

    return isa_diff


test_isa_diff = diff_test(8)
