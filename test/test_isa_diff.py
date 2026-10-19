"""RTL vs xf.isasim: every 8-bit ISA fixture (test/fixtures/isa) must leave data memory bit-identical."""
from enum import Enum

import cocotb

from xf.isa_fixtures import FIXTURE_DIR, load
from .helpers.runner import data_bus, run_kernel


def diff_test(word_bits):
    # An Enum so cocotb names each test after its fixture (it numbers long strings).
    cases = Enum("Case", sorted((p.stem,) * 2 for p in FIXTURE_DIR.glob("*.json")
                                if load(p)["params"]["word_bits"] == word_bits))

    @cocotb.test(name="isa_diff")
    @cocotb.parametrize(case=list(cases))
    async def isa_diff(dut, case):
        case = case.value
        fx = load(FIXTURE_DIR / f"{case}.json")
        p = fx["params"]
        assert data_bus(dut) == (p["word_bits"], p["addr_bits"]), f"{case}: DUT data bus is {data_bus(dut)}, fixture {p}"
        assert int(dut.THREADS_PER_BLOCK.value) == p.get("threads_per_block", 4), f"{case}: THREADS_PER_BLOCK differs"
        _, mem = await run_kernel(dut, fx["program"], fx["data"], fx["threads"], timeout_cycles=200_000)
        want = fx["memory"] + [0] * (len(mem.memory) - len(fx["memory"]))
        bad = [a for a, (g, w) in enumerate(zip(mem.memory, want)) if g != w]
        assert not bad, (f"{case}: {len(bad)} words differ, first at {bad[0]}: RTL {mem.memory[bad[0]]:#x}, "
                         f"model {want[bad[0]]:#x}")

    return isa_diff


test_isa_diff = diff_test(8)
