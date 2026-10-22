"""RTL vs xf.isasim: every 8-bit ISA fixture (test/fixtures/isa) must leave data memory bit-identical.

The model runs with the DUT's WARPS_PER_CORE. At WARPS_PER_CORE > 1 the RTL interleaves warps differently from
the model, so a fixture is compared only if the model gives the same memory and steps under "lrr" and "gto"
(ADR-012); isa_diff_skipped logs the others. warp_issues_total must equal the model's steps.
"""
from enum import Enum

import cocotb

from xf.isa_fixtures import FIXTURE_DIR, load
from xf.isasim import run
from .helpers.logger import logger
from .helpers.runner import data_bus, run_kernel


def model(fx, warps, policy="lrr"):
    return run(fx["program"], fx["data"], fx["threads"], **fx["params"], warps_per_core=warps, warp_policy=policy)


def expected(fx, warps):
    """The model's Result if it does not depend on the warp interleaving, else None."""
    r = model(fx, warps)
    if warps > 1:
        g = model(fx, warps, "gto")
        if (r.memory, r.steps) != (g.memory, g.steps):
            return None
    return r


def diff_test(word_bits):
    top = getattr(cocotb, "top", None)  # None when imported outside a simulation (xf/tests)
    warps = int(top.WARPS_PER_CORE.value) if top is not None else 1
    fixtures = {p.stem: load(p) for p in sorted(FIXTURE_DIR.glob("*.json"))}
    fixtures = {k: fx for k, fx in fixtures.items() if fx["params"]["word_bits"] == word_bits}
    want = {k: expected(fx, warps) for k, fx in fixtures.items()}
    skipped = sorted(k for k, r in want.items() if r is None)
    # An Enum so cocotb names each test after its fixture (it numbers long strings).
    cases = Enum("Case", [(k,) * 2 for k in sorted(want) if want[k] is not None])

    @cocotb.test(name="isa_diff")
    @cocotb.parametrize(case=list(cases))
    async def isa_diff(dut, case):
        case = case.value
        fx, r = fixtures[case], want[case]
        p = fx["params"]
        assert data_bus(dut) == (p["word_bits"], p["addr_bits"]), f"{case}: DUT data bus is {data_bus(dut)}, fixture {p}"
        assert int(dut.THREADS_PER_BLOCK.value) == p.get("threads_per_block", 4), f"{case}: THREADS_PER_BLOCK differs"
        if warps == 1:
            assert r.memory[:len(fx["memory"])] == fx["memory"], f"{case}: model no longer reproduces the fixture"
        _, mem = await run_kernel(dut, fx["program"], fx["data"], fx["threads"], timeout_cycles=200_000)
        bad = [a for a, (g, w) in enumerate(zip(mem.memory, r.memory)) if g != w]
        assert not bad, (f"{case}: {len(bad)} words differ, first at {bad[0]}: RTL {mem.memory[bad[0]]:#x}, "
                         f"model {r.memory[bad[0]]:#x}")
        issues = int(dut.warp_issues_total.value)
        assert issues == r.steps, f"{case}: warp_issues_total {issues} != model steps {r.steps}"
        if warps == 1:
            assert int(dut.bar_stall_cycles_total.value) == 0, f"{case}: bar_stall_cycles_total at one warp"

    @cocotb.test(name="isa_diff_skipped")
    async def isa_diff_skipped(dut):
        """Interleaving-dependent fixtures, not compared at this WARPS_PER_CORE."""
        logger.info(f"isa_diff {word_bits}-bit, WARPS_PER_CORE={warps}: skipped {skipped}")
        assert warps > 1 or not skipped, f"skipped at one warp: {skipped}"

    return isa_diff, isa_diff_skipped


test_isa_diff, test_isa_diff_skipped = diff_test(8)
