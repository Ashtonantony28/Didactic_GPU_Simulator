"""AUDIT-12 repros (test coverage). Each xfail is strict at 44a396c and passes once the gap is closed; the
controls pass and check that the helpers below can see what the xfails look for."""
import re
import types

import pytest

from xf import isasim
from xf.asm import assemble
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.tests.legacy import ROOT

MAKEFILE = (ROOT / "Makefile").read_text()


def _var(name):
    """A Makefile variable's right-hand side (`NAME := ...`)."""
    m = re.search(rf"^{re.escape(name)}\s*:?=\s*(.*)$", MAKEFILE, re.M)
    return m[1] if m else ""


def _matrix_suite(vvp):
    """The test list of matrix_run's suite on `vvp` (sim.vvp: 8-bit, sim_xf.vvp: xf widths)."""
    body = MAKEFILE.split("matrix_run:", 1)[1].split("\n\n", 1)[0]
    return next(line for line in body.splitlines() if f"$(BUILD)/{vvp}," in line)


def _test_sources():
    """Every cocotb module and pytest file except this one."""
    files = list((ROOT / "test").glob("*.py")) + list((ROOT / "test" / "helpers").glob("*.py"))
    files += [p for p in (ROOT / "xf" / "tests").glob("*.py") if p.name != "test_audit12.py"]
    return {p.name: p.read_text() for p in files}


def _model_mutant(old, new):
    """xf.isasim with one source edit, as a fresh module."""
    src = open(isasim.__file__).read()
    assert src.count(old) == 1, old
    m = types.ModuleType("isasim_mutant")
    exec(compile(src.replace(old, new), "isasim_mutant", "exec"), m.__dict__)
    return m


def _fixtures_that_differ(mutant):
    """(fixture, warps) whose memory differs between xf.isasim and `mutant`, at every warps count test_isa_diff uses."""
    out = []
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = load(p)
        for w in (1, 2, 4):
            if fx["params"].get("threads_per_block", 4) % w:
                continue
            args = (fx["program"], fx["data"], fx["threads"])
            a = isasim.run(*args, **fx["params"], warps_per_core=w)
            b = mutant.run(*args, **fx["params"], warps_per_core=w)
            if a.memory != b.memory:
                out.append((p.stem, w))
    return out


# ---- H1: coalescing at 12-bit addresses -------------------------------------------------------------------------

def _controller_loads_above_byte():
    """Does any test_controller probe load from an address >= 256 at the xf widths? Every word holds its address + 1,
    so a stored value > 256 came from such a load."""
    from test import test_controller as tc
    for src in (v for v in vars(tc).values() if isinstance(v, str) and "LDR" in v):
        prog = assemble(src.format(n=tc.N))
        mem = isasim.run(prog, [a + 1 for a in range(4096)], tc.N, word_bits=32, addr_bits=12).memory
        if any(v > 256 for a, v in enumerate(mem) if v != a + 1):
            return True
    return False


@pytest.mark.xfail(strict=True, reason="AUDIT-12 H1: the coalescing test never runs at 12-bit addresses")
def test_coalescing_probe_runs_at_xf_addresses():
    """K08 (coalesce on equal low address bytes) survives matrix_coalesce and matrix_all: test_controller runs only
    in the 8-bit suite, and no xf test has two lanes load a and a + 256k in the same cycle."""
    xf_suite = _matrix_suite("sim_xf.vvp")
    reused = "controller" in xf_suite or any("test_controller" in s for n, s in _test_sources().items()
                                            if n.startswith("test_xf_"))
    assert reused and _controller_loads_above_byte()


# ---- H2: I-cache hit/miss counters ------------------------------------------------------------------------------

# A comparison of one counter with an expected value; `... == sum(per_core)` (the per-core total) does not count.
ICACHE_EXACT = re.compile(r"""\[["']icache_(hits|misses)["']\]\s*==(?!\s*sum\()|==\s*\w*\[["']icache_(hits|misses)["']\]|"""
                          r"""icache_(hits|misses)_total\.value\)?\s*==""")


@pytest.mark.xfail(strict=True, reason="AUDIT-12 H2: no test checks an I-cache hit or miss count against a value")
def test_icache_counts_are_asserted():
    """I02 (hits and misses swapped) and I04 (hits never counted) survive matrix_icache and matrix_all. The only checks
    are that the total equals the per-core sum (runner.check_counters, test_controller), which both mutants keep."""
    hits = {n for n, s in _test_sources().items() if ICACHE_EXACT.search(s)}
    assert hits, "no test compares icache_hits or icache_misses with an expected count"


# ---- H3: STS lane order ------------------------------------------------------------------------------------------

STS_LOOP = "                for a, v in [(regs[t][rs] & 0xFF & (smem_size - 1), regs[t][rt]) for t in lanes]:"


@pytest.mark.xfail(strict=True, reason="AUDIT-12 H3: no RTL differential case has two lanes STS one address")
def test_isa_fixtures_pin_sts_lane_order():
    """S02 (lowest lane wins an STS conflict) survives every target. The model pins 'highest lane wins'
    (test_isasim::test_smem_highest_lane_wins), but no ISA fixture tells the two orders apart, so the RTL is never
    compared on it."""
    lowest_wins = _model_mutant(STS_LOOP, STS_LOOP.replace("for a, v in [", "for a, v in reversed([")
                                .replace("for t in lanes]:", "for t in lanes]):"))
    assert _fixtures_that_differ(lowest_wins)


# ---- M1: GTLRR inside the core ----------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-12 M1: WARP_POLICY=2 inside the core runs in neither test_all nor matrix")
def test_gtlrr_core_is_gated():
    """W03/W04 (GTLRR reduced to LRR) are killed only by `make test_multiwarp` (multiwarp_w4_gtlrr), which test_all
    and matrix never call; test_warp_scheduler_gtlrr asserts only the issue count."""
    matrix_cfgs = " ".join(_var(f"CFG_{c}") for c in _var("MATRIX").split())
    test_all = MAKEFILE.split("test_all:", 1)[1].split("\n\n", 1)[0]
    assert "WARP_POLICY=2" in matrix_cfgs or "test_multiwarp" in test_all


# ---- M2: contract 8 segment bounds ------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-12 M2: contract 8's ±0.0025 gain and 1.1× RMS bounds are not pinned")
def test_contract8_segment_bounds_pinned():
    """P01 (GAIN_TOL 0.25, RMS_TOL 11) passes pytest with NANOGPT_DIR and export --check: test_audit7 reads the bounds
    from xf.export, so loosening them loosens the test."""
    pinned = [n for n, s in _test_sources().items() if re.search(r"GAIN_TOL\s*==\s*0\.0025|RMS_TOL\s*==\s*1\.1\b", s)]
    assert pinned


# ---- M3: MeDiC test infers installs from the priority signal ----------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-12 M3: test_medic classifies install/bypass from warp_high_util")
def test_medic_observes_the_bypass_decision():
    """L07 (warp_low_utility at h <= 2) survives test_medic: the cache bypasses at h = 2 while warp_high_utility still
    reads 1, and the monitor counts that fill as an install."""
    src = _test_sources()["test_medic.py"]
    monitor = src.split("async def monitor", 1)[1].split("\n\n\n", 1)[0]
    assert "low_util" in monitor or "warp_low_utility" in monitor


# ---- controls ----------------------------------------------------------------------------------------------------

def test_control_matrix_parsing():
    """The helpers read the real matrix: the coalesce, icache and GTO configurations exist and test_controller runs in
    the 8-bit suite."""
    assert "controller" in _matrix_suite("sim.vvp")
    assert _var("CFG_coalesce") == "-Pgpu.COALESCE=1" and _var("CFG_icache") == "-Pgpu.ICACHE_LINES=16"
    assert "WARP_POLICY=1" in _var("CFG_warps4gto")
    assert not _controller_loads_above_byte()


def test_control_model_mutant_harness():
    """_fixtures_that_differ sees a model mutant that the fixtures do pin (SMEM index not masked to 64 words)."""
    old = "vals = [smem[regs[t][rs] & 0xFF & (smem_size - 1)] for t in lanes]"
    assert _fixtures_that_differ(_model_mutant(old, old.replace("& 0xFF & (smem_size - 1)", "& 0x7")))


class _Sig:
    def __init__(self, value, bits):
        self.value, self._bits = value, bits

    def __len__(self):
        return self._bits


def test_control_check_counters_rejects_a_wrong_coalesced_count():
    """runner.check_counters (contract 12) rejects a coalesced count the controller handshakes do not support."""
    from test.helpers.runner import check_counters
    dut = types.SimpleNamespace(per_core_icache_hits=_Sig(3 | 4 << 32, 64), per_core_icache_misses=_Sig(1 | 2 << 32, 64))
    ok = dict(data_mem_requests=10, data_mem_requests_coalesced=2, icache_hits=7, icache_misses=3)
    check_counters(dut, ok, 6, 4, 8)
    with pytest.raises(AssertionError, match="coalesced"):
        check_counters(dut, {**ok, "data_mem_requests_coalesced": 3}, 6, 4, 8)
