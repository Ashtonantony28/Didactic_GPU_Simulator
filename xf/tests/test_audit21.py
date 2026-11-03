"""AUDIT-21 repros (final audit: SG28's cross-warp STR race exclusion and the warps2lat matrix config). Strict xfails
fail today and pass once the finding is fixed; the controls check the machinery the repros rely on. RTL-side numbers
come from build/a21 and build/mut runs recorded in AUDIT-21.md (no simulator here).
"""
import re

import numpy as np

from test.helpers.runner import MAX_RACED, memory_diff
from xf import elementwise as ew, golden
from xf.asm import assemble
from xf.image import Image
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.isasim import run
from xf.tests.legacy import ROOT
from xf.tests.test_audit18 import H1_RTL_DIFF, _h1_replay

import pytest

H1_RACED = set(H1_RTL_DIFF)                 # {100, 140..147}: test_cross_warp.RACED
H1_LEGAL_WORD100 = {4, 8}                   # warp 0's lane 3 or warp 1's lane 3 (highest lane of each warp, ADR-016)


# ---- controls ----------------------------------------------------------------------------------------------------

def test_memory_diff_control():
    """memory_diff leaves out exactly the named raced words, catches any other differing word, and enforces the exact
    set and the 16-word cap."""
    r = _h1_replay()
    rtl = list(r.memory)
    rtl[100] = 8
    rtl[140:148] = [8] * 8                  # the RTL's real H1 result (AUDIT-18)
    assert memory_diff("h1", rtl, r, H1_RACED) == []
    rtl[120] ^= 1
    assert memory_diff("h1", rtl, r, H1_RACED) == [120]
    with pytest.raises(AssertionError):
        memory_diff("h1", rtl, r, H1_RACED - {147})
    with pytest.raises(AssertionError):
        memory_diff("h1", rtl, r, set(range(MAX_RACED + 1)))


def _rmsnorm_cases():
    """test_xf_rmsnorm.test_synthetic's cases (same rng and order)."""
    rng = np.random.default_rng(5)
    img, cases = Image(), []

    def seg(name, data, dtype="i8"):
        img.alloc(name, np.shape(data), dtype, data=np.asarray(data, dtype=np.int64))

    def case(name, x, w, lut, m, s):
        seg(f"{name}.x", x)
        img.alloc(name, np.shape(x), "i8")
        cases.append((name, ew.rmsnorm(img, f"{name}.x", w, lut, name, m, s),
                      golden.rmsnorm_int(x, img.read(w), img.read(lut), m, s)))

    for i, (t, d, n) in enumerate(((4, 8, 64), (3, 5, 16), (5, 3, 8), (2, 16, 1024), (1, 1, 4))):
        seg(f"w{i}", rng.integers(-128, 128, d))
        lut = rng.integers(0, 1 << 10, n)
        lut[[0, -1]] = [(1 << 10) - 1, 0]
        seg(f"lut{i}", lut, "i32")
        m, s = int(rng.integers(1, 1 << 6)), int(rng.integers(12, 20))
        x = rng.integers(-128, 128, (t, d)) >> rng.integers(0, 8, (t, 1))
        case(f"rand{i}", x, f"w{i}", f"lut{i}", m, s)
        case(f"zero{i}", np.zeros((t, d)), f"w{i}", f"lut{i}", m, s)
        case(f"ext{i}", np.where(rng.random((t, d)) < 0.5, -128, 127), f"w{i}", f"lut{i}", m, s)
    seg("w_sat", [-128, 127, -128, 127])
    seg("lut_sat", [(1 << 17) - 1, 1 << 16, 1 << 15, 1 << 14], "i32")
    case("sat", [[-128, -128, 127, 127], [1, -1, 1, -1]], "w_sat", "lut_sat", 1, 0)
    return img, cases


def test_rmsnorm_model_at_warps2lat_shape_matches_golden():
    """Control for item 2: the rmsnorm kernel has no 4-thread assumption. At THREADS_PER_BLOCK 8, 2 warps the ISA
    model reproduces golden.rmsnorm_int on every test_synthetic case (the RTL passes too: AUDIT-21 L3)."""
    img, cases = _rmsnorm_cases()
    mem = list(img.words)
    for _, (prog, threads), _ in cases:
        mem = run(prog, mem, threads, word_bits=32, addr_bits=12, threads_per_block=8, warps_per_core=2).memory
    assert [n for n, _, want in cases if not np.array_equal(img.read(n, mem), want)] == []


# ---- M1: an excluded raced word may hold any value -----------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-21 M1: memory_diff skips raced words whatever the RTL wrote")
def test_m1_raced_word_must_hold_a_legal_outcome():
    """H1's word 100 can legally end as 4 or 8 (the highest lane of either warp's STR). The lane-order mutant
    (build/mut/rev: writes claim channels from the highest consumer down) leaves 1 or 5 there, and its read-backs at
    140..147; test_cross_warp::test_race_flagged_and_excluded passes on it. memory_diff should reject a raced word
    whose value no order of the racing stores produces."""
    r = _h1_replay()
    rtl = list(r.memory)
    rtl[100] = 1
    rtl[140:148] = [1] * 8                  # observed on build/mut/rev, case (2, 1, 1)
    assert rtl[100] not in H1_LEGAL_WORD100
    assert memory_diff("h1", rtl, r, H1_RACED) != []


# ---- L1: str_conflict at one lane per warp compares no output word -------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-21 L1: test_isa_diff RACED covers every word str_conflict writes")
@pytest.mark.parametrize("bits", [8, 32])
def test_l1_str_conflict_keeps_a_compared_word_at_one_lane_per_warp(bits):
    """At warps4gto (TPB 4, W 4: one lane per warp) the race exclusion covers all 12 words the fixture changes, so the
    fixture checks nothing there but warp_issues_total."""
    fx = load(FIXTURE_DIR / f"str_conflict{bits}.json")
    p = fx["params"]
    base = run(fx["program"], fx["data"], fx["threads"], **p)
    changed = {a for a, v in enumerate(base.memory) if v != (fx["data"] + [0] * len(base.memory))[a]}
    r = run(fx["program"], fx["data"], fx["threads"], **dict(p, warps_per_core=4))
    assert changed - r.race_words, (sorted(changed), sorted(r.race_words))


# ---- L2: race_words over-approximates "words whose final value depends on a race" ----------------------------------

# One lane per warp. Warp 0 stores 1 then 2 to word 50; warp 1 stores 2. Each warp's STRs complete in program order,
# so word 50 ends 2 in every order, yet it is flagged and left out of the comparison.
SAME_FINAL = """
    CONST R1, #50
    CONST R2, #2
    CONST R0, #0
    CMP %threadIdx, R0
    BRz W0
    STR R1, R2
    RET
W0: CONST R3, #1
    STR R1, R3
    STR R1, R2
    RET
"""


@pytest.mark.xfail(strict=True, reason="AUDIT-21 L2: a word with one possible final value is flagged as raced")
@pytest.mark.parametrize("policy", ["lrr", "gto"])
def test_l2_deterministic_word_not_raced(policy):
    r = run(assemble(SAME_FINAL), [], 2, threads_per_block=2, warps_per_core=2, warp_policy=policy)
    assert r.memory[50] == 2
    assert 50 not in r.race_words, (r.store_races, sorted(r.race_words))


# ---- L3: the warps2lat exclusions -----------------------------------------------------------------------------------

def _make_var(name):
    mk = (ROOT / "Makefile").read_text()
    return re.search(rf"^{name}\s*:=\s*(.*)$", mk, re.M).group(1).split()


@pytest.mark.xfail(strict=True, reason="AUDIT-21 L3: test_xf_rmsnorm passes at warps2lat but is left out of it")
def test_l3_warps2lat_runs_rmsnorm():
    """SG28 left test_xf_rmsnorm out of warps2lat as failing; at 829b8a3 it passes there (2/2, about 53 s), and the
    kernel has no 4-thread assumption (control above)."""
    assert "test_xf_rmsnorm" in _make_var("XF_warps2lat")
