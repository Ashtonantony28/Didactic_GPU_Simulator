"""AUDIT-14 repros (encoding soundness, round 4). Each xfail is strict at 4444c25 and passes once the finding is
fixed; the controls pass and check the machinery the xfails rely on.

`_traced` is xf.isasim.run with its warp pick replaced by a callback (one source edit), so a test can replay any
legal issue order: each pick must name a warp that is neither done nor held at a BAR.
"""
import inspect
import random
import re
import types

import pytest

from xf import isasim
from xf.asm import assemble
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.tests.legacy import ROOT

_PICK = '''            order = range(cur, cur + warps_per_core) if warp_policy == "lrr" else [cur, *range(warps_per_core)]
            cur = next(i % warps_per_core for i in order if not warps[i % warps_per_core].done
                       and not warps[i % warps_per_core].bar)'''


def _traced():
    src = inspect.getsource(isasim)
    assert src.count(_PICK) == 1, "xf.isasim's warp pick changed; update _PICK"
    src = src.replace(_PICK, "            cur = _pick(b, warps, cur)")
    src = src.replace("def run(program, data, threads, *,", "def run_traced(_pick, program, data, threads, *,", 1)
    m = types.ModuleType("isasim_traced")
    exec(compile(src, "isasim_traced", "exec"), m.__dict__)
    return m.run_traced


def _random_pick(seed):
    rng = random.Random(seed)

    def pick(b, warps, cur):
        return rng.choice([i for i, w in enumerate(warps) if not w.done and not w.bar])
    return pick


def _lrr_pick(warps_per_core):
    nxt = {}

    def pick(b, warps, cur):
        c = nxt.get(b, 0)
        for i in range(c, c + warps_per_core):
            w = i % warps_per_core
            if not warps[w].done and not warps[w].bar:
                nxt[b] = (w + 1) % warps_per_core
                return w
    return pick


# ---- control: the replay machinery reproduces xf.isasim ---------------------------------------------------------

@pytest.mark.parametrize("warps", [2, 4])
def test_traced_lrr_replay_matches_isasim(warps):
    """Replaying round-robin through the patched model gives xf.isasim's lrr result on every 8-bit fixture."""
    run_traced = _traced()
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = load(p)
        if fx["params"]["word_bits"] != 8:
            continue
        args, kw = (fx["program"], fx["data"], fx["threads"]), dict(fx["params"], warps_per_core=warps)
        a = isasim.run(*args, **kw)
        b = run_traced(_lrr_pick(warps), *args, **kw)
        assert (a.memory, a.steps) == (b.memory, b.steps), p.stem


def test_sts_conflict_fixture_pins_lane_order_at_one_warp():
    """Control for L1: at WARPS_PER_CORE=1 every legal schedule (there is one) gives the fixture's memory."""
    run_traced = _traced()
    for w in (8, 32):
        fx = load(FIXTURE_DIR / f"sts_conflict{w}.json")
        got = {tuple(run_traced(_random_pick(s), fx["program"], fx["data"], fx["threads"], **fx["params"]).memory[:8])
               for s in range(20)}
        assert got == {tuple(fx["memory"][:8])}, got


# ---- L1: the sts_conflict fixture is interleaving-dependent at W > 1 --------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-14 L1: sts_conflict's result depends on the warp issue order at W > 1")
def test_sts_conflict_fixture_is_interleaving_independent():
    """xf/isa_fixtures.py says the BAR makes STS_CONFLICT's result independent of the warp interleaving, and
    docs/isa.md says the fixtures pin "highest lane wins". At W > 1 the BAR comes after the conflicting STS, so the
    winner is whichever warp issues its STS last (ADR-007): other legal schedules store 1, 2 (W=4) or 0."""
    run_traced = _traced()
    fx = load(FIXTURE_DIR / "sts_conflict8.json")
    for warps in (2, 4):
        got = {tuple(run_traced(_random_pick(s), fx["program"], fx["data"], fx["threads"], **fx["params"],
                                warps_per_core=warps).memory[:8]) for s in range(200)}
        assert got == {tuple(fx["memory"][:8])}, (warps, sorted(got))


# ---- L2: docs/isa.md RET row ------------------------------------------------------------------------------------

def test_ret_ends_only_the_issuing_warp():
    """Control for L2: at W = 2 without the stack, warp 0's RET does not stop warp 1 (model; the RTL agrees,
    AUDIT-14 fuzz)."""
    prog = assemble("""
        CONST R0, #2
        DIV R1, %threadIdx, R0
        CONST R2, #0
        CMP R1, R2
        BRn rest            ; warp 1 (threads 2, 3) branches, warp 0 returns
        RET
    rest:
        STR %threadIdx, R0
        RET""")
    assert isasim.run(prog, [], 4, warps_per_core=2).memory[:4] == [0, 0, 2, 2]


@pytest.mark.xfail(strict=True, reason="AUDIT-14 L2: docs/isa.md says RET finishes the block; at W > 1 it ends the warp")
def test_isa_md_ret_row_names_the_warp():
    row = next(l for l in (ROOT / "docs" / "isa.md").read_text().splitlines() if re.match(r"\|\s*`1111`", l))
    assert "warp" in row, row


# ---- H1: STR lane conflicts: the model's documented winner is not the RTL's ---------------------------------------

STR_CONFLICT = """
    CONST R0, #100
    CONST R1, #1
    ADD R2, %threadIdx, R1
    STR R0, R2              ; every lane of the block: MEM[100] = t + 1
    RET"""


def test_store_conflict_is_flagged():
    """Control for H1: the model flags the conflict (fixtures skip such programs) and stores its last lane."""
    r = isasim.run(assemble(STR_CONFLICT), [], 4)
    assert r.stores_conflict == [(0, 3, 100)] and r.memory[100] == 4


@pytest.mark.xfail(strict=True, reason="AUDIT-14 H1: STR conflict: model stores the highest lane, the RTL the lowest")
def test_store_conflict_model_matches_rtl():
    """On the RTL (default build, AUDIT-14.md H1 command) MEM[100] ends as 1, lane 0's value: the four lanes reach the
    four data channels in the same cycle and the memory applies channel 0 last. In general the lowest lane of the last
    group of DATA_MEM_NUM_CHANNELS lanes wins (lane 4 of 8, lane 12 of 16), at W = 1/2/4, MEM_LATENCY 0/3/10 and
    COALESCE=1. xf.isasim stores 4 (its docstring and test_isasim::test_store_conflict: "the last lane wins").
    Passes once the model agrees with the RTL or refuses the program."""
    try:
        r = isasim.run(assemble(STR_CONFLICT), [], 4)
    except (isasim.IsaSimError, ValueError):
        return
    assert r.memory[100] == 1, r.memory[100]
