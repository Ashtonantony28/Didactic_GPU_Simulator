"""AUDIT-14 repros (encoding soundness, round 4). H1, L1 and L2 were strict xfails at 4444c25 and pass since SG25;
the controls check the machinery they rely on.

A callable `warp_policy` runs xf.isasim under any legal issue order: each pick names a ready warp (neither done nor
held at a BAR).
"""
import random
import re
from types import SimpleNamespace

import pytest
from cocotb.types import LogicArray

from test.helpers.memory import Memory
from xf import isasim
from xf.asm import assemble
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.tests.legacy import ROOT


def _random_pick(seed, log=None):
    rng = random.Random(seed)

    def pick(b, ready):
        w = rng.choice(ready)
        if log is not None:
            log.setdefault(b, []).append(w)
        return w
    return pick


def _lrr_pick(warps_per_core):
    nxt = {}

    def pick(b, ready):
        c = nxt.get(b, 0)
        w = next(i % warps_per_core for i in range(c, c + warps_per_core) if i % warps_per_core in ready)
        nxt[b] = (w + 1) % warps_per_core
        return w
    return pick


# ---- control: a callable warp_policy reproduces xf.isasim -------------------------------------------------------

@pytest.mark.parametrize("warps", [2, 4])
def test_traced_lrr_replay_matches_isasim(warps):
    """Round-robin through a callable warp_policy gives xf.isasim's lrr result on every 8-bit fixture."""
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = load(p)
        if fx["params"]["word_bits"] != 8:
            continue
        args, kw = (fx["program"], fx["data"], fx["threads"]), dict(fx["params"], warps_per_core=warps)
        a = isasim.run(*args, **kw)
        b = isasim.run(*args, **kw, warp_policy=_lrr_pick(warps))
        assert (a.memory, a.steps) == (b.memory, b.steps), p.stem


def test_sts_conflict_fixture_pins_lane_order_at_one_warp():
    """Control for L1: at WARPS_PER_CORE=1 every legal schedule (there is one) gives the fixture's memory."""
    for w in (8, 32):
        fx = load(FIXTURE_DIR / f"sts_conflict{w}.json")
        got = {tuple(isasim.run(fx["program"], fx["data"], fx["threads"], **fx["params"],
                                warp_policy=_random_pick(s)).memory[:8]) for s in range(20)}
        assert got == {tuple(fx["memory"][:8])}, got


# ---- L1: interleaving-dependent fixtures are compared against a replay of the RTL's issue order -----------------

def test_sts_conflict_fixture_is_interleaving_dependent():
    """At W > 1 the BAR comes after the conflicting STS, so the winner is whichever warp issues its STS last (ADR-007);
    lrr == gto does not show it, which is why that gate was unsound."""
    fx = load(FIXTURE_DIR / "sts_conflict8.json")
    args = (fx["program"], fx["data"], fx["threads"])
    for warps in (2, 4):
        kw = dict(fx["params"], warps_per_core=warps)
        assert isasim.run(*args, **kw).memory == isasim.run(*args, **kw, warp_policy="gto").memory
        got = {tuple(isasim.run(*args, **kw, warp_policy=_random_pick(s)).memory[:8]) for s in range(200)}
        assert len(got) > 1, (warps, got)


@pytest.mark.parametrize("name", ["sts_conflict8", "str_conflict8", "str_conflict32"])
def test_replay_reproduces_any_legal_order(name):
    """Replaying the order a random legal schedule took (run_trace) reproduces its memory and steps, so test_isa_diff
    and test_divergence, which replay the RTL's recorded order at W > 1, compare interleaving-dependent programs
    exactly."""
    fx = load(FIXTURE_DIR / f"{name}.json")
    args = (fx["program"], fx["data"], fx["threads"])
    for warps in (2, 4):
        kw = dict(fx["params"], warps_per_core=warps)
        for s in range(20):
            log = {}
            r = isasim.run(*args, **kw, warp_policy=_random_pick(s, log))
            t = isasim.run_trace(log, *args, **kw)
            assert (t.memory, t.steps) == (r.memory, r.steps), (name, warps, s)


def test_run_trace_rejects_illegal_orders():
    """A trace that is short, too long, or picks a warp held at a BAR is an error, not a silent pass."""
    fx = load(FIXTURE_DIR / "sts_conflict8.json")
    args, kw = (fx["program"], fx["data"], fx["threads"]), dict(fx["params"], warps_per_core=2)
    log = {}
    isasim.run(*args, **kw, warp_policy=_random_pick(0, log))
    for bad in ({**log, 0: log[0][:-1]}, {**log, 0: log[0] + [0]}, {**log, 0: [0] * len(log[0])}):
        with pytest.raises(isasim.IsaSimError):
            isasim.run_trace(bad, *args, **kw)


def test_differential_harnesses_replay_the_rtl_order():
    """The lrr == gto gate is gone from both differential harnesses."""
    for f in ("test_isa_diff.py", "test_divergence.py"):
        src = (ROOT / "test" / f).read_text()
        assert "run_trace" in src and '"gto"' not in src, f


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


def test_isa_md_ret_row_names_the_warp():
    row = next(l for l in (ROOT / "docs" / "isa.md").read_text().splitlines() if re.match(r"\|\s*`1111`", l))
    assert "warp" in row, row


# ---- H1: STR lane conflicts: highest lane wins in the model and the RTL -----------------------------------------

STR_CONFLICT = """
    CONST R0, #100
    CONST R1, #1
    ADD R2, %threadIdx, R1
    STR R0, R2              ; every lane of the block: MEM[100] = t + 1
    RET"""


def test_store_conflict_is_flagged():
    """Control for H1: the model flags the conflict and stores its highest lane."""
    r = isasim.run(assemble(STR_CONFLICT), [], 4)
    assert r.stores_conflict == [(0, 3, 100)] and r.memory[100] == 4


def test_store_conflict_model_matches_rtl():
    """At 4444c25 the RTL stored lane 0 here (the lowest lane of the last DATA_MEM_NUM_CHANNELS group): the memory
    model applied same-cycle channel writes from the top channel down. Since SG25 it applies them from channel 0 up and
    writes skip MeDiC's priority pass, so one STR's lanes reach the channels in lane order and the highest lane wins
    (ADR-016), as in xf.isasim. test_isa_diff runs str_conflict{8,32} on the RTL in every matrix configuration and
    test_divergence runs xf.isa_fixtures.STR_CONFLICT at 8 lanes."""
    assert isasim.run(assemble(STR_CONFLICT), [], 4).memory[100] == 4
    for w in (8, 32):
        fx = load(FIXTURE_DIR / f"str_conflict{w}.json")
        assert isasim.run(fx["program"], fx["data"], fx["threads"], **fx["params"]).stores_conflict


class _Sig:
    value = 0


def test_memory_model_highest_channel_wins():
    """Channels 0 and 1 write one address in the same cycle: channel 1 (the higher lane) is applied last."""
    width, n = 8, 2
    dut = SimpleNamespace(**{f"data_mem_{k}": _Sig() for k in (
        "read_valid", "read_address", "read_ready", "read_data",
        "write_valid", "write_address", "write_data", "write_ready")})
    m = Memory(dut, width, width, n, "data")
    dut.data_mem_read_valid.value = LogicArray.from_unsigned(0, n)
    dut.data_mem_read_address.value = LogicArray.from_unsigned(0, n * width)
    dut.data_mem_write_valid.value = LogicArray.from_unsigned(0b11, n)
    dut.data_mem_write_address.value = LogicArray.from_unsigned(7 << width | 7, n * width)
    dut.data_mem_write_data.value = LogicArray.from_unsigned(22 << width | 11, n * width)   # channel 1 writes 22
    m.run()
    assert m.memory[7] == 22
