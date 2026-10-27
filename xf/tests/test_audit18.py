"""AUDIT-18 repros (encoding soundness, round 5). Strict xfails mark findings open at e9c9eef; the controls check the
machinery the repros rely on. RTL-side numbers come from build/a18 runs recorded in AUDIT-18.md (no simulator here).
"""
import re
from types import SimpleNamespace

import pytest
from cocotb.types import LogicArray

from test.helpers.memory import Memory
from xf import isasim
from xf.asm import assemble
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.tests.legacy import ROOT

# ---- H1: cross-warp STR order is not issue order under channel contention ---------------------------------------
# Block 0 (core 0, two 4-lane warps): warp w spins d[w] iterations, then every lane stores threadIdx + 1 to word 100
# (kind 0) or loads it (kind 1); after BAR every lane reads word 100 back to 140 + t. Block 1 (core 1) hogs the four
# data channels with 4-lane loads. With d = (2, 1) warp 1 issues its STR first, so issue order says warp 0's lane 3
# (value 4) wins. Build: THREADS_PER_BLOCK=8 WARPS_PER_CORE=2 ICACHE_LINES=16 PIPELINED=1, MEM_LATENCY=10.
H1_SRC = """
    CONST R1, #1
    CONST R8, #0
    CMP %blockIdx, R8
    BRz B0
    CONST R5, #2
    LDR R4, R5
HD: SUB R4, R4, R1
    CMP R4, R8
    BRn HD
    CONST R4, #30
H:  CONST R6, #32
    ADD R6, R6, %threadIdx
    LDR R7, R6
    SUB R4, R4, R1
    CMP R4, R8
    BRn H
    RET
B0: CONST R2, #4
    DIV R3, %threadIdx, R2
    LDR R4, R3
    CONST R5, #4
    ADD R5, R5, R3
    LDR R5, R5
D:  SUB R4, R4, R1
    CMP R4, R8
    BRn D
    CONST R9, #100
    ADD R11, %threadIdx, R1
    CMP R5, R8
    BRz S
    LDR R12, R9
    CONST R6, #120
    ADD R6, R6, %threadIdx
    STR R6, R12
    BRnzp J
S:  STR R9, R11
J:  BAR
    LDR R7, R9
    CONST R6, #140
    ADD R6, R6, %threadIdx
    STR R6, R7
    RET
"""
H1_DATA = [2, 1, 1] + [0] * 29 + list(range(32, 64)) + [0] * 36 + [77, 0, 0, 0]
H1_THREADS = 16
H1_KW = dict(threads_per_block=8, warps_per_core=2)
# Warp issue order per block recorded by test.helpers.runner.run_kernel(trace=...) on that build.
H1_TRACE = {0: [int(c) for c in "010101010101010010110010101010101010100010100101101"],
            1: [int(c) for c in "0101010101010001010101011100101010101100000011010101010011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010100011101010101111"]}
H1_RTL_WORD100 = 8   # RTL data memory word 100 (and 140..147) after the run


def _h1_replay():
    return isasim.run_trace(H1_TRACE, assemble(H1_SRC), H1_DATA, H1_THREADS, **H1_KW)


def test_h1_recorded_order_is_legal_and_issue_order_picks_warp0():
    """Control: the RTL's recorded order is a legal schedule (run_trace accepts it) with a cross-warp STR conflict, and
    in that order warp 1's STR issues first, so the model stores warp 0's lane 3 (4)."""
    r = _h1_replay()
    assert r.stores_conflict and r.memory[100] == 4 and r.memory[140:148] == [4] * 8


@pytest.mark.xfail(strict=True, reason="AUDIT-18 H1: RTL applies warp 1's later-claimed STR last (8), model 4")
def test_h1_model_matches_rtl_on_recorded_order():
    assert _h1_replay().memory[100] == H1_RTL_WORD100


@pytest.mark.xfail(strict=True, reason="AUDIT-18 H1: docs/isa.md states no cross-warp STR rule")
def test_h1_isa_md_states_cross_warp_str_rule():
    text = (ROOT / "docs" / "isa.md").read_text()
    para = text[text.index("When several lanes of one `STR`"):]
    para = para[:para.index("\n\n")]
    assert "warp" in para, para


# ---- H2: no matrix configuration runs several warps with memory latency or an I-cache ---------------------------

def _matrix():
    mk = (ROOT / "Makefile").read_text()
    names = re.search(r"^MATRIX := (.*)$", mk, re.M).group(1).split()
    defs = dict(re.findall(r"^(CFG_\w+|ENV_\w+)\s*:=\s*(.*)$", mk, re.M))

    def expand(v):
        return re.sub(r"\$\((CFG_\w+)\)", lambda m: expand(defs.get(m.group(1), "")), v)
    return {n: (expand(defs.get(f"CFG_{n}", "")), defs.get(f"ENV_{n}", "")) for n in names}


def test_matrix_parse_control():
    m = _matrix()
    assert "-Pgpu.WARPS_PER_CORE=2" in m["warps2"][0] and m["latency"][1] == "MEM_LATENCY=3"
    assert "-Pgpu.ICACHE_LINES=16" in m["all"][0]


@pytest.mark.xfail(strict=True, reason="AUDIT-18 H2: H1 needs WARPS_PER_CORE > 1 with latency/I-cache; no matrix "
                                       "config has both, so make matrix passes the H1 RTL")
def test_h2_matrix_has_multiwarp_with_latency_or_icache():
    assert any("WARPS_PER_CORE" in cfg and ("MEM_LATENCY" in env or "ICACHE_LINES" in cfg)
               for cfg, env in _matrix().values())


# ---- M1: the harness memory model's other ordering assumption -----------------------------------------------------

class _Sig:
    value = 0


def _mem(n=2, width=8):
    dut = SimpleNamespace(**{f"data_mem_{k}": _Sig() for k in (
        "read_valid", "read_address", "read_ready", "read_data",
        "write_valid", "write_address", "write_data", "write_ready")})
    return dut, Memory(dut, width, width, n, "data")


def test_memory_model_same_cycle_read_sees_old_value():
    """Control for M1: a read on channel 0 and a write on channel 1 to one word in the same cycle: the read returns the
    old word. Nothing in ADR-016 or docs/isa.md requires an external memory to do this."""
    dut, m = _mem()
    m.memory[7] = 5
    dut.data_mem_read_valid.value = LogicArray.from_unsigned(0b10, 2)        # channel 0 (MSB first)
    dut.data_mem_read_address.value = LogicArray.from_unsigned(7 << 8, 16)
    dut.data_mem_write_valid.value = LogicArray.from_unsigned(0b01, 2)       # channel 1
    dut.data_mem_write_address.value = LogicArray.from_unsigned(7, 16)
    dut.data_mem_write_data.value = LogicArray.from_unsigned(9, 16)
    m.run()
    assert m.memory[7] == 9 and (dut.data_mem_read_data.value >> 8) & 0xFF == 5


@pytest.mark.xfail(strict=True, reason="AUDIT-18 M1: ADR-016 does not state the read-before-write and in-order "
                                       "completion assumptions the STR rule and cross-warp order rest on")
def test_m1_adr016_states_memory_ordering_assumptions():
    adr = (ROOT / "docs" / "decisions" / "ADR-016-str-conflicts.md").read_text().lower()
    assert re.search(r"read[^.]*before[^.]*write|write[^.]*after[^.]*read", adr) and "latency" in adr


# ---- controls: replay soundness -----------------------------------------------------------------------------------

def test_run_trace_rejects_skipped_warp_and_post_ret_issue():
    """A trace that never issues one warp, or issues a warp after its RET, is an error."""
    fx = load(FIXTURE_DIR / "str_conflict8.json")
    args, kw = (fx["program"], fx["data"], fx["threads"]), dict(fx["params"], warps_per_core=2)
    log = {}
    isasim.run(*args, **kw, warp_policy=lambda b, ready: log.setdefault(b, []).append(ready[0]) or ready[0])
    assert isasim.run_trace(log, *args, **kw).memory == isasim.run(*args, **kw, warp_policy="gto").memory
    for bad in ({**log, 0: [0] * len(log[0])}, {**log, 0: log[0] + [log[0][-1]]}, {1: log[1]}):
        with pytest.raises(isasim.IsaSimError):
            isasim.run_trace(bad, *args, **kw)
