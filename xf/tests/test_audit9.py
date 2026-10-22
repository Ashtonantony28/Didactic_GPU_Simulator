"""AUDIT-9 repros (encoding soundness). Each is a strict xfail at aa27256: it passes once the gap is closed."""
import pytest

from test import simt_fatal, test_divergence as dv
from xf.asm import assemble
from xf.isasim import IsaSimError, run

OP_BAR = 0b1100


def _depth_needed(prog, threads, tpb):
    for d in range(2 * tpb):
        try:
            run(prog, [0] * 8, threads, threads_per_block=tpb, simt=True, stack_depth=d)
            return d
        except IsaSimError as e:
            if "deeper" not in str(e):
                raise
    raise AssertionError("no depth runs the program")


@pytest.mark.xfail(strict=True, reason="AUDIT-9 L1: overflow $fatal only checked at 10 entries, not at DEPTH + 1 = 9")
def test_overflow_fatal_covers_depth_plus_one():
    """S03 (`sp + 2 > DEPTH + 1`) survives make test_divergence: no fatal run needs exactly 9 entries."""
    depths = {}
    for k, (src, n) in simt_fatal.PROGRAMS.items():
        try:
            depths[k] = _depth_needed(assemble(src), n, 8)
        except IsaSimError:   # a nosync run
            pass
    assert 9 in depths.values(), depths


@pytest.mark.xfail(strict=True, reason="AUDIT-9 M2: no fatal run has a divergent branch after its SYNC was consumed")
def test_nosync_fatal_covers_a_consumed_sync():
    """S02 (SYNC never consumed) survives: the only nosync run has no SYNC at all."""
    hits = []
    for k, (src, n) in simt_fatal.PROGRAMS.items():
        prog = assemble(src)
        if any(w >> 12 == 1 and (w >> 9) & 7 == 0 for w in prog):
            try:
                run(prog, [0] * 8, n, threads_per_block=8 if n > 4 else 4, simt=True)
            except IsaSimError as e:
                if "without SYNC" in str(e):
                    hits.append(k)
    assert hits, "every nosync program lacks a SYNC entirely"


@pytest.mark.xfail(strict=True, reason="AUDIT-9 M1: no SIMT_STACK=1 RTL test executes BAR")
def test_simt_rtl_tests_execute_bar():
    """C08 (BAR dropped when SIMT_STACK=1) survives: test_divergence never runs a BAR, test_multiwarp runs stack-off."""
    from xf.tests import test_isasim as cases
    srcs = [cases.M1.format(sync="SYNC done"), cases.NESTED, cases.LOOP, dv.RET_IN_REGION, dv.CAPPED_LOOP.format(cap=4)]
    progs = [assemble(s) for s in srcs] + [dv.Structured(seed).program() for seed in range(16)]
    assert any(w >> 12 == OP_BAR for p in progs for w in p)


@pytest.mark.xfail(strict=True, reason="AUDIT-9 L5: xf.isasim accepts launches the 8-bit DCR cannot hold")
def test_isasim_rejects_more_than_255_threads():
    prog = assemble("STR %threadIdx, %threadIdx\nRET")
    with pytest.raises(ValueError):
        run(prog, [], 256)
