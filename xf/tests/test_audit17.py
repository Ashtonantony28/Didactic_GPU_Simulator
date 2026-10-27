"""AUDIT-17 repros (test coverage, mutation, round 5). Each xfail is strict at e9c9eef and passes once the gap is
closed; the controls pass and show that the helpers below see what the xfails look for.

The check_counters repros run every xf/tests function that calls runner.check_counters against a one-line mutant
of it (an assertion deleted): an xfail asserts that some existing test then fails. The model repros rerun every ISA
fixture on a one-line mutant of xf.isasim, as test_audit13 does."""
import inspect
import types
from pathlib import Path

import pytest

from test.helpers import runner
from xf import isasim
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.tests.legacy import ROOT

HERE = Path(__file__).name


def _mutant(mod, old, new, suffix="_audit17"):
    """`mod` with one source edit, as a fresh module."""
    src = Path(mod.__file__).read_text()
    assert src.count(old) == 1, old
    m = types.ModuleType(mod.__name__ + suffix)
    m.__package__, m.__file__ = mod.__package__, mod.__file__
    exec(compile(src.replace(old, new), mod.__file__, "exec"), m.__dict__)
    return m


# ---- L3: runner.check_counters' assertions are not themselves tested --------------------------------------------

CHECKS = {
    "requests": '    assert metrics["data_mem_requests"] == reads + writes, \\',
    "icache_per_core_sum": "        assert metrics[key] == sum(per_core), f",
    "fetches_equal_issues": "    if _safe_int(dut.WARPS_PER_CORE.value) == 1:",
    "no_icache_no_hits": '    assert _safe_int(dut.ICACHE_LINES.value) or not metrics["icache_hits"]',
    "coalesced": '    assert metrics["data_mem_requests_coalesced"] == consumer_reads - reads, \\',
}


def _weakened(check):
    old = CHECKS[check]
    new = old.replace("== 1:", "== 0:") if check == "fetches_equal_issues" else old.replace("assert ", "assert True or ", 1)
    return _mutant(runner, old, new).check_counters


def _callers():
    """xf/tests functions (not this file's) whose source calls check_counters."""
    import importlib
    out = []
    for p in sorted((ROOT / "xf" / "tests").glob("test_*.py")):
        if p.name == HERE or "check_counters(" not in p.read_text():
            continue
        mod = importlib.import_module(f"xf.tests.{p.stem}")
        for name, fn in sorted(vars(mod).items()):
            if (name.startswith("test_") and inspect.isfunction(fn) and not inspect.signature(fn).parameters
                    and "check_counters(" in inspect.getsource(fn)
                    and not any(m.name in ("xfail", "skip") for m in getattr(fn, "pytestmark", []))):
                out.append(fn)
    return out


def _killers(check, monkeypatch):
    monkeypatch.setattr(runner, "check_counters", _weakened(check))
    killed = []
    for fn in _callers():
        try:
            fn()
        except BaseException:   # noqa: B036 - any failure kills the mutant
            killed.append(fn.__name__)
    monkeypatch.undo()
    return killed


@pytest.mark.xfail(strict=True, reason="AUDIT-17 L3: only check_counters' coalesced assertion has a test that fails "
                                       "when it is deleted (CK1-CK4)")
@pytest.mark.parametrize("check", ["requests", "icache_per_core_sum", "fetches_equal_issues", "no_icache_no_hits"])
def test_check_counters_assertion_is_tested(check, monkeypatch):
    assert _killers(check, monkeypatch)


def test_control_coalesced_check_is_tested(monkeypatch):
    """Control: deleting the coalesced assertion is caught (test_audit12's control), so _killers sees a kill."""
    assert _killers("coalesced", monkeypatch)


# ---- L1: the reset value of GPRs and NZP is unpinned --------------------------------------------------------------

def _fixtures_that_differ(mutant):
    """ISA fixtures whose memory differs between xf.isasim and `mutant` (W = 1, as test_isa_diff runs them)."""
    out = []
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = load(p)
        a = (fx["program"], fx["data"], fx["threads"])
        try:
            same = isasim.run(*a, **fx["params"]).memory == mutant.run(*a, **fx["params"]).memory
        except mutant.IsaSimError:   # e.g. a loop bound on %threadIdx no longer terminates
            same = False
        if not same:
            out.append(p.stem)
    return out


REGS0 = "regs = [[0] * 13 + ["
NZP0 = "nzp, smem = [0] * tpb, [0] * smem_size"


@pytest.mark.xfail(strict=True, reason="AUDIT-17 L1: no ISA fixture reads a GPR before writing it, so the RTL's "
                                       "per-block register reset is unpinned (N11 survives)")
def test_isa_fixtures_read_an_unwritten_register():
    assert _fixtures_that_differ(_mutant(isasim, REGS0, "regs = [[0x5A] * 13 + ["))


@pytest.mark.xfail(strict=True, reason="AUDIT-17 L1: no ISA fixture branches before its first CMP, so the RTL's "
                                       "NZP reset is unpinned (N02 survives)")
def test_isa_fixtures_branch_before_cmp():
    assert _fixtures_that_differ(_mutant(isasim, NZP0, "nzp, smem = [0b010] * tpb, [0] * smem_size"))


def test_control_fixture_scan_sees_special_register_init():
    """Control: a model that starts %threadIdx at the wrong value changes fixtures, so the scan works."""
    m = _mutant(isasim, REGS0 + "b & wmask, tpb & wmask, t & wmask]", REGS0 + "b & wmask, tpb & wmask, (t + 1) & wmask]")
    assert _fixtures_that_differ(m)


# ---- L2: gpu.sv's parameter guards are untested -------------------------------------------------------------------

GUARDS = {"WORD_BITS": "!= DATA_MEM_DATA_BITS", "WARP_POLICY": "is not 0, 1 or 2",
          "WARPS_PER_CORE": "does not divide THREADS_PER_BLOCK"}


def _test_sources():
    files = [*(ROOT / "test").glob("*.py"), *(ROOT / "xf" / "tests").glob("*.py"), ROOT / "Makefile"]
    return {p.name: p.read_text() for p in files if p.name != HERE}


def test_control_guards_exist_in_gpu_sv():
    src = (ROOT / "src" / "gpu.sv").read_text()
    assert all(msg in src for msg in GUARDS.values())


@pytest.mark.xfail(strict=True, reason="AUDIT-17 L2: no test builds a gpu that trips its $fatal parameter guards "
                                       "(N09, N10, N16 survive)")
@pytest.mark.parametrize("param", sorted(GUARDS))
def test_gpu_parameter_guard_is_tested(param):
    assert any(GUARDS[param] in s for s in _test_sources().values())
