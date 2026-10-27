"""AUDIT-13 repros (test coverage, round 4). Each xfail is strict at 4444c25 and passes once the gap is closed; the
controls pass and show that the helpers below see what the xfails look for.

The binder repros run the repo's own binder tests in-process against a one-line mutant of xf.kernels or
xf.elementwise: an xfail asserts that some existing test kills the mutant, the control does the same for a mutant
the tests do kill."""
import inspect
import itertools
import re
import types

import pytest

from xf import elementwise, isasim, kernels
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.tests import test_audit4, test_audit8, test_audit11, test_elementwise, test_kernels
from xf.tests.legacy import ROOT

BINDER_TESTS = (test_elementwise, test_kernels, test_audit4, test_audit8, test_audit11)


def _mutant(mod, old, new):
    """`mod` with one source edit, as a fresh module."""
    src = open(mod.__file__).read()
    assert src.count(old) == 1, old
    m = types.ModuleType(mod.__name__ + "_audit13")
    m.__package__, m.__file__ = mod.__package__, mod.__file__
    exec(compile(src.replace(old, new), mod.__file__, "exec"), m.__dict__)
    return m


def _cases(fn):
    """Every parametrize combination of a test function, or None if it needs a fixture."""
    marks = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
    grids = []
    for m in marks:
        names = [n.strip() for n in (m.args[0].split(",") if isinstance(m.args[0], str) else m.args[0])]
        rows = [v.values if hasattr(v, "values") else v for v in m.args[1]]
        grids.append([dict(zip(names, r if len(names) > 1 else (r,))) for r in rows])
    params = set(inspect.signature(fn).parameters)
    combos = [dict(kv for d in c for kv in d.items()) for c in itertools.product(*grids)] if grids else [{}]
    if any(set(c) != params for c in combos):
        return None
    return combos


def _killed(orig, mutant, monkeypatch):
    """(test, case) of the binder tests that fail with `mutant` in place of the module `orig`."""
    killed = []
    for tm in BINDER_TESTS:
        for alias in [k for k, v in vars(tm).items() if v is orig]:
            monkeypatch.setattr(tm, alias, mutant)
        for name, fn in sorted(vars(tm).items()):
            if not (name.startswith("test_") and inspect.isfunction(fn)):
                continue
            if any(m.name in ("xfail", "skip") for m in getattr(fn, "pytestmark", [])):
                continue
            for case in _cases(fn) or []:
                try:
                    fn(**case)
                except BaseException:   # noqa: B036 - any failure (incl. pytest.fail) kills the mutant
                    killed.append((f"{tm.__name__}.{name}", case))
    monkeypatch.undo()
    return killed


# ---- H1: SMEM address wrap -----------------------------------------------------------------------------------------

STS_LOOP = """                for a, v in [(regs[t][rs] & 0xFF & (smem_size - 1), regs[t][rt]) for t in lanes]:
                    smem[a] = v"""


def _fixtures_that_differ(mutant):
    """ISA fixtures whose memory differs between xf.isasim and `mutant` (W = 1, as test_isa_diff runs them)."""
    out = []
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = load(p)
        a = (fx["program"], fx["data"], fx["threads"])
        if isasim.run(*a, **fx["params"]).memory != mutant.run(*a, **fx["params"]).memory:
            out.append(p.stem)
    return out


def test_isa_fixtures_pin_smem_wrap():
    """SM1 (shared_memory.sv writes `mem[lane_write_addr[i]]` unmasked, so STS to 64..255 is dropped) survived every
    target at default parameters at 4444c25: no fixture had an LDS or STS address >= 64. The smem_wrap fixtures (SG26)
    do, at both widths, so a model that drops such stores no longer reproduces them."""
    drop = _mutant(isasim, STS_LOOP, STS_LOOP.replace("& 0xFF & (smem_size - 1), regs", ", regs")
                   .replace("smem[a] = v", "if a < smem_size:\n                        smem[a] = v"))
    assert {"smem_wrap8", "smem_wrap32"} <= set(_fixtures_that_differ(drop))


# ---- H2: the top of the 8-bit thread range ----------------------------------------------------------------------

def test_a_fixture_launches_253_threads_or_more():
    """B02 (dispatch.sv rounds `thread_count + THREADS_PER_BLOCK - 1` in 8 bits) survived every default-config target at
    4444c25: 253-255 threads give 0 blocks and the launch ends in 2 cycles with nothing written. The binders accept up
    to 255 threads; the launch fixtures (SG26) launch 253 and 255 at both widths."""
    launched = {(load(p)["params"]["word_bits"], load(p)["threads"]) for p in FIXTURE_DIR.glob("*.json")}
    assert {(w, n) for w in (8, 32) for n in (253, 255)} <= launched


# ---- M1: no test pins a cycle count -----------------------------------------------------------------------------

CYCLES_EXACT = re.compile(r"""\[["']cycles["']\]\s*(==|<=|<)|cycles\s*==\s*\d""")


def _test_sources():
    files = list((ROOT / "test").glob("*.py")) + list((ROOT / "test" / "helpers").glob("*.py"))
    files += [p for p in (ROOT / "xf" / "tests").glob("*.py") if p.name != "test_audit13.py"]
    return {p.name: p.read_text() for p in files}


@pytest.mark.xfail(strict=True, reason="AUDIT-13 M1: no test bounds a kernel's cycle count; U02/BC5 survive")
def test_some_test_bounds_cycles():
    """U02 (the LSU finishes a store without waiting for mem_write_ready) survives every matrix configuration: data is
    unchanged there and only the cycle counts move (raw probe 263 -> 251 cycles at default). Nothing compares
    metrics['cycles'] with an expected value or bound."""
    assert [n for n, s in _test_sources().items() if CYCLES_EXACT.search(s)]


# ---- L1/L2: binder refusals --------------------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-13 L1: elementwise binders' 255-thread refusal is unpinned (Q03)")
def test_elementwise_thread_limit_pinned(monkeypatch):
    """Q03 (`threads > 256`): a 256-element relu binds, the 8-bit DCR sees 0 threads and nothing is written."""
    m = _mutant(elementwise, "    if threads > 255:", "    if threads > 256:")
    assert _killed(elementwise, m, monkeypatch)


@pytest.mark.xfail(strict=True, reason="AUDIT-13 L2: matmul_tiled's N % 4 refusal is pinned only for N > 4 (Q13)")
def test_tiled_small_n_refusal_pinned(monkeypatch):
    """Q13 (accept N % 4 != 0 when N < 4): (4x3)@(3x2) binds 2 threads and the model's result differs from golden."""
    m = _mutant(kernels, "        if n % TILE:", "        if n % TILE and n > TILE:")
    assert _killed(kernels, m, monkeypatch)


# ---- L3: isa_diff may skip fixtures without limit at W > 1 --------------------------------------------------------

def _skipped(mod, warps):
    """Fixtures test_isa_diff would skip at this warps count with model `mod` (lrr and gto disagree)."""
    out = []
    for p in sorted(FIXTURE_DIR.glob("*.json")):
        fx = load(p)
        a = (fx["program"], fx["data"], fx["threads"])
        r = mod.run(*a, **fx["params"], warps_per_core=warps)
        g = mod.run(*a, **fx["params"], warps_per_core=warps, warp_policy="gto")
        if (r.memory, r.steps) != (g.memory, g.steps):
            out.append(p.stem)
    return out


@pytest.mark.xfail(strict=True, reason="AUDIT-13 L3: isa_diff_skipped bounds the skipped set only at one warp")
def test_isa_diff_skip_set_bounded_at_several_warps():
    """At 4444c25 no fixture is skipped at W = 2 or 4, but test_isa_diff only asserts `warps > 1 or not skipped`, so a
    model or fixture change that makes results interleaving-dependent silently drops them from the RTL comparison."""
    src = (ROOT / "test" / "test_isa_diff.py").read_text()
    assert "assert warps > 1 or not skipped" not in src and re.search(r"assert not skipped|skipped == \[\]", src)


# ---- controls ----------------------------------------------------------------------------------------------------

def test_control_smem_wrap_mutant_is_live():
    """The H1 model mutant does change a program that stores above SMEM word 63 (STS 64 + t, LDS t)."""
    from xf.asm import assemble
    drop = _mutant(isasim, STS_LOOP, STS_LOOP.replace("& 0xFF & (smem_size - 1), regs", ", regs")
                   .replace("smem[a] = v", "if a < smem_size:\n                        smem[a] = v"))
    prog = assemble("CONST R1, #64\nADD R1, R1, %threadIdx\nSTS R1, %threadIdx\nLDS R2, %threadIdx\n"
                    "CONST R3, #8\nADD R3, R3, %threadIdx\nSTR R3, R2\nRET")
    data = [9] * 16
    assert isasim.run(prog, data, 4).memory[8:12] == [0, 1, 2, 3]
    assert drop.run(prog, data, 4).memory[8:12] != [0, 1, 2, 3]


def test_control_binder_harness_kills_a_pinned_mutant(monkeypatch):
    """_killed sees a binder mutant the tests do pin (transpose output shape not transposed)."""
    m = _mutant(elementwise, "if _shape(img, out) != [c, r]:", "if _shape(img, out) != [r, c]:")
    assert _killed(elementwise, m, monkeypatch)


def test_control_no_fixture_skipped_at_head_and_bar_mutant_skips():
    """At HEAD nothing is skipped at W = 2; a model whose BAR releases when any warp arrives makes 11 fixtures
    interleaving-dependent, and test_isa_diff would skip them instead of failing."""
    assert _skipped(isasim, 2) == []
    m = _mutant(isasim, "if all(w.bar for w in live):", "if any(w.bar for w in live):")
    assert len(_skipped(m, 2)) == 11
