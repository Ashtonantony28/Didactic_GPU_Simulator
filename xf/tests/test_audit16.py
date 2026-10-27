"""AUDIT-16 repros (API/SDK consistency, round 4). Each finding is a strict xfail until fixed; the plain tests are
controls that pin today's behaviour the findings rely on."""
import re
import subprocess
from pathlib import Path

import pytest

from xf.asm import assemble
from xf.isa_fixtures import STS_CONFLICT
from xf.isasim import run

ROOT = Path(__file__).resolve().parents[2]


def _dry_run(target, params=None, tmp="build/a16-dry"):
    """`make -n <target>` (nothing is built) -> the iverilog command lines, by output .vvp."""
    cmd = ["make", "-n", "--no-print-directory", target, f"BUILD={tmp}"]
    if params is not None:
        cmd.append(f"PARAMS={params}")
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    lines = [ln for ln in res.stdout.splitlines() if ln.startswith("iverilog ")]
    return res.returncode, {re.search(r"-o (\S+)", ln)[1]: ln for ln in lines}


@pytest.mark.xfail(strict=True, reason="AUDIT-16 H1 (AUDIT-11 M1 re-rated): the Makefile's documented PARAMS form "
                                       "-PNAME=V is dropped by iverilog with exit 0; nothing rejects it")
def test_compile_rejects_unscoped_params():
    rc, _ = _dry_run("compile", "-PICACHE_LINES=16 -PCOALESCE=1")
    assert rc != 0, "make compile accepted unscoped -P switches, which build the default GPU"


def test_control_scoped_params_reach_iverilog():
    rc, cmds = _dry_run("compile", "-Pgpu.ICACHE_LINES=16")
    assert rc == 0 and "-Pgpu.ICACHE_LINES=16" in cmds["build/a16-dry/sim.vvp"]


@pytest.mark.xfail(strict=True, reason="AUDIT-16 M1: a PARAMS given to `make test_medic` replaces its "
                                       "DCACHE_LINES/MEDIC=1 configuration; both runs build MEDIC=0 and pass")
def test_medic_params_are_appended():
    rc, cmds = _dry_run("test_medic", "-Pgpu.DCACHE_LINES=16")
    assert rc == 0
    assert "-Pgpu.MEDIC=1" in cmds["build/a16-dry/sim.vvp"], cmds


def test_control_medic_default_params():
    rc, cmds = _dry_run("test_medic")
    assert rc == 0
    assert "-Pgpu.MEDIC=1" in cmds["build/a16-dry/sim.vvp"]
    assert "-Pgpu.MEDIC=0" in cmds["build/a16-dry/medic0/sim.vvp"]


def test_control_sts_conflicts_are_not_flagged():
    """STS conflicts: highest lane wins per block, and Result.stores_conflict stays empty (it covers STR only)."""
    r = run(assemble(STS_CONFLICT), [], 8)
    assert r.memory[:8] == [3, 3, 3, 3, 7, 7, 7, 7] and r.stores_conflict == []


@pytest.mark.xfail(strict=True, reason="AUDIT-16 L1: Result.stores_conflict covers STR only, but neither contract 9, "
                                       "isasim nor isa_fixtures says so; the fixture comment says it stores the thread id")
def test_stores_conflict_scope_documented():
    src = (ROOT / "xf" / "isasim.py").read_text()
    field = next(ln for ln in src.splitlines() if "stores_conflict:" in ln)
    assert "STR" in field, field
    fx = (ROOT / "xf" / "isa_fixtures.py").read_text()
    assert "stores its thread id" not in fx


@pytest.mark.xfail(strict=True, reason="AUDIT-16 L2: icache_misses_total counts every fetch when ICACHE_LINES=0 and "
                                       "check_counters enforces hits + misses == warp issues at W=1; no doc says so")
def test_icache_counter_semantics_documented():
    for doc in ("docs/architecture.md", "docs/decisions/ADR-012-multi-warp.md"):
        row = next(ln for ln in (ROOT / doc).read_text().splitlines() if "icache_misses_total" in ln)
        assert re.search(r"every (instruction )?fetch|ICACHE_LINES ?= ?0|warp_issues", row), f"{doc}: {row}"


@pytest.mark.xfail(strict=True, reason="AUDIT-16 L3: README and CLAUDE.md describe `make test_all` without "
                                       "`isa_fixtures --check` (README also without harness_selftest)")
def test_test_all_descriptions_match_makefile():
    for doc in ("README.md", "CLAUDE.md"):
        line = next(ln for ln in (ROOT / doc).read_text().splitlines() if "make test_all" in ln and "#" in ln)
        assert "isa_fixtures" in line and "harness_selftest" in line, f"{doc}: {line}"

