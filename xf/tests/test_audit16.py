"""AUDIT-16 repros (API/SDK consistency, round 4). Each finding is a strict xfail until fixed; the plain tests are
controls that pin today's behaviour the findings rely on."""
import importlib.util
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


def test_compile_rejects_unscoped_params():
    rc, _ = _dry_run("compile", "-PICACHE_LINES=16 -PCOALESCE=1")
    assert rc != 0, "make compile accepted unscoped -P switches, which build the default GPU"


@pytest.mark.parametrize("target", ["compile", "compile_xf", "test_matadd", "test_xf_bench", "test_medic",
                                    "multiwarp_w2_lrr", "divergence_w1"])
@pytest.mark.parametrize("params", ["-PICACHE_LINES=16", "-Pgpu.COALESCE=1 -PICACHE_LINES=16",
                                    "-Pwarp_scheduler_tb.POLICY=1"])
def test_every_gpu_build_rejects_unscoped_params(target, params):
    res = subprocess.run(["make", "-n", target, "BUILD=build/a16-dry", f"PARAMS={params}"],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode != 0 and "scope -P switches to gpu." in res.stderr, res.stderr
    assert not [ln for ln in res.stdout.splitlines() if ln.startswith("iverilog ")]


def test_control_scoped_params_reach_iverilog():
    rc, cmds = _dry_run("compile", "-Pgpu.ICACHE_LINES=16")
    assert rc == 0 and "-Pgpu.ICACHE_LINES=16" in cmds["build/a16-dry/sim.vvp"]


def test_medic_params_are_appended():
    rc, cmds = _dry_run("test_medic", "-Pgpu.DCACHE_LINES=16")
    assert rc == 0
    assert "-Pgpu.MEDIC=1" in cmds["build/a16-dry/sim.vvp"], cmds


def test_medic_params_win_on_shared_names():
    """PARAMS follow the target's own switches, and iverilog keeps the last -P, so DCACHE_LINES=32 wins."""
    rc, cmds = _dry_run("test_medic", "-Pgpu.DCACHE_LINES=32 -Pgpu.ICACHE_LINES=16")
    assert rc == 0
    for vvp, medic in (("build/a16-dry/sim.vvp", 1), ("build/a16-dry/medic0/sim.vvp", 0)):
        assert re.search(rf"-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC={medic} -Pgpu.DCACHE_LINES=32 -Pgpu.ICACHE_LINES=16 ",
                         cmds[vvp]), cmds


def test_iverilog_keeps_the_last_parameter_override(tmp_path):
    (tmp_path / "t.v").write_text('module gpu #(parameter A = 1) (); initial $display("A=%0d", A); endmodule\n')
    out = {}
    for order in ("2 3", "3 2"):
        a, b = order.split()
        subprocess.run(["iverilog", "-o", "t.vvp", "-s", "gpu", f"-Pgpu.A={a}", f"-Pgpu.A={b}", "t.v"],
                       cwd=tmp_path, check=True)
        out[order] = subprocess.run(["vvp", "t.vvp"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "A=3" in out["2 3"] and "A=2" in out["3 2"], out


def test_medic_rejects_medic0_in_params():
    res = subprocess.run(["make", "-n", "test_medic", "BUILD=build/a16-dry", "PARAMS=-Pgpu.MEDIC=0"],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode != 0 and "drop MEDIC=0" in res.stderr, res.stderr


def test_bench_medic_params_are_appended():
    """`bench/run.py --params "DCACHE_LINES=16" medic` runs `make test_medic PARAMS=<iv(params)>`."""
    spec = importlib.util.spec_from_file_location("bench_run", ROOT / "bench" / "run.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rc, cmds = _dry_run("test_medic", bench.iv("DCACHE_LINES=16"))
    assert rc == 0
    assert "-Pgpu.MEDIC=1" in cmds["build/a16-dry/sim.vvp"] and "-Pgpu.MEDIC=0" in cmds["build/a16-dry/medic0/sim.vvp"]


def test_matrix_medic_configs_build_medic1():
    """matrix_medic/matrix_all re-enter test_medic with a full MEDIC=1 PARAMS; the build stays MEDIC=1."""
    rc, cmds = _dry_run("test_medic", "-Pgpu.ICACHE_LINES=16 -Pgpu.PIPELINED=1 -Pgpu.COALESCE=1 "
                                      "-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1")
    assert rc == 0
    assert cmds["build/a16-dry/sim.vvp"].split()[-2] == "-Pgpu.MEDIC=1"
    assert cmds["build/a16-dry/medic0/sim.vvp"].split()[-2] == "-Pgpu.MEDIC=0"


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

