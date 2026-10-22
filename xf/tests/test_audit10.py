"""AUDIT-10 repros (performance under load). Each is a strict xfail until fixed.

The RTL repros build the GPU (or one module) with sv2v and iverilog under build/a10-pytest/ and run one probe of
test/test_audit10.py with cocotb, without the Makefile; they skip when the toolchain is missing. About 30 s in all.
"""
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "a10-pytest"
TOOLS = ("sv2v", "iverilog", "vvp", "cocotb-config")
needs_rtl = pytest.mark.skipif(any(shutil.which(t) is None for t in TOOLS), reason="needs sv2v, iverilog, cocotb")


def _build(name, params, top="gpu", sources=None):
    """sv2v + iverilog into build/a10-pytest/<name>/sim.vvp (what `make compile` does, with `top` as the top)."""
    d = BUILD / name
    vvp = d / "sim.vvp"
    d.mkdir(parents=True, exist_ok=True)
    srcs = sources or sorted(str(p) for p in (ROOT / "src").glob("*.sv") if p.name != "warp_scheduler_tb.sv")
    subprocess.run(["sv2v", "-w", str(d / "raw.v"), *srcs], check=True, capture_output=True)
    (d / "top.v").write_text("`timescale 1ns/1ns\n" + (d / "raw.v").read_text())
    subprocess.run(["iverilog", "-o", str(vvp), "-s", top, "-g2012", *params.split(), str(d / "top.v")],
                   check=True, capture_output=True)
    return vvp


def _probe(vvp, test, **env):
    """Run test/test_audit10.py::<test> on a build; True when it passes."""
    lib = subprocess.run(["cocotb-config", "--lib-dir"], check=True, capture_output=True, text=True).stdout.strip()
    libpython = subprocess.run(["cocotb-config", "--libpython"], check=True, capture_output=True, text=True).stdout.strip()
    results = vvp.parent / f"{test}.xml"
    results.unlink(missing_ok=True)
    run_env = {k: v for k, v in os.environ.items() if k != "MEM_LATENCY"} | {
        "PYGPI_PYTHON_BIN": shutil.which("python3"), "LIBPYTHON_LOC": libpython,
        "COCOTB_TEST_MODULES": "test.test_audit10", "COCOTB_TEST_FILTER": f"{test}$",
        "COCOTB_RESULTS_FILE": str(results), **{k: str(v) for k, v in env.items()}}
    out = subprocess.run(["vvp", "-M", lib, "-m", "libcocotbvpi_icarus", str(vvp)], cwd=ROOT, env=run_env,
                         capture_output=True, text=True, timeout=600)
    cases = [tc for tc in ET.parse(results).iter("testcase") if tc.get("name") == test]
    assert len(cases) == 1, out.stdout[-2000:]
    return cases[0].find("failure") is None


C4 = "-Pgpu.NUM_CORES=4"


@needs_rtl
@pytest.mark.xfail(strict=True, reason="AUDIT-10 M1: the one program channel claims fetchers in fixed core order, so "
                                       "at NUM_CORES=4 cores 0-2 take 15 of 16 blocks and core 3 runs one")
def test_four_cores_share_the_blocks():
    assert _probe(_build("c4", C4), "dispatch_spread", AUDIT10_LATENCY=0)


@needs_rtl
def test_four_cores_share_the_blocks_with_icache():
    """Control: with the I-cache the program channel is idle most of the time and every core runs 4 blocks."""
    assert _probe(_build("c4ic", C4 + " -Pgpu.ICACHE_LINES=16"), "dispatch_spread", AUDIT10_LATENCY=0)


@needs_rtl
@pytest.mark.xfail(strict=True, reason="AUDIT-10 M2: an L1-D miss fill and a write-through of the same address that "
                                       "complete in the same cycle leave the pre-write value cached")
def test_l1d_fill_racing_a_write_is_not_stale():
    vvp = _build("l1d", "-Pl1d_cache.NUM_CONSUMERS=2 -Pl1d_cache.LINES=4", top="l1d_cache",
                 sources=[str(ROOT / "src" / "l1d_cache.sv")])
    assert _probe(vvp, "l1d_stale_fill")


W4 = "-Pgpu.NUM_CORES=1 -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY={}"


@needs_rtl
@pytest.mark.xfail(strict=True, reason="AUDIT-10 L1: once warp ages saturate at 254, GTO breaks every tie to the "
                                       "lowest warp; warps 2 and 3 finish 23% after warps 0 and 1")
def test_gto_stays_fair_past_age_saturation():
    assert _probe(_build("c1w4gto", W4.format(1)), "gto_long_run_fairness", AUDIT10_LATENCY=0)


@needs_rtl
def test_lrr_long_run_is_fair():
    """Control: the same 409-issue-per-warp loop under LRR; the four warps finish within 36 cycles."""
    assert _probe(_build("c1w4lrr", W4.format(0)), "gto_long_run_fairness", AUDIT10_LATENCY=0)


@pytest.mark.xfail(strict=True, reason="AUDIT-10 M3: matrix_warps2 runs every xf test, the attention and MLP chains "
                                       "included, at 2.4x the default cycles: about 44 min, the matrix's long pole")
def test_matrix_skips_long_chains_on_slow_configs():
    mk = (ROOT / "Makefile").read_text()
    slow = [c for c in re.search(r"^MATRIX\s*:=\s*(.*)$", mk, re.M).group(1).split() if c.startswith("warps")]
    assert slow
    missing = [c for c in slow if not re.search(rf"^XF_{c}\s*:=", mk, re.M)]
    assert not missing, f"multi-warp matrix configurations that run the full xf suite: {missing}"
