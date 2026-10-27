"""AUDIT-15 repros (performance under load, round 4). Strict xfails until fixed; the rest are passing controls.

The RTL tests write a small cocotb probe module into build/a15-pytest/ (the rules keep audit probes out of test/),
build the GPU there with sv2v and iverilog, and run one probe without the Makefile. They skip when the toolchain is
missing. About 60 s in all.
"""
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / "build" / "a15-pytest"
TOOLS = ("sv2v", "iverilog", "vvp", "cocotb-config")
needs_rtl = pytest.mark.skipif(any(shutil.which(t) is None for t in TOOLS), reason="needs sv2v, iverilog, cocotb")

# Per-lane loop of data[gid] (1..4) trips under SYNC, then an SMEM neighbour exchange between two BARs, OUTER times.
# Without the BARs (NOBAR) the warps never wait for each other.
DIVBAR = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R8, #0
    CONST R9, #{outer}
    CONST R5, #0
outer:
    LDR R2, R0
    CONST R3, #0
inner:
    SYNC done
    ADD R5, R5, R2
    ADD R3, R3, R1
    CMP R3, R2
    BRn inner
done:
    STS %threadIdx, R5
    BAR
    ADD R6, %threadIdx, R1
    DIV R7, R6, %blockDim
    MUL R7, R7, %blockDim
    SUB R6, R6, R7
    LDS R4, R6
    ADD R5, R5, R4
    BAR
    ADD R8, R8, R1
    CMP R8, R9
    BRn outer
    STR R0, R5
    RET
"""
NOBAR = DIVBAR.replace("    BAR\n", "")

PROBE = '''
"""AUDIT-15 probe, written by xf/tests/test_audit15.py: run A15_KERNEL on the DUT at MEM_LATENCY; memory and
warp_issues_total must equal xf.isasim replaying the RTL issue order; check_counters runs inside run_kernel."""
import json, os
import cocotb
from xf.asm import assemble
from xf.isasim import run, run_trace
from test.helpers.runner import run_kernel


@cocotb.test()
async def probe(dut):
    p = lambda n: int(getattr(dut, n).value)
    prog = assemble(os.environ["A15_KERNEL"])
    threads, w = int(os.environ["A15_THREADS"]), p("WARPS_PER_CORE")
    data = [1 + i % 4 for i in range(256)]
    kw = dict(word_bits=p("DATA_MEM_DATA_BITS"), addr_bits=p("DATA_MEM_ADDR_BITS"), simt=bool(p("SIMT_STACK")),
              threads_per_block=p("THREADS_PER_BLOCK"), warps_per_core=w)
    trace = {} if w > 1 else None
    metrics, mem = await run_kernel(dut, prog, data, threads, timeout_cycles=2_000_000, trace=trace)
    want = run_trace(trace, prog, data, threads, **kw) if w > 1 else run(prog, data, threads, **kw)
    out = dict(metrics, issues=p("warp_issues_total"), steps=want.steps, bar_stalls=p("bar_stall_cycles_total"),
               memory_ok=mem.memory[:256] == want.memory[:256])
    with open(os.environ["A15_OUT"], "w") as f:
        json.dump(out, f)
    assert out["memory_ok"] and out["issues"] == out["steps"], out
'''


def _build(name, params):
    """sv2v + iverilog into build/a15-pytest/<name>/sim.vvp (what `make compile` does)."""
    d = BUILD / name
    vvp = d / "sim.vvp"
    d.mkdir(parents=True, exist_ok=True)
    srcs = sorted(str(p) for p in (ROOT / "src").glob("*.sv") if p.name != "warp_scheduler_tb.sv")
    subprocess.run(["sv2v", "-w", str(d / "raw.v"), *srcs], check=True, capture_output=True)
    (d / "top.v").write_text("`timescale 1ns/1ns\n" + (d / "raw.v").read_text())
    subprocess.run(["iverilog", "-o", str(vvp), "-s", "gpu", "-g2012", *params.split(), str(d / "top.v")],
                   check=True, capture_output=True)
    return vvp


def _probe(vvp, kernel, threads, latency, tag):
    """Run the probe on a build; returns its metrics (cycles, issues, steps, bar_stalls, memory_ok, ...)."""
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "probe_a15.py").write_text(PROBE)
    lib = subprocess.run(["cocotb-config", "--lib-dir"], check=True, capture_output=True, text=True).stdout.strip()
    libpython = subprocess.run(["cocotb-config", "--libpython"], check=True, capture_output=True, text=True).stdout.strip()
    out, results = vvp.parent / f"{tag}.json", vvp.parent / f"{tag}.xml"
    out.unlink(missing_ok=True)
    env = os.environ | {
        "PYGPI_PYTHON_BIN": shutil.which("python3"), "LIBPYTHON_LOC": libpython,
        "PYTHONPATH": f"{ROOT}{os.pathsep}{BUILD}", "COCOTB_TEST_MODULES": "probe_a15",
        "COCOTB_RESULTS_FILE": str(results), "MEM_LATENCY": str(latency), "A15_KERNEL": kernel,
        "A15_THREADS": str(threads), "A15_OUT": str(out)}
    run = subprocess.run(["vvp", "-M", lib, "-m", "libcocotbvpi_icarus", str(vvp)], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=900)
    assert out.exists(), run.stdout[-3000:]
    return json.loads(out.read_text())


C1W4IC = "-Pgpu.NUM_CORES=1 -Pgpu.WARPS_PER_CORE=4 -Pgpu.ICACHE_LINES=16 -Pgpu.WARP_POLICY={}"


def _gto_over_lrr(kernel):
    k = "bar" if "BAR" in kernel else "nobar"
    lrr = _probe(_build("c1w4ic_lrr", C1W4IC.format(0)), kernel, 4, 10, f"lrr_{k}")
    gto = _probe(_build("c1w4ic_gto", C1W4IC.format(1)), kernel, 4, 10, f"gto_{k}")
    assert lrr["memory_ok"] and gto["memory_ok"]
    return gto["cycles"] / lrr["cycles"] - 1


@needs_rtl
@pytest.mark.xfail(strict=True, reason="AUDIT-15 L1: on a kernel with BAR, GTO is 12% slower than LRR on one block "
                                       "(19-32% at 4 cores x 4 warps); the final report's 3.5-6.5% comes from "
                                       "BAR-free kernels only")
def test_gto_cost_on_a_bar_kernel_is_within_the_reported_range():
    """docs/final-report.md: 'With fetch latency in play LRR beats GTO by 3.5-6.5% at L10'. Same setup (both
    memories at L10, ICACHE_LINES=16), one block of 4 warps, a loop with two BARs per trip: 7195 vs 6411 cycles."""
    assert _gto_over_lrr(DIVBAR.format(outer=6)) <= 0.065


@needs_rtl
def test_gto_cost_without_bar_is_within_the_reported_range():
    """Control: the same kernel with the BARs removed: GTO 6419 vs LRR 6349 cycles (+1.1%)."""
    assert _gto_over_lrr(NOBAR.format(outer=6)) <= 0.065


@needs_rtl
def test_counters_hold_at_255_threads_under_load():
    """Control (contract 12): 255 threads (64 blocks, a 3-thread last block) on 4 cores x 4 warps under GTO with
    SIMT_STACK, COALESCE, DCACHE, MEDIC and the I-cache, both memories at latency 10, a divergent loop and two BARs
    per trip. Memory and warp_issues_total equal the model's replay of the RTL issue order; check_counters passes."""
    params = ("-Pgpu.NUM_CORES=4 -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1 -Pgpu.SIMT_STACK=1 -Pgpu.COALESCE=1 "
              "-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1 -Pgpu.ICACHE_LINES=16")
    m = _probe(_build("c4w4_load", params), DIVBAR.format(outer=1), 255, 10, "load")
    assert m["memory_ok"] and m["issues"] == m["steps"]
    assert m["dcache_hits"] + m["dcache_misses"] == 255   # one LDR per thread
    assert m["data_mem_requests"] == m["dcache_misses"] - m["data_mem_requests_coalesced"] + 255   # + one STR each


@pytest.mark.xfail(strict=True, reason="AUDIT-15 L2: test_isa_diff and test_divergence give every kernel a fixed "
                                       "200 000-cycle timeout; xf_mlp_block_matmul_naive needs 208 858 cycles at "
                                       "NUM_CORES=1 WARPS_PER_CORE=4 MEM_LATENCY=16 and fails with TimeoutError")
def test_differential_timeouts_scale_with_the_run():
    """Structural: the differential tests should derive the timeout from the model's steps and the latency
    (AUDIT-10 L2's fix), not hard-code a cycle count."""
    fixed = [f"{p.name}: {m}" for p in (ROOT / "test" / "test_isa_diff.py", ROOT / "test" / "test_divergence.py")
             for m in re.findall(r"timeout_cycles=\d[\d_]*", p.read_text())]
    assert not fixed, fixed
