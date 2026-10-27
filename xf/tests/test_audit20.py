"""AUDIT-20 repros (API/SDK consistency, round 5). Each finding is a strict xfail until fixed; the plain tests are
controls that pin today's behaviour the findings rely on."""
import importlib.util
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from xf import elementwise as ew
from xf.asm import assemble
from xf.image import Image
from xf.isa_fixtures import FIXTURE_DIR, load
from xf.isasim import IsaSimError, run, run_trace

ROOT = Path(__file__).resolve().parents[2]

# -P forms that start with -Pgpu. (so SG27's scope check passes them) but that iverilog ignores with exit 0: a value it
# cannot parse (it prints an error and keeps the default) or a hierarchical name (dropped without a message).
IGNORED = ["-Pgpu.WARP_POLICY=GTO", "-Pgpu.ICACHE_LINES=16k", "-Pgpu.ICACHE_LINES",
           "-Pgpu.cores[0].core_instance.SMEM_SIZE=32", "-Pgpu.gpu.ICACHE_LINES=16"]


def _dry_run(target, params, tmp="build/a20-dry"):
    """`make -n <target>` (nothing is built) -> (exit code, the iverilog command lines)."""
    res = subprocess.run(["make", "-n", "--no-print-directory", target, f"BUILD={tmp}", f"PARAMS={params}"],
                         cwd=ROOT, capture_output=True, text=True)
    return res.returncode, [ln for ln in res.stdout.splitlines() if ln.startswith("iverilog ")]


def _bench():
    spec = importlib.util.spec_from_file_location("bench_run", ROOT / "bench" / "run.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench


def _iverilog_value(tmp_path, switch):
    """(iverilog exit code, the value gpu.A elaborates to) for one -P switch on a two-module design."""
    (tmp_path / "t.v").write_text('module sub #(parameter S = 64) (); endmodule\n'
                                  'module gpu #(parameter A = 1) (); sub s(); initial $display("A=%0d", A); endmodule\n')
    rc = subprocess.run(["iverilog", "-o", "t.vvp", "-s", "gpu", "-g2012", switch, "t.v"],
                        cwd=tmp_path, capture_output=True, text=True).returncode
    out = subprocess.run(["vvp", "t.vvp"], cwd=tmp_path, capture_output=True, text=True).stdout
    return rc, int(re.search(r"A=(\d+)", out)[1])


# ---- M1: the SG27 scope check passes -Pgpu. switches that iverilog ignores ---------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-20 M1: compile accepts -Pgpu. switches iverilog ignores with exit 0 "
                                       "(unparsable value, hierarchical name); the default GPU is built")
@pytest.mark.parametrize("params", IGNORED)
def test_compile_rejects_params_iverilog_ignores(params):
    rc, _ = _dry_run("compile", params)
    assert rc != 0, f"make compile accepted {params!r}, which iverilog ignores"


@pytest.mark.parametrize("switch", ["-Pgpu.A=GTO", "-Pgpu.A=16k", "-Pgpu.A", "-Pgpu.s.S=8", "-Pgpu.gpu.A=2"])
def test_control_iverilog_ignores_these_forms_with_exit_0(tmp_path, switch):
    assert _iverilog_value(tmp_path, switch) == (0, 1)


def test_control_valid_scoped_param_applies(tmp_path):
    assert _iverilog_value(tmp_path, "-Pgpu.A=2") == (0, 2)
    assert _dry_run("compile", "-PICACHE_LINES=16")[0] != 0      # SG27: unscoped is rejected


# ---- M2: bench/run.py prefixes every --params word with -Pgpu. -------------------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-20 M2: bench/run.py --params '-Pgpu.ICACHE_LINES=16' becomes "
                                       "-Pgpu.-Pgpu.ICACHE_LINES=16, which passes the scope check and iverilog drops")
def test_bench_params_in_make_form_are_not_dropped():
    bench = _bench()
    try:
        params = bench.iv("-Pgpu.ICACHE_LINES=16")
    except (ValueError, SystemExit):
        return
    rc, cmds = _dry_run("compile", params)
    assert rc != 0 or all("-Pgpu.-P" not in c for c in cmds), cmds


def test_control_bench_iv_doubles_the_prefix():
    bench = _bench()
    assert bench.iv("ICACHE_LINES=16 COALESCE=1") == "-Pgpu.ICACHE_LINES=16 -Pgpu.COALESCE=1"
    assert bench.iv("-Pgpu.ICACHE_LINES=16") == "-Pgpu.-Pgpu.ICACHE_LINES=16"


# ---- L1: SG25's trace API is outside the contracts, and a callable warp_policy is trusted ----------------------

BAR_PROG = assemble("CONST R0, #0\nBAR\nSTS R0, %threadIdx\nLDS R1, R0\nSTR %threadIdx, R1\nRET")


@pytest.mark.xfail(strict=True, reason="AUDIT-20 L1: contract 9 still says warp_policy is 'lrr' or 'gto'; it does not "
                                       "name the callable form, run_trace or run_kernel(trace=...)")
def test_contract9_names_the_trace_api():
    plan = (ROOT / "PLAN.md").read_text()
    c9 = plan[plan.index("9. **ISA reference model**"):plan.index("10. **Reconvergence encoding**")]
    assert "run_trace" in c9 and "callable" in c9, c9


@pytest.mark.xfail(strict=True, reason="AUDIT-20 L1: run() trusts a callable warp_policy: a warp held at BAR runs past "
                                       "it silently, a finished warp raises IndexError instead of IsaSimError")
@pytest.mark.parametrize("pick", ["held", "done"])
def test_callable_policy_pick_is_validated(pick):
    seen = []

    def policy(b, ready):
        seen.append(list(ready))
        if pick == "held":
            return 0 if len(seen) == 3 else ready[0]      # warp 0 waits at BAR from issue 2; picked once there
        return 0                                          # warp 0 keeps being picked after its RET
    with pytest.raises(IsaSimError):
        run(BAR_PROG, [], 4, warps_per_core=2, warp_policy=policy)


def test_control_run_trace_replays_a_recorded_order():
    order = {}

    def lrr(b, ready, cur={}):
        w = next(i % 2 for i in range(cur.get(b, 0), cur.get(b, 0) + 2) if i % 2 in ready)
        cur[b] = (w + 1) % 2
        order.setdefault(b, []).append(w)
        return w
    want = run(BAR_PROG, [], 8, warps_per_core=2, warp_policy="lrr")
    got = run(BAR_PROG, [], 8, warps_per_core=2, warp_policy=lrr)
    assert (got.memory, got.steps) == (want.memory, want.steps)
    again = run_trace(order, BAR_PROG, [], 8, warps_per_core=2)
    assert (again.memory, again.steps) == (want.memory, want.steps)
    with pytest.raises(IsaSimError):
        run_trace({b: s + [0] for b, s in order.items()}, BAR_PROG, [], 8, warps_per_core=2)


# ---- L2: docs/isa.md says every same-word STR is listed in stores_conflict ------------------------------------

SAME_VALUE = assemble("CONST R0, #0\nCONST R1, #7\nSTR R0, R1\nRET")
DIFF_VALUE = assemble("CONST R0, #0\nSTR R0, %threadIdx\nRET")


@pytest.mark.xfail(strict=True, reason="AUDIT-20 L2: docs/isa.md says isasim lists same-word STR lanes in "
                                       "stores_conflict, but lanes storing the same value are not listed")
def test_same_value_str_conflict_listed_or_documented():
    isa = (ROOT / "docs" / "isa.md").read_text()
    para = next(p for p in isa.split("\n\n") if "stores_conflict" in p)
    assert run(SAME_VALUE, [], 4).stores_conflict or re.search(r"differ|different value", para), para


def test_control_str_conflict_highest_lane_and_listed():
    r = run(DIFF_VALUE, [], 4)
    assert r.memory[0] == 3 and r.stores_conflict == [(0, 1, 0)]
    assert run(SAME_VALUE, [], 4).stores_conflict == []


# ---- L3: ADR-009's test description predates SG25's reads-only priority ----------------------------------------

@pytest.mark.xfail(strict=True, reason="AUDIT-20 L3: ADR-009 says test_medic fails if a low-utility lane gets a channel "
                                       "while a high-utility lane waits; since SG25 priority and the monitor cover reads")
def test_adr009_test_paragraph_says_reads():
    adr = (ROOT / "docs" / "decisions" / "ADR-009-medic.md").read_text()
    tests = adr[adr.index("## Tests"):adr.index("## Results")]
    sentence = re.search(r"low-utility lane gets a channel[^.]*\.", " ".join(tests.split()))[0]
    assert "read" in sentence, sentence


# ---- SG26 against the binder limit: controls -------------------------------------------------------------------

def test_control_relu_binder_takes_255_and_refuses_256():
    img = Image()
    for n in (255, 256):
        img.alloc(f"x{n}", [n], "i8", data=np.zeros(n, dtype=np.int64))
        img.alloc(f"y{n}", [n], "i8")
    assert ew.relu(img, "x255", "y255")[1] == 255
    with pytest.raises(ValueError, match="8-bit thread count"):
        ew.relu(img, "x256", "y256")
    assert "y256.args" not in img.segments


def test_control_launch_fixtures_sit_at_the_binder_limit():
    fx = {p.stem: load(p) for p in FIXTURE_DIR.glob("launch*_25[35].json")}
    assert sorted(fx) == ["launch32_253", "launch32_255", "launch8_253", "launch8_255"]
    for name, f in fx.items():
        assert f["threads"] <= 255
        r = run(f["program"], f["data"], f["threads"], **f["params"])
        assert r.memory[:len(f["memory"])] == f["memory"], name
