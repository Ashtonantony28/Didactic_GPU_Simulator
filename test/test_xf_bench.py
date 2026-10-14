"""Benchmark workloads for bench/run.py --xf: naive vs tiled matmul on the SG4 shapes, and the MLP block.

Each test checks every output bit-exact against the golden model before it reports metrics.
"""
import cocotb
import numpy as np

from xf import export, fixture, golden, kernels, mlp
from xf.image import Image
from .helpers.runner import Launch, run_launches, write_bench_result

KINDS = ("naive", "tiled")
SHAPES = ((4, 8, 16), (4, 16, 8), (1, 8, 8))


def _rows(metrics, cols):
    return [{**cols(m), **{k: v for k, v in m.items() if k not in ("launch", "name")}} for m in metrics]


@cocotb.test()
async def bench_matmul(dut):
    rng = np.random.default_rng(0)
    img, launches, want = Image(), [], {}
    for m, k, n in SHAPES:
        shape = f"{m}x{k}x{n}"
        a, b = rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (k, n))
        img.alloc(f"{shape}.a", a.shape, "i8", data=a)
        img.alloc(f"{shape}.b", b.shape, "i8", data=b)
        for kind in KINDS:
            c = f"{shape}.{kind}"
            img.alloc(c, (m, n), "i32")
            launches.append(Launch(*kernels.matmul(img, kind, f"{shape}.a", f"{shape}.b", c), c, 200_000))
            want[c] = golden.matmul_i8(a, b)
    metrics, mem = await run_launches(dut, img, launches)
    bad = [c for c in want if not np.array_equal(img.read(c, mem.memory), want[c])]
    assert not bad, f"mismatch: {bad}"
    write_bench_result("xf_matmul", {"rows": _rows(metrics, lambda m: dict(zip(("case", "kind"), m["name"].split("."))))})


@cocotb.test()
async def bench_mlp(dut):
    fx = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")
    img, x = fx["image"], fx["inputs"]["x"]
    launches, acts = [], {}
    for kind in KINDS:
        acts[kind] = mlp.alloc_acts(img, f"{kind}.")
        img.write(acts[kind]["x"], x)
        launches += [Launch(p, t, out, 200_000) for out, (p, t) in mlp.launches(img, kind, acts[kind])]
    metrics, mem = await run_launches(dut, img, launches)
    want = export.mlp_block_int(img, x)
    bad = [(kind, n) for kind in KINDS for n in mlp.ACTS[1:]
           if not np.array_equal(img.read(acts[kind][n], mem.memory), want[n])]
    assert not bad, f"mismatch (kind, segment): {bad}"
    err = np.abs(want["out"] * img.segments["out"]["scale"] - fx["reference"]["out"]).max()
    assert err <= fx["tol"], f"max error {err} > tol {fx['tol']}"
    write_bench_result("xf_mlp", {"rows": _rows(metrics, lambda m: dict(zip(("kind", "case"), m["name"].split("."))))})
