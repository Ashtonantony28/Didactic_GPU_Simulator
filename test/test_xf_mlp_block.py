"""MLP block chain (rmsnorm -> gate, up -> swiglu -> down -> residual) on the exported image.

Every intermediate is bit-exact vs the golden chain; the dequantized output is within the
fixture tol of the nanoGPT float reference.
"""
import cocotb
import numpy as np

from xf import export, fixture, mlp
from xf.image import Image
from .helpers.runner import Launch, run_launches

FIXTURES = [fixture.load(fixture.FIXTURE_DIR / f"mlp_block_{i}.json") for i in range(4)]
T, D = FIXTURES[0]["image"].segments["x"]["shape"]


async def _run(dut, kind, xs):
    """Run the chain once per input on one image; return (image, [(acts read back, golden)])."""
    img = Image.from_dict(FIXTURES[0]["image"].to_dict())
    sets, launches = [], []
    for i, x in enumerate(xs):
        acts = mlp.alloc_acts(img, f"in{i}.") if i else {n: n for n in mlp.ACTS}
        img.write(acts["x"], x)
        launches += [Launch(p, t, out, timeout_cycles=200_000) for out, (p, t) in mlp.launches(img, kind, acts)]
        sets.append(acts)
    _, mem = await run_launches(dut, img, launches)
    got = [{n: img.read(a[n], mem.memory) for n in mlp.ACTS[1:]} for a in sets]
    return img, [(g, export.mlp_block_int(img, x)) for g, x in zip(got, xs)]


def _bit_exact(results):
    bad = [(i, n) for i, (got, want) in enumerate(results) for n in want if not np.array_equal(got[n], want[n])]
    assert not bad, f"mismatch (input, segment): {bad}"


async def _fixtures(dut, kind):
    img, results = await _run(dut, kind, [fx["inputs"]["x"] for fx in FIXTURES])
    _bit_exact(results)
    scale = img.segments["out"]["scale"]
    for i, (fx, (got, want)) in enumerate(zip(FIXTURES, results)):
        for n, g in fx["golden"].items():
            assert np.array_equal(want[n], g), f"fixture {i}: golden {n} is stale"
        err = np.abs(got["out"] * scale - fx["reference"]["out"]).max()
        assert err <= fx["tol"], f"fixture {i}: max error {err} > tol {fx['tol']}"


async def _edges(dut, kind):
    rng = np.random.default_rng(6)
    _, results = await _run(dut, kind, [
        np.zeros((T, D)),                                      # rmsnorm of a zero row
        np.full((T, D), -128),
        np.full((T, D), 127),
        [[-128, 127] * (D // 2), [127] * D, [0] * (D - 1) + [-128], [1] * D],
        np.where(rng.random((T, D)) < 0.5, -128, 127),
        rng.integers(-128, 128, (T, D)),
    ])
    _bit_exact(results)


@cocotb.test()
async def test_fixtures_naive(dut):
    await _fixtures(dut, "naive")


@cocotb.test()
async def test_fixtures_tiled(dut):
    await _fixtures(dut, "tiled")


@cocotb.test()
async def test_edges_naive(dut):
    await _edges(dut, "naive")


@cocotb.test()
async def test_edges_tiled(dut):
    await _edges(dut, "tiled")
