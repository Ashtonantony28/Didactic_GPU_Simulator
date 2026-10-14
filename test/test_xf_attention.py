"""Attention head: transpose and softmax kernels, then the chain (q, k, v -> kT -> scores -> p -> o -> y).

Kernels and every chain intermediate are bit-exact vs xf.golden / attention.attention_int; the
dequantized output is within the fixture tol of nanoGPT's CausalSelfAttention; every softmax row
sums to the fixed-point one within 1 LSB.
"""
import cocotb
import numpy as np

from xf import attention, elementwise as ew, fixture, golden
from xf.image import Image
from .helpers.runner import Launch, run_launches

FIXTURES = [fixture.load(fixture.FIXTURE_DIR / f"attention_{i}.json") for i in range(4)]
T, D = FIXTURES[0]["image"].segments["x"]["shape"]
ONE = attention.ONE


def _rows_sum_to_one(p, one=ONE):
    sums = np.asarray(p).sum(axis=-1)
    assert np.abs(sums - one).max() <= 1, f"row sums {sums.tolist()} not within 1 LSB of {one}"


async def check(dut, img, cases):
    """cases: [(out segment, (program, threads), golden)]; one launch per case on one image."""
    _, mem = await run_launches(dut, img, [Launch(p, t, out, timeout_cycles=200_000) for out, (p, t), _ in cases])
    got = {out: img.read(out, mem.memory) for out, _, _ in cases}
    bad = [out for out, _, want in cases if not np.array_equal(got[out], want)]
    assert not bad, f"mismatch: {bad}"
    return got


def _seg(img, name, data, dtype="i8"):
    data = np.asarray(data, dtype=np.int64)
    img.alloc(name, data.shape, dtype, data=data)
    return name


@cocotb.test()
async def test_transpose(dut):
    rng = np.random.default_rng(10)
    img, cases = Image(), []
    for name, x in (("4x8", rng.integers(-128, 128, (4, 8))), ("8x4", rng.integers(-128, 128, (8, 4))),
                    ("3x5", rng.integers(-128, 128, (3, 5))), ("1x7", [[-128, 127, 0, -1, 1, 5, -5]])):
        _seg(img, f"{name}.in", x)
        img.alloc(name, list(np.shape(x))[::-1], "i8")
        cases.append((name, ew.transpose(img, f"{name}.in", name), np.asarray(x).T))
    await check(dut, img, cases)


@cocotb.test()
async def test_softmax(dut):
    rng = np.random.default_rng(11)
    img, cases, exported = Image(), [], []
    _seg(img, "lut_fix", FIXTURES[0]["image"].read("exp_lut"), "i32")
    _seg(img, "lut_rand", np.append(rng.integers(0, 1 << 23, 255), 1 << 23), "i32")
    _seg(img, "lut_sparse", np.where(np.arange(256) < 250, 0, rng.integers(1, 1 << 23, 256)), "i32")
    _seg(img, "lut_flat", np.ones(256), "i32")
    _seg(img, "lut_mid", np.append(rng.integers(0, 1 << 20, 255), 1 << 20), "i32")   # e*1000 + S/2 fits int32

    def case(name, s, lut, one=ONE, dtype="i8"):
        _seg(img, f"{name}.in", s)
        img.alloc(name, list(np.shape(s)), dtype)
        cases.append((name, ew.softmax(img, f"{name}.in", lut, name, one), golden.softmax_lut(s, img.read(lut), one)))
        if lut == "lut_fix" and one == ONE:
            exported.append(name)

    for i in range(3):
        case(f"rand{i}", rng.integers(-128, 128, (T, T)), "lut_fix")
        case(f"rand_lut{i}", rng.integers(-128, 128, (T, T)), "lut_rand")
    case("zero", np.zeros((T, T)), "lut_fix")
    case("min", np.full((T, T), -128), "lut_fix")
    case("max", np.full((T, T), 127), "lut_fix")
    case("span", [[-128, 127, 127, 127], [127, -128, 127, 127], [-128, -128, 127, -128], [127, -128, -128, -128]], "lut_fix")
    case("masked_hi", np.triu(np.full((T, T), 255), 1) - 128, "lut_rand")       # masked entries above rowmax
    case("sparse", rng.integers(-128, 128, (T, T)), "lut_sparse")
    case("flat", rng.integers(-128, 128, (T, T)), "lut_flat")
    case("t1", [[-7]], "lut_fix")
    case("t8", rng.integers(-128, 128, (8, 8)), "lut_fix")
    case("one1000", rng.integers(-128, 128, (T, T)), "lut_mid", 1000, "i32")
    got = await check(dut, img, cases)
    for name in exported:
        _rows_sum_to_one(got[name])


async def _run(dut, kind, xs):
    """Run the chain once per input on one image; return (image, [(acts read back, golden)])."""
    img = Image.from_dict(FIXTURES[0]["image"].to_dict())
    sets, launches = [], []
    for i, x in enumerate(xs):
        acts = attention.alloc_acts(img, f"in{i}.") if i else {n: n for n in attention.ACTS}
        img.write(acts["x"], x)
        launches += [Launch(p, t, out, timeout_cycles=200_000) for out, (p, t) in attention.launches(img, kind, acts)]
        sets.append(acts)
    _, mem = await run_launches(dut, img, launches)
    got = [{n: img.read(a[n], mem.memory) for n in attention.ACTS[1:]} for a in sets]
    results = [(g, attention.attention_int(img, x)) for g, x in zip(got, xs)]
    bad = [(i, n) for i, (g, want) in enumerate(results) for n in want if not np.array_equal(g[n], want[n])]
    assert not bad, f"mismatch (input, segment): {bad}"
    for g, _ in results:
        _rows_sum_to_one(g["p"])
    return img, results


async def _fixtures(dut, kind):
    img, results = await _run(dut, kind, [fx["inputs"]["x"] for fx in FIXTURES])
    scale = img.segments["y"]["scale"]
    for i, (fx, (got, want)) in enumerate(zip(FIXTURES, results)):
        for n, g in fx["golden"].items():
            assert np.array_equal(want[n], g), f"fixture {i}: golden {n} is stale"
        err = np.abs(got["y"] * scale - fx["reference"]["y"]).max()
        assert err <= fx["tol"], f"fixture {i}: max error {err} > tol {fx['tol']}"


async def _edges(dut, kind):
    rng = np.random.default_rng(12)
    await _run(dut, kind, [
        np.zeros((T, D)),
        np.full((T, D), -128),
        np.full((T, D), 127),
        [[-128, 127] * (D // 2), [127] * D, [0] * (D - 1) + [-128], [1] * D],
        np.where(rng.random((T, D)) < 0.5, -128, 127),
        rng.integers(-128, 128, (T, D)),
    ])


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
