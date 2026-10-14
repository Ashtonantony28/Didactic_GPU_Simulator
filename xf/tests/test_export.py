"""Exporter: fixed-point selection, and the committed mlp_block fixtures (torch-free)."""

import numpy as np
import pytest

from xf import export, fixture, golden

FIXTURES = sorted(fixture.FIXTURE_DIR.glob("mlp_block_*.json"))


@pytest.mark.parametrize("seed", range(20))
def test_fixed_point_fits_and_is_tight(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 3))
    ms = list(rng.uniform(1e-4, 4, n))
    bounds = [int(b) for b in rng.integers(1, 1 << 20, n)]
    mi, s = export.fixed_point(ms, bounds)
    assert 0 <= s <= 31
    assert sum(b * m for b, m in zip(bounds, mi)) + ((1 << (s - 1)) if s else 0) <= golden.I32[1]
    for m, q in zip(ms, mi):
        assert q == round(m * (1 << s))
    if s < 31:
        bigger = [round(m * (1 << (s + 1))) for m in ms]
        assert sum(b * m for b, m in zip(bounds, bigger)) + (1 << s) > golden.I32[1]


def test_fixed_point_rejects_impossible():
    with pytest.raises(OverflowError):
        export.fixed_point([1.0], [1 << 31])
    with pytest.raises(ValueError):
        export.fixed_point([1e-12], [1 << 24])


def test_rsqrt_relative_error_bounded():
    """lut[idx] / 2^k against the exact rsqrt every int8 row's sum of squares needs (AUDIT-1 H1)."""
    img = fixture.load(FIXTURES[0])["image"]
    sx, sh, sw = (img.segments[n]["scale"] for n in ("x", "h", "norm_w"))
    d = img.segments["x"]["shape"][-1]
    ss = np.arange(1, d * 128 * 128 + 1)
    idx, k = golden.rsqrt_index(ss, export.RSQRT_LEN)
    got = img.read("rsqrt_lut")[idx] / 2.0 ** (k + int(img.read("norm_s")[0]))
    want = 1 / np.sqrt(sx ** 2 * ss / d + export.EPS) * sx * sw / sh
    assert np.abs(got / want - 1).max() < 2 ** -8


def test_fixtures_committed():
    assert len(FIXTURES) == export.N_FIXTURES


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem)
def fx(request):
    return fixture.load(request.param)


def test_fixture_layout(fx):
    img = fx["image"]
    assert (img.word_bits, img.addr_bits) == (32, 12) and len(img.words) <= 1 << 12
    assert set(fx["inputs"]) == {"x"} and set(fx["reference"]) == {"out"}
    assert set(fx["golden"]) == {"h", "gate", "up", "act", "down", "out"}
    for name, v in {**fx["inputs"], **fx["golden"]}.items():
        assert list(v.shape) == img.segments[name]["shape"] and img.segments[name]["dtype"] == "i8"
        assert not img.read(name).any()
    assert fx["tol"] == pytest.approx(export.TOL_LSB * img.segments["out"]["scale"])


def test_fixture_golden_reproduces(fx):
    got = export.mlp_block_int(fx["image"], fx["inputs"]["x"])
    for name, want in fx["golden"].items():
        assert np.array_equal(got[name], want), name


def test_fixture_within_tol(fx):
    out = fx["golden"]["out"] * fx["image"].segments["out"]["scale"]
    assert np.abs(out - fx["reference"]["out"]).max() <= fx["tol"]


def test_fixture_bounds(fx):
    worst = export.check_bounds(fx["image"])
    assert all(v <= golden.I32[1] for v in worst.values())


@pytest.mark.parametrize("seed", range(10))
def test_no_overflow_on_any_int8_input(fx, seed):
    rng = np.random.default_rng(seed)
    shape = fx["inputs"]["x"].shape
    for x in (np.full(shape, -128), np.full(shape, 127), rng.integers(-128, 128, shape)):
        out = export.mlp_block_int(fx["image"], x)["out"]
        assert out.min() >= -128 and out.max() <= 127


@pytest.mark.parametrize("block", export.BLOCKS)
def test_export_matches_fixtures(nanogpt_dir, tmp_path, block):
    *_, files = export.fixture_bytes(nanogpt_dir, tmp_path, block)
    for name, data in files.items():
        assert (fixture.FIXTURE_DIR / name).read_bytes() == data


@pytest.mark.parametrize("block", export.BLOCKS)
def test_heldout_activations_unsaturated(nanogpt_dir, block):
    _, peak = export.heldout(nanogpt_dir, block, fixture.load(fixture.FIXTURE_DIR / f"{block}_0.json")["image"])
    assert max(peak.values()) <= 127.5, f"{block}: activations saturate on held-out inputs: {peak}"
