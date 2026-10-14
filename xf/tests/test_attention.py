"""Attention head image and golden chain: committed attention fixtures, bounds, softmax row sums (torch-free)."""
import numpy as np
import pytest

from xf import attention, export, fixture, golden

FIXTURES = sorted(fixture.FIXTURE_DIR.glob("attention_*.json"))


def test_fixtures_committed():
    assert len(FIXTURES) == export.N_FIXTURES


@pytest.fixture(params=FIXTURES, ids=lambda p: p.stem)
def fx(request):
    return fixture.load(request.param)


def test_fixture_layout(fx):
    img = fx["image"]
    assert (img.word_bits, img.addr_bits) == (32, 12) and len(img.words) <= 1 << 12
    assert set(fx["inputs"]) == {"x"} and set(fx["reference"]) == {"y"}
    assert set(fx["golden"]) == set(attention.ACTS[1:])
    for name, v in {**fx["inputs"], **fx["golden"]}.items():
        assert list(v.shape) == img.segments[name]["shape"] and img.segments[name]["dtype"] == "i8"
        assert not img.read(name).any()
    assert int(img.read("one")[0]) == attention.ONE
    assert fx["tol"] == pytest.approx(export.ATTN_TOL_LSB * img.segments["y"]["scale"])


def test_fixture_golden_reproduces(fx):
    got = attention.attention_int(fx["image"], fx["inputs"]["x"])
    for name, want in fx["golden"].items():
        assert np.array_equal(got[name], want), name


def test_fixture_within_tol(fx):
    y = fx["golden"]["y"] * fx["image"].segments["y"]["scale"]
    assert np.abs(y - fx["reference"]["y"]).max() <= fx["tol"]


def test_fixture_bounds(fx):
    worst = attention.check_bounds(fx["image"])
    assert all(v <= golden.I32[1] for v in worst.values())


def test_rows_sum_to_one_within_1_lsb(fx):
    img = fx["image"]
    t = img.segments["p"]["shape"][0]
    assert attention.row_sum_error(img.read("exp_lut"), attention.ONE, t) <= 1


def test_row_sum_error_finds_the_t_over_2_case():
    # four equal exps with one = 126: each p = (126 + 2) // 4 = 32, row sum 128 = one + 2
    assert attention.row_sum_error(np.ones(256, dtype=np.int64), 126, 4) == 2
    assert attention.row_sum_error(np.ones(256, dtype=np.int64), 126, 3) <= 1


def test_exp_bits_tight():
    e = attention.exp_bits(4)
    assert (attention.ONE + 2) << e <= golden.I32[1] < (attention.ONE + 2) << (e + 1)


@pytest.mark.parametrize("seed", range(10))
def test_no_overflow_on_any_int8_input(fx, seed):
    rng = np.random.default_rng(seed)
    shape = fx["inputs"]["x"].shape
    for x in (np.full(shape, -128), np.full(shape, 127), rng.integers(-128, 128, shape)):
        o = attention.attention_int(fx["image"], x)
        assert np.abs(o["p"].sum(axis=-1) - attention.ONE).max() <= 1
