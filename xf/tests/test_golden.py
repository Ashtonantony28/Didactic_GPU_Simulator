"""Golden ops vs. scalar Python references (exact rationals, explicit loops)."""
import math
from fractions import Fraction

import numpy as np
import pytest

from xf import golden

SEEDS = range(10)
EDGE = [-128, -127, -1, 0, 1, 127]


def ref_round_shift(v, s):
    """round-half-up(v / 2^s), exact."""
    return math.floor(Fraction(v, 1 << s) + Fraction(1, 2)) if s else v


def ref_requant(acc, m, s):
    return max(-128, min(127, ref_round_shift(acc * m, s)))


def ref_matmul(a, b):
    return [[sum(int(a[i][k]) * int(b[k][j]) for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def rand_i8(rng, shape):
    return rng.integers(-128, 128, shape)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("shape", [(4, 8, 16), (4, 16, 8), (1, 8, 8), (4, 5, 3)])
def test_matmul_random(seed, shape):
    rng = np.random.default_rng(seed)
    m, k, n = shape
    a, b = rand_i8(rng, (m, k)), rand_i8(rng, (k, n))
    assert golden.matmul_i8(a, b).tolist() == ref_matmul(a, b)


@pytest.mark.parametrize("va,vb", [(-128, -128), (127, 127), (-128, 127), (0, -128)])
def test_matmul_extremes(va, vb):
    a, b = np.full((4, 16), va), np.full((16, 8), vb)
    assert golden.matmul_i8(a, b).tolist() == ref_matmul(a, b)


def test_matmul_rejects_non_int8():
    with pytest.raises(ValueError):
        golden.matmul_i8(np.full((1, 1), 128), np.ones((1, 1)))


def test_matmul_accumulator_bound():
    # 2^17 products of 16384 each overflow int32
    with pytest.raises(OverflowError):
        golden.matmul_i8(np.full((1, 1 << 17), -128), np.full((1 << 17, 1), -128))


@pytest.mark.parametrize("seed", SEEDS)
def test_requant_random(seed):
    rng = np.random.default_rng(seed)
    acc = rng.integers(-(1 << 20), 1 << 20, 64)
    m, s = int(rng.integers(1, 1 << 10)), int(rng.integers(0, 32))
    assert golden.requant(acc, m, s).tolist() == [ref_requant(int(v), m, s) for v in acc]


@pytest.mark.parametrize("acc", [-(1 << 31), -3, -2, -1, 0, 1, 2, 3, (1 << 31) - 1])
@pytest.mark.parametrize("s", [0, 1, 2, 31])
def test_requant_edges(acc, s):
    if s and acc + (1 << (s - 1)) > golden.I32[1]:
        with pytest.raises(OverflowError):
            golden.requant(np.array([acc]), 1, s)
    else:
        assert int(golden.requant(np.array([acc]), 1, s)[0]) == ref_requant(acc, 1, s)


def test_requant_saturates_and_rounds_half_up():
    assert golden.requant(np.array([1000, -1000, 3, -3, 5, -5]), 1, 1).tolist() == [127, -128, 2, -1, 3, -2]


@pytest.mark.parametrize("acc,m,s", [((1 << 30), 2, 0), (-(1 << 30), 3, 0), ((1 << 31) - 1, 1, 1), (0, 1, 32), (0, 1, -1)])
def test_requant_rejects_overflow_and_bad_shift(acc, m, s):
    with pytest.raises((OverflowError, ValueError)):
        golden.requant(np.array([acc]), m, s)


def test_relu():
    assert golden.relu(np.array(EDGE)).tolist() == [max(0, v) for v in EDGE]


def _silu_lut():
    return np.array([round(127 * (x / 32) / (1 + math.exp(-x / 32)) / 4) for x in range(-128, 128)])


@pytest.mark.parametrize("seed", SEEDS)
def test_swiglu_random(seed):
    rng = np.random.default_rng(seed)
    gate, up = rand_i8(rng, (4, 16)), rand_i8(rng, (4, 16))
    lut, m, s = _silu_lut(), int(rng.integers(1, 256)), int(rng.integers(4, 16))
    want = [[ref_requant(int(lut[g + 128]) * int(u), m, s) for g, u in zip(gr, ur)] for gr, ur in zip(gate, up)]
    assert golden.swiglu_lut(gate, up, lut, m, s).tolist() == want


def test_swiglu_lut_endpoints():
    lut = np.arange(256) - 128
    gate, up = np.array(EDGE), np.array([-128] * len(EDGE))
    assert golden.swiglu_lut(gate, up, lut, 1, 7).tolist() == [ref_requant(g * -128, 1, 7) for g in EDGE]
    with pytest.raises(ValueError):
        golden.swiglu_lut(gate, up, lut[:-1], 1, 7)


@pytest.mark.parametrize("seed", SEEDS)
def test_residual_add(seed):
    rng = np.random.default_rng(seed)
    a, b = rand_i8(rng, (4, 8)), rand_i8(rng, (4, 8))
    ma, mb, s = int(rng.integers(1, 1 << 12)), int(rng.integers(1, 1 << 12)), int(rng.integers(0, 16))
    want = [[ref_requant(int(x) * ma + int(y) * mb, 1, s) for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]
    assert golden.residual_add(a, b, ma, mb, s).tolist() == want


def test_residual_add_saturates():
    assert golden.residual_add(np.array([127, -128]), np.array([127, -128]), 1, 1, 0).tolist() == [127, -128]


def _rsqrt_lut(shift, n=256):
    return np.array([round((1 << 12) / math.sqrt(((i + 0.5) * (1 << shift)) / 8 + 1)) for i in range(n)])


def ref_rmsnorm(x, w, lut, shift, m, s):
    out = []
    for row in x:
        ss = sum(int(v) * int(v) for v in row)
        r = int(lut[min(ss >> shift, len(lut) - 1)])
        out.append([ref_requant(int(v) * int(g) * r, m, s) for v, g in zip(row, w)])
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_rmsnorm_random(seed):
    rng = np.random.default_rng(seed)
    x, w = rand_i8(rng, (4, 8)), rand_i8(rng, (8,))
    lut = _rsqrt_lut(9)
    assert golden.rmsnorm_int(x, w, lut, 9, 1, 14).tolist() == ref_rmsnorm(x, w, lut, 9, 1, 14)


def test_rmsnorm_zero_and_extreme_rows():
    x = np.array([[0] * 8, [-128] * 8, [127] * 8, EDGE + [0, 0]])
    w = np.array([127, -128, 1, -1, 0, 64, -64, 127])
    lut = _rsqrt_lut(9)
    got = golden.rmsnorm_int(x, w, lut, 9, 1, 14)
    assert got.tolist() == ref_rmsnorm(x, w, lut, 9, 1, 14)
    assert got[0].tolist() == [0] * 8
    # sum of squares 8*16384 >> 9 = 256 clamps to the last LUT entry
    assert golden.rmsnorm_int(x[1:2], w, lut, 9, 1, 14).tolist() == ref_rmsnorm(x[1:2], w, lut, 9, 1, 14)


def test_rmsnorm_rejects_overflowing_lut():
    with pytest.raises(OverflowError):
        golden.rmsnorm_int(np.full((1, 8), -128), np.full(8, -128), np.full(256, 1 << 20), 9, 1, 0)


def _exp_lut():
    return np.array([round((1 << 15) * math.exp((d - 255) / 16)) for d in range(256)])


def ref_exp_rows(scores, lut, causal):
    rows = []
    for i, row in enumerate(scores):
        keep = [j <= i or not causal for j in range(len(row))]
        mx = max(int(v) for v, k in zip(row, keep) if k)
        rows.append([int(lut[int(v) - mx + 255]) if k else 0 for v, k in zip(row, keep)])
    return rows


def ref_softmax(scores, lut, one, causal):
    return [[(x * one + sum(e) // 2) // sum(e) for x in e] for e in ref_exp_rows(scores, lut, causal)]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("causal", [True, False])
def test_softmax_random(seed, causal):
    rng = np.random.default_rng(seed)
    scores, lut = rand_i8(rng, (4, 4)), _exp_lut()
    got = golden.softmax_lut(scores, lut, 127, causal)
    assert got.tolist() == ref_softmax(scores, lut, 127, causal)
    # each p is within half an LSB of e*one/S, so a row sums to `one` within T/2 LSB
    for gr, e in zip(got, ref_exp_rows(scores, lut, causal)):
        assert all(abs(int(g) - Fraction(x * 127, sum(e))) <= Fraction(1, 2) for g, x in zip(gr, e))
        assert abs(int(gr.sum()) - 127) <= 2


def test_softmax_extremes_and_mask():
    scores = np.array([[127, -128, 5, 5], [-128, 127, 0, 0], [0, 0, 0, 0], [-128, -128, -128, 127]])
    lut = _exp_lut()
    got = golden.softmax_lut(scores, lut, 127)
    assert got.tolist() == ref_softmax(scores, lut, 127, True)
    assert got[0].tolist() == [127, 0, 0, 0]
    assert np.all(np.triu(got, 1) == 0)
    assert got[2, :3].tolist() == [42, 42, 42]


def test_softmax_rejects_bad_lut():
    with pytest.raises(ValueError):
        golden.softmax_lut(np.zeros((2, 2)), np.ones(255), 127)
    with pytest.raises(ZeroDivisionError):
        golden.softmax_lut(np.zeros((2, 2)), np.zeros(256), 127)
