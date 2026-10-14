"""AUDIT-6 H1 / SG14: attention TOL (6 LSB) holds on unsaturated adversarial and non-Gaussian inputs."""

import numpy as np
import pytest

from xf import attention, export, fixture

ATTN = fixture.load(fixture.FIXTURE_DIR / "attention_0.json")
N = 200_000
# found by hill-climbing int8 inputs under the constraint that no float activation exceeds 127.5 LSB:
# AUDIT-6's on the pre-ln_1 chain, then SG14's on the ln_1 chain (4.37, 4.01, 5.35, 5.23 LSB)
WORST = [
    [[60, -18, -43, 4, -78, 28, 41, 47], [-105, -24, 30, -22, 28, -6, -37, -23],
     [6, 56, 65, -101, -21, -80, -36, -50], [-31, 83, 116, 71, 25, 38, -62, -116]],
    [[100, 114, -36, -18, -124, 64, 7, 88], [-97, -65, 55, -36, 11, -79, -86, -118],
     [-53, -50, -63, -38, -119, 1, -19, -76], [46, 112, -99, 41, 63, 48, 6, -41]],
    [[104, -21, -40, 65, 107, 97, -92, 14], [85, -100, -3, 127, -125, 67, 24, -49],
     [-34, 55, 8, -3, -46, -48, 21, -8], [-89, -27, 10, -40, -110, -96, -93, -105]],
    [[-58, -94, 7, 2, 42, -98, -17, -88], [11, 86, -84, -7, -7, 87, 120, 127],
     [-34, -38, -109, -101, -93, 85, 115, 35], [25, -18, 106, -101, -92, -108, 116, 17]],
    [[-77, 66, 27, -36, -45, -89, -13, -82], [95, 7, 127, -51, 92, 102, -22, -46],
     [54, 28, -49, 69, 83, 118, 36, 56], [21, 100, -1, 14, 68, -98, 104, 106]],
]


def _scale(name):
    return ATTN["image"].segments[name]["scale"]


def _err(blk, xq):
    ref = export.float_attention(blk, xq * _scale("x"))
    peak = max(float(np.abs(v).max()) / _scale(k) for k, v in ref.items() if k != "x")
    assert peak <= 127.5, f"input saturates ({peak:.1f} LSB)"
    y = attention.attention_int(ATTN["image"], xq)["y"] * _scale("y")
    return np.abs(y - ref["y"]).max(axis=(-2, -1)) / _scale("y")


def _within_tol(err):
    assert err.max() <= export.ATTN_TOL_LSB, f"{(err > export.ATTN_TOL_LSB).sum()} of {err.size} inputs, max {err.max():.2f} LSB"


@pytest.mark.parametrize("i", range(len(WORST)))
def test_attention_tol_unsaturated_worst(blk, i):
    _within_tol(_err(blk, np.array(WORST[i])))


def test_worst_inputs_exceed_old_tol(blk):
    assert _err(blk, np.array(WORST[3:])).min() > 5


@pytest.mark.parametrize("c", range(export.N_EMBD))
def test_attention_tol_outlier_channel(blk, c):
    _within_tol(_err(blk, export.quantize(np.random.default_rng(99).standard_normal((N, export.T, export.N_EMBD))
                                     * np.where(np.arange(export.N_EMBD) == c, 4, 1), _scale("x"))))


@pytest.mark.parametrize("dist", ["gauss", "laplace", "student_t3"])
def test_attention_tol_fresh_distributions(blk, dist):
    rng = np.random.default_rng(2026)
    x = {"gauss": lambda: rng.standard_normal((N, export.T, export.N_EMBD)),
         "laplace": lambda: rng.laplace(size=(N, export.T, export.N_EMBD)) / np.sqrt(2),
         "student_t3": lambda: rng.standard_t(3, (N, export.T, export.N_EMBD)) / np.sqrt(3)}[dist]()
    _within_tol(_err(blk, export.quantize(x, _scale("x"))))
