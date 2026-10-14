"""AUDIT-6 repros: attention TOL on inputs that saturate nothing (within TOL since SG13's ln_1; SG14 re-runs the search)."""
import os

import numpy as np
import pytest

from xf import attention, export, fixture

ATTN = fixture.load(fixture.FIXTURE_DIR / "attention_0.json")
# found by hill-climbing int8 inputs under the constraint that no float activation exceeds 127.5 LSB
WORST = [[60, -18, -43, 4, -78, 28, 41, 47], [-105, -24, 30, -22, 28, -6, -37, -23],
         [6, 56, 65, -101, -21, -80, -36, -50], [-31, 83, 116, 71, 25, 38, -62, -116]]


def _scale(name):
    return ATTN["image"].segments[name]["scale"]


def _err(xq):
    pytest.importorskip("torch")
    if not os.environ.get("NANOGPT_DIR"):
        pytest.skip("NANOGPT_DIR not set")
    ref = export.float_attention(export.build_block(os.environ["NANOGPT_DIR"]), xq * _scale("x"))
    peak = max(float(np.abs(v).max()) / _scale(k) for k, v in ref.items() if k != "x")
    assert peak <= 127.5, f"input saturates ({peak:.1f} LSB)"
    y = attention.attention_int(ATTN["image"], xq)["y"] * _scale("y")
    return np.abs(y - ref["y"]).max(axis=(-2, -1))


def test_attention_tol_unsaturated_worst():
    err = _err(np.array(WORST))
    assert err <= ATTN["tol"], f"{err / _scale('y'):.2f} LSB"


def test_attention_tol_outlier_channel():
    x = np.random.default_rng(99).standard_normal((200_000, export.T, export.N_EMBD)) * np.r_[4, [1] * 7]
    err = _err(export.quantize(x, _scale("x")))
    assert err.max() <= ATTN["tol"], f"{(err > ATTN['tol']).sum()} of {len(err)} inputs, max {err.max() / _scale('y'):.2f} LSB"
