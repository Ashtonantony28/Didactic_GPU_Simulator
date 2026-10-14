"""AUDIT-2 repros: TOL on 200k held-out inputs."""
import os

import numpy as np
import pytest

from xf import attention, export, fixture

MLP = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")
ATTN = fixture.load(fixture.FIXTURE_DIR / "attention_0.json")
HELDOUT_SEED, HELDOUT_N = 12345, 200_000


def _scale(img, name):
    return img.segments[name]["scale"]


def _nanogpt():
    pytest.importorskip("torch")
    if not os.environ.get("NANOGPT_DIR"):
        pytest.skip("NANOGPT_DIR not set")
    return export.build_block(os.environ["NANOGPT_DIR"])


def _heldout(img):
    x = np.random.default_rng(HELDOUT_SEED).standard_normal((HELDOUT_N, export.T, export.N_EMBD))
    return export.quantize(x, _scale(img, "x"))


def test_mlp_tol_on_heldout():
    blk, img = _nanogpt(), MLP["image"]
    xq = _heldout(img)
    out = export.mlp_block_int(img, xq)["out"] * _scale(img, "out")
    err = np.abs(out - export.float_acts(blk, xq * _scale(img, "x"))["out"]).max(axis=(1, 2))
    assert err.max() <= MLP["tol"], f"{(err > MLP['tol']).sum()} of {len(err)} inputs, max {err.max() / _scale(img, 'out'):.2f} LSB"


def test_attention_tol_on_heldout():
    attn, img = _nanogpt().attn, ATTN["image"]
    xq = _heldout(img)
    ref = export.float_attention(attn, xq * _scale(img, "x"))["y"]
    y = attention.attention_int(img, xq)["y"] * _scale(img, "y")
    err = np.abs(y - ref).max(axis=(1, 2))
    assert err.max() <= ATTN["tol"], f"{(err > ATTN['tol']).sum()} of {len(err)} inputs, max {err.max() / _scale(img, 'y'):.2f} LSB"
