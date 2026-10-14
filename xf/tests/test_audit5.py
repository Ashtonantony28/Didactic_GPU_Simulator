"""AUDIT-5 H1: with ln_1 in front, the attention head stays within TOL at every input loudness, like the MLP block."""
import os

import numpy as np
import pytest

from xf import attention, export, fixture

MLP = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")["image"]
ATTN = fixture.load(fixture.FIXTURE_DIR / "attention_0.json")["image"]


@pytest.fixture(scope="module")
def blk():
    pytest.importorskip("torch")
    if not os.environ.get("NANOGPT_DIR"):
        pytest.skip("NANOGPT_DIR not set")
    return export.build_block(os.environ["NANOGPT_DIR"])


def _worst(img, a, golden, ref, out):
    sx, so = img.segments["x"]["scale"], img.segments[out]["scale"]
    xq = export.quantize(np.random.default_rng(7).standard_normal((2000, export.T, export.N_EMBD)) * a, sx)
    return np.abs(golden(img, xq)[out] * so - ref(xq * sx)[out]).max() / so


@pytest.mark.parametrize("a", [1.5, 2.0, 3.0])
def test_mlp_block_within_tol_on_louder_inputs(blk, a):
    worst = _worst(MLP, a, export.mlp_block_int, lambda x: export.float_acts(blk, x), "out")
    assert worst <= export.TOL_LSB, f"max error {worst:.2f} LSB"


@pytest.mark.parametrize("a", [0.05, 0.1, 0.3, 1.0, 2.0, 3.0, 10.0])
def test_attention_within_tol_across_input_scales(blk, a):
    worst = _worst(ATTN, a, attention.attention_int, lambda x: export.float_attention(blk, x), "y")
    assert worst <= export.ATTN_TOL_LSB, f"max error {worst:.2f} LSB"


@pytest.mark.parametrize("a", [0.05, 0.1])
def test_attention_not_vacuous_on_quiet_inputs(blk, a):
    sx, sy = ATTN.segments["x"]["scale"], ATTN.segments["y"]["scale"]
    xq = export.quantize(np.random.default_rng(7).standard_normal((2000, export.T, export.N_EMBD)) * a, sx)
    zero_passes = np.abs(export.float_attention(blk, xq * sx)["y"]).max(axis=(1, 2)) <= export.ATTN_TOL_LSB * sy
    assert zero_passes.mean() < 0.01, f"an all-zero y passes {zero_passes.mean():.1%} of inputs"
