"""AUDIT-1 repros: expected failures until the finding's sub-goal lands (strict, so a fix flips them)."""
import os

import numpy as np
import pytest

from xf import export, fixture

IMG = fixture.load(fixture.FIXTURE_DIR / "mlp_block_0.json")["image"]
# input 1666 of rng(7).standard_normal((2000, 4, 8)) quantized at scale(x): row 0 has sum(x^2) = 98
QUIET = [[4, -2, 4, 5, -2, -2, 5, -2], [-31, -35, -4, 56, 35, 2, 3, 28],
         [-50, 47, -56, 11, -30, 51, 16, 35], [-50, -7, 34, -39, 43, 10, -1, 59]]


def rmsnorm_ref(x):
    """nanoGPT RMSNorm on dequant(x) with the quantized weight, in h units."""
    sx, sh, sw = (IMG.segments[n]["scale"] for n in ("x", "h", "norm_w"))
    xf = np.asarray(x) * sx
    return xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + export.EPS) * IMG.read("norm_w") * sw / sh


@pytest.mark.xfail(strict=True, reason="AUDIT-1 H1: bucketed rsqrt LUT is not scale-invariant on quiet rows (SG9)")
@pytest.mark.parametrize("x", [QUIET, [[1] + [0] * 7, [2] + [0] * 7, [4] + [0] * 7, [6, -6, 6, -6, 6, -6, 6, -6]]],
                         ids=["quiet_token", "single_element"])
def test_rmsnorm_matches_reference_on_quiet_rows(x):
    h = export.mlp_block_int(IMG, np.asarray(x))["h"]
    assert np.abs(h - np.clip(rmsnorm_ref(x), -128, 127)).max() <= 2


@pytest.mark.xfail(strict=True, reason="AUDIT-1 H1: MLP block exceeds TOL on in-distribution inputs (SG9)")
def test_mlp_block_within_tol_on_fresh_seed():
    pytest.importorskip("torch")
    if not os.environ.get("NANOGPT_DIR"):
        pytest.skip("NANOGPT_DIR not set")
    img, _, _ = export.export(os.environ["NANOGPT_DIR"])
    blk = export.build_block(os.environ["NANOGPT_DIR"])
    sx, so = img.segments["x"]["scale"], img.segments["out"]["scale"]
    xq = export.quantize(np.random.default_rng(7).standard_normal((2000, export.T, export.N_EMBD)), sx)
    ref = export.float_acts(blk, xq * sx)["out"]
    worst = max(np.abs(export.mlp_block_int(img, x)["out"] * so - r).max() / so for x, r in zip(xq, ref))
    assert worst <= export.TOL_LSB, f"max error {worst:.2f} LSB"
