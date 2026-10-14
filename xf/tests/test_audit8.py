"""AUDIT-8 repros: expected failures until the finding is fixed (strict, so a fix flips them)."""
import numpy as np
import pytest

from xf import elementwise as ew
from xf.image import Image


@pytest.mark.xfail(strict=True, reason="AUDIT-8 L1: rmsnorm binder accepts an i32 x; the kernel then indexes past rsqrt_lut")
def test_rmsnorm_rejects_non_i8_input():
    img = Image()
    img.alloc("x", [1, 8], "i32", data=np.full((1, 8), 1000))
    img.alloc("w", [8], "i8", data=np.ones(8, dtype=np.int64))
    img.alloc("lut", [1024], "i32", data=np.arange(1024))
    img.alloc("out", [1, 8], "i8")
    with pytest.raises(ValueError):
        ew.rmsnorm(img, "x", "w", "lut", "out", 1, 0)


@pytest.mark.xfail(strict=True, reason="AUDIT-8 L1: swiglu binder accepts an i32 gate; the kernel then indexes past silu_lut")
def test_swiglu_rejects_non_i8_gate():
    img = Image()
    img.alloc("g", [4], "i32", data=[300, 0, 0, 0])
    img.alloc("u", [4], "i8", data=[1] * 4)
    img.alloc("lut", [256], "i8")
    img.alloc("o", [4], "i8")
    with pytest.raises(ValueError):
        ew.swiglu(img, "g", "u", "lut", "o", 1, 0)
