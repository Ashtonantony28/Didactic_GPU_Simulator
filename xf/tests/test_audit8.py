"""AUDIT-8 repros (L1, L2), fixed in SG22."""
import numpy as np
import pytest

from xf import elementwise as ew, kernels
from xf.image import Image


def test_rmsnorm_rejects_non_i8_input():
    img = Image()
    img.alloc("x", [1, 8], "i32", data=np.full((1, 8), 1000))
    img.alloc("w", [8], "i8", data=np.ones(8, dtype=np.int64))
    img.alloc("lut", [1024], "i32", data=np.arange(1024))
    img.alloc("out", [1, 8], "i8")
    with pytest.raises(ValueError, match="x must be i8"):
        ew.rmsnorm(img, "x", "w", "lut", "out", 1, 0)


def test_swiglu_rejects_non_i8_gate():
    img = Image()
    img.alloc("g", [4], "i32", data=[300, 0, 0, 0])
    img.alloc("u", [4], "i8", data=[1] * 4)
    img.alloc("lut", [256], "i8")
    img.alloc("o", [4], "i8")
    with pytest.raises(ValueError, match="g must be i8"):
        ew.swiglu(img, "g", "u", "lut", "o", 1, 0)


BINDERS = {
    "matmul_naive": (lambda img: kernels.matmul(img, "naive", "a", "b", "o"), ("a", "b")),
    "matmul_tiled": (lambda img: kernels.matmul(img, "tiled", "a", "b", "o"), ("a", "b")),
    "swiglu": (lambda img: ew.swiglu(img, "a", "b", "lut8", "o", 1, 0), ("a", "b")),
    "residual_add": (lambda img: ew.residual_add(img, "a", "b", "o", 1, 1, 0), ("a", "b")),
    "rmsnorm": (lambda img: ew.rmsnorm(img, "a", "w", "lut32", "o", 1, 0), ("a", "w")),
    "relu": (lambda img: ew.relu(img, "a", "o"), ("a",)),
    "softmax": (lambda img: ew.softmax(img, "a", "lut32", "o", 127), ("a",)),
}


def _img(i32=()):
    img = Image()
    for name, shape in (("a", [4, 4]), ("b", [4, 4]), ("w", [4]), ("o", [4, 4])):
        img.alloc(name, shape, "i32" if name in i32 else "i8")
    img.alloc("lut8", [256], "i8")
    img.alloc("lut32", [256], "i32")
    return img


@pytest.mark.parametrize("binder,operand", [(b, o) for b, (_, ops) in BINDERS.items() for o in ops])
def test_binders_require_i8_operands(binder, operand):
    bind = BINDERS[binder][0]
    bind(_img())
    with pytest.raises(ValueError, match=f"{operand} must be i8"):
        bind(_img(i32={operand}))


@pytest.mark.parametrize("binder", ["matmul_naive", "matmul_tiled", "requant", "residual_add"])
def test_i32_outputs_still_accepted(binder):
    bind = {"requant": lambda img: ew.requant(img, "o", "b", 1, 0), **{k: v[0] for k, v in BINDERS.items()}}[binder]
    bind(_img(i32={"o"}))


def test_rmsnorm_header_matches_binder():
    from xf.kernels import KERNEL_DIR
    assert "d, max(kmax, 1), lutmax" in (KERNEL_DIR / "rmsnorm.asm").read_text()
    img = Image(addr_bits=16)
    img.alloc("x", [1, 1], "i8")
    img.alloc("w", [1], "i8")
    img.alloc("lut", [1 << 15], "i32")   # len > d*128^2: kmax = 0
    img.alloc("o", [1, 1], "i8")
    ew.rmsnorm(img, "x", "w", "lut", "o", 1, 0)
    assert img.read("o.args")[5] == 1
