import pytest

from xf import elementwise as ew, kernels
from xf.image import Image


def _img():
    img = Image()
    for name in ("a", "b", "out"):
        img.alloc(name, [4, 8], "i8")
    img.alloc("lut", [256], "i8")
    img.alloc("w", [8], "i8")
    img.alloc("rlut", [1024], "i32")
    return img


@pytest.mark.parametrize("name", ["requant", "relu", "swiglu", "residual_add", "rmsnorm"])
def test_prologue(name):
    assert kernels.program(name, 0x123)[1:4] == kernels.program(name, 0)[1:4]


def test_args():
    img = _img()
    assert ew.swiglu(img, "a", "b", "lut", "out", 7, 3)[1] == 32
    assert img.read("out.args").tolist() == [0, 32, 96, 64, 7, 3, 4]
    img = _img()
    ew.rmsnorm(img, "a", "w", "rlut", "out", 7, 1, 0)
    assert img.read("out.args").tolist() == [0, 352, 360, 64, 8, 7, 1023, 1, 0, 0]
    img = _img()
    ew.residual_add(img, "a", "b", "out", 5, 6, 31)
    assert img.read("out.args").tolist()[-2:] == [31, 1 << 30]


@pytest.mark.parametrize("bind", [
    lambda img: ew.requant(img, "a", "out", 1, 32),
    lambda img: ew.relu(img, "lut", "out"),
    lambda img: ew.swiglu(img, "a", "b", "w", "out", 1, 0),
    lambda img: ew.residual_add(img, "a", "w", "out", 1, 1, 0),
    lambda img: ew.rmsnorm(img, "a", "lut", "rlut", "out", 7, 1, 0),
    lambda img: ew.rmsnorm(img, "a", "w", "rlut", "out", 32, 1, 0),
    lambda img: ew.relu(img, "rlut", "rlut"),
])
def test_rejects(bind):
    with pytest.raises(ValueError):
        bind(_img())
