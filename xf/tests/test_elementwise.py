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


@pytest.mark.parametrize("name", ["requant", "relu", "swiglu", "residual_add", "rmsnorm", "transpose", "softmax"])
def test_prologue(name):
    assert kernels.program(name, 0x123)[1:4] == kernels.program(name, 0)[1:4]


def test_args():
    img = _img()
    assert ew.swiglu(img, "a", "b", "lut", "out", 7, 3)[1] == 32
    assert img.read("out.args").tolist() == [0, 32, 96, 64, 7, 3, 4]
    img = _img()
    ew.rmsnorm(img, "a", "w", "rlut", "out", 3, 9)
    assert img.read("out.args").tolist() == [0, 352, 360, 64, 8, 4, 1023, 3, 9, 1 << 9]
    img = _img()
    ew.residual_add(img, "a", "b", "out", 5, 6, 31)
    assert img.read("out.args").tolist()[-2:] == [31, 1 << 30]
    img = _img()
    img.alloc("t", [8, 4], "i8")
    assert ew.transpose(img, "a", "t")[1] == 32
    assert img.read("t.args").tolist() == [0, img.base("t"), 4, 8]
    img = _img()
    img.alloc("sq", [4, 4], "i8")
    img.alloc("p", [4, 4], "i8")
    assert ew.softmax(img, "sq", "lut", "p", 127)[1] == 16
    assert img.read("p.args").tolist() == [img.base("sq"), 96, img.base("p"), 4, 127]


@pytest.mark.parametrize("bind", [
    lambda img: ew.requant(img, "a", "out", 1, 32),
    lambda img: ew.relu(img, "lut", "out"),
    lambda img: ew.swiglu(img, "a", "b", "w", "out", 1, 0),
    lambda img: ew.residual_add(img, "a", "w", "out", 1, 1, 0),
    lambda img: ew.rmsnorm(img, "a", "lut", "rlut", "out", 1, 0),
    lambda img: ew.rmsnorm(img, "a", "w", "rlut", "out", 1, 27),
    lambda img: ew.rmsnorm(img, "a", "w", "lut", "out", 1, -1),
    lambda img: ew.relu(img, "rlut", "rlut"),
    lambda img: ew.transpose(img, "a", "out"),
    lambda img: ew.softmax(img, "a", "lut", "out", 127),
    lambda img: ew.softmax(img, "out", "rlut", "out", 127),
])
def test_rejects(bind):
    with pytest.raises(ValueError):
        bind(_img())
