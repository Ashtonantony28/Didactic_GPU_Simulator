import pytest

from xf import kernels
from xf.asm import disassemble
from xf.image import Image


@pytest.mark.parametrize("name", ["matmul_naive", "matmul_tiled"])
@pytest.mark.parametrize("base", [0, 255, 256, 4095])
def test_args_prologue(name, base):
    words = kernels.program(name, base)
    hi, lo = (int(disassemble([words[i]])[0].split("#")[1]) for i in (0, 4))
    assert hi * 256 + lo == base
    assert words[1:4] + words[5:] == kernels.program(name, 0)[1:4] + kernels.program(name, 0)[5:]


def _img(m, k, n):
    img = Image()
    img.alloc("a", [m, k], "i8")
    img.alloc("b", [k, n], "i8")
    img.alloc("c", [m, n], "i32")
    return img


def test_matmul_binding():
    img = _img(5, 3, 8)
    _, threads = kernels.matmul(img, "tiled", "a", "b", "c", mq=7, s=4)
    assert threads == 16
    assert img.read("c.args").tolist() == [0, 15, 39, 5, 3, 8, 7, 4, 8]
    assert kernels.matmul(_img(5, 3, 8), "naive", "a", "b", "c")[1] == 40


@pytest.mark.parametrize("kind,shape,kw", [
    ("tiled", (4, 4, 6), {}),           # N % 4
    ("naive", (16, 4, 16), {}),          # 256 threads
    ("naive", (4, 4, 4), {"s": 32}),
    ("dense", (4, 4, 4), {}),
])
def test_matmul_rejects(kind, shape, kw):
    with pytest.raises(ValueError):
        kernels.matmul(_img(*shape), kind, "a", "b", "c", **kw)
