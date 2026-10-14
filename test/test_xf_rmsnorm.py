"""rmsnorm kernel vs xf.golden.rmsnorm_int, bit-exact."""
import cocotb
import numpy as np

from xf import elementwise as ew, golden
from xf.image import Image
from .test_xf_elementwise import FIXTURES, check, _seg


def _case(img, cases, name, x, w, lut, m, s):
    _seg(img, f"{name}.x", x)
    img.alloc(name, np.shape(x), "i8")
    cases.append((name, ew.rmsnorm(img, f"{name}.x", w, lut, name, m, s),
                  golden.rmsnorm_int(x, img.read(w), img.read(lut), m, s)))


@cocotb.test()
async def test_exported(dut):
    """The exported norm weight, rsqrt LUT and constants on every fixture and on edge rows."""
    img = Image.from_dict(FIXTURES[0]["image"].to_dict())
    m, s = (int(img.read(n)[0]) for n in ("norm_M", "norm_s"))
    cases = []
    for i, fx in enumerate(FIXTURES):
        _case(img, cases, f"fix{i}", fx["inputs"]["x"], "norm_w", "rsqrt_lut", m, s)
        assert np.array_equal(cases[-1][2], fx["golden"]["h"])
    d = img.segments["x"]["shape"][1]
    one_hot = np.zeros((4, d), dtype=np.int64)
    one_hot[np.arange(4), [0, 3, 5, d - 1]] = [1, -1, 127, -128]
    edges = {
        "zero": np.zeros((4, d)),                                    # sum 0 -> lut[0]
        "min": np.full((4, d), -128),                                 # most normalization steps
        "max": np.full((4, d), 127),
        "mixed": [[0] * d, [-128] * d, [127] * d, [-128, 127] * (d // 2)],
        "one_hot": one_hot,
        "quiet": [[1] + [0] * (d - 1), [31] + [1] * (d - 1), [32] + [0] * (d - 1), [4, -2, 4, 5, -2, -2, 5, -2]],
    }
    for name, x in edges.items():
        _case(img, cases, name, x, "norm_w", "rsqrt_lut", m, s)
    await check(dut, img, cases)


@cocotb.test()
async def test_synthetic(dut):
    """Random LUTs, widths and constants, including d not a multiple of the block and saturation."""
    rng = np.random.default_rng(5)
    img, cases = Image(), []
    for i, (t, d, n) in enumerate(((4, 8, 64), (3, 5, 16), (5, 3, 8), (2, 16, 1024), (1, 1, 4))):
        _seg(img, f"w{i}", rng.integers(-128, 128, d))
        lut = rng.integers(0, 1 << 10, n)
        lut[[0, -1]] = [(1 << 10) - 1, 0]
        _seg(img, f"lut{i}", lut, "i32")
        m, s = int(rng.integers(1, 1 << 6)), int(rng.integers(12, 20))
        x = rng.integers(-128, 128, (t, d)) >> rng.integers(0, 8, (t, 1))   # rows of every loudness
        _case(img, cases, f"rand{i}", x, f"w{i}", f"lut{i}", m, s)
        _case(img, cases, f"zero{i}", np.zeros((t, d)), f"w{i}", f"lut{i}", m, s)
        _case(img, cases, f"ext{i}", np.where(rng.random((t, d)) < 0.5, -128, 127), f"w{i}", f"lut{i}", m, s)
    _seg(img, "w_sat", [-128, 127, -128, 127])
    _seg(img, "lut_sat", [(1 << 17) - 1, 1 << 16, 1 << 15, 1 << 14], "i32")
    _case(img, cases, "sat", [[-128, -128, 127, 127], [1, -1, 1, -1]], "w_sat", "lut_sat", 1, 0)
    await check(dut, img, cases)
