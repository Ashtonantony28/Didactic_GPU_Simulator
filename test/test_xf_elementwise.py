"""requant, relu, swiglu and residual_add kernels vs xf.golden, bit-exact."""
import cocotb
import numpy as np

from xf import elementwise as ew, fixture, golden
from xf.image import Image
from .helpers.runner import Launch, run_launches

I8 = np.arange(-128, 128)
FIXTURES = [fixture.load(fixture.FIXTURE_DIR / f"mlp_block_{i}.json") for i in range(4)]


async def check(dut, img, cases):
    """cases: [(out segment, (program, threads), golden)]; one launch per case on one image."""
    _, mem = await run_launches(dut, img, [Launch(p, t, out) for out, (p, t), _ in cases])
    bad = [out for out, _, want in cases if not np.array_equal(img.read(out, mem.memory), want)]
    assert not bad, f"mismatch: {bad}"


def _seg(img, name, data, dtype="i8"):
    data = np.asarray(data, dtype=np.int64)
    img.alloc(name, data.shape, dtype, data=data)
    return name


@cocotb.test()
async def test_requant(dut):
    rng = np.random.default_rng(0)
    img, cases = Image(), []

    def case(name, acc, m, s):
        _seg(img, f"{name}.in", acc, "i32")
        img.alloc(name, np.shape(acc), "i8")
        cases.append((name, ew.requant(img, f"{name}.in", name, m, s), golden.requant(acc, m, s)))

    for i in range(4):
        case(f"rand{i}", rng.integers(-(1 << 20), 1 << 20, (4, 16)), int(rng.integers(1, 1 << 10)), int(rng.integers(8, 24)))
    top = (1 << 30) - 1
    case("s0", rng.integers(-300, 300, 64), 1, 0)
    case("s31", [-(1 << 31), -(1 << 30), -1, 0, 1, top, top - 1, 1 << 29], 1, 31)
    case("int32_ends", [-(1 << 31), (1 << 31) - 1, -129, -128, 127, 128, 0, -1], 1, 0)
    case("ragged", rng.integers(-(1 << 16), 1 << 16, (3, 5)), 77, 9)
    case("ties", [-3, -2, -1, 1, 2, 3, 5, -5], 1, 1)   # x.5 rounds up: floor(x/2 + 1/2)
    await check(dut, img, cases)


@cocotb.test()
async def test_relu(dut):
    rng = np.random.default_rng(1)
    img, cases = Image(), []
    for name, x in (("lo", I8[:128]), ("hi", I8[128:]), ("rand", rng.integers(-128, 128, (4, 16))), ("odd", [-128, 0, 127])):
        _seg(img, f"{name}.in", x)
        img.alloc(name, np.shape(x), "i8")
        cases.append((name, ew.relu(img, f"{name}.in", name), golden.relu(x)))
    await check(dut, img, cases)


@cocotb.test()
async def test_swiglu(dut):
    rng = np.random.default_rng(2)
    img, cases = Image(), []
    fimg = FIXTURES[0]["image"]
    _seg(img, "lut_rand", rng.integers(-128, 128, 256))
    _seg(img, "lut_min", np.full(256, -128))
    _seg(img, "lut_silu", fimg.read("silu_lut"))

    def case(name, gate, up, lut, m, s):
        for part, v in (("gate", gate), ("up", up)):
            _seg(img, f"{name}.{part}", v)
        img.alloc(name, np.shape(gate), "i8")
        cases.append((name, ew.swiglu(img, f"{name}.gate", f"{name}.up", lut, name, m, s),
                      golden.swiglu_lut(gate, up, img.read(lut), m, s)))

    case("ends", [-128, 127, -128, 127, -128, 127, 0, -1], [-128, -128, 127, 127, 0, 1, -128, 127], "lut_rand", 1, 0)
    for i in range(3):
        case(f"rand{i}", rng.integers(-128, 128, (4, 16)), rng.integers(-128, 128, (4, 16)),
             "lut_rand", int(rng.integers(1, 1 << 16)), int(rng.integers(10, 24)))
    case("allgate", I8[:128], rng.integers(-128, 128, 128), "lut_rand", 3, 6)   # every LUT entry
    case("allgate_hi", I8[128:], rng.integers(-128, 128, 128), "lut_rand", 3, 6)
    case("sat", np.full(16, 127), np.full(16, -128), "lut_min", 1 << 15, 4)

    for i, fx in enumerate(FIXTURES):   # the exported LUT and constants on each fixture's gate/up
        g = fx["golden"]
        case(f"fix{i}", g["gate"], g["up"], "lut_silu", int(fimg.read("act_M")[0]), int(fimg.read("act_s")[0]))
        assert np.array_equal(cases[-1][2], g["act"])
    await check(dut, img, cases)


@cocotb.test()
async def test_residual_add(dut):
    rng = np.random.default_rng(3)
    img, cases = Image(), []

    def case(name, a, b, ma, mb, s):
        _seg(img, f"{name}.a", a)
        _seg(img, f"{name}.b", b)
        img.alloc(name, np.shape(a), "i8")
        cases.append((name, ew.residual_add(img, f"{name}.a", f"{name}.b", name, ma, mb, s),
                      golden.residual_add(a, b, ma, mb, s)))

    for i in range(4):
        case(f"rand{i}", rng.integers(-128, 128, (4, 8)), rng.integers(-128, 128, (4, 8)),
             int(rng.integers(1, 1 << 22)), int(rng.integers(1, 1 << 22)), int(rng.integers(16, 24)))
    ends = np.array(np.meshgrid([-128, -1, 0, 1, 127], [-128, -1, 0, 1, 127])).reshape(2, -1)
    case("ends", ends[0], ends[1], 1 << 20, 1 << 20, 20)
    case("sat", np.full(8, 127), np.full(8, 127), 1 << 23, 1 << 23, 23)
    case("sat_neg", np.full(8, -128), np.full(8, -128), 1 << 23, 1 << 23, 23)
    case("s0", rng.integers(-128, 128, 12), rng.integers(-128, 128, 12), 1, 1, 0)
    case("unaligned", rng.integers(-128, 128, 7), rng.integers(-128, 128, 7), 3, 1 << 20, 21)

    fimg = FIXTURES[0]["image"]
    ma, mb, s = (int(fimg.read(n)[0]) for n in ("res_Ma", "res_Mb", "res_s"))
    for i, fx in enumerate(FIXTURES):
        case(f"fix{i}", fx["inputs"]["x"], fx["golden"]["down"], ma, mb, s)
        assert np.array_equal(cases[-1][2], fx["golden"]["out"])
    await check(dut, img, cases)
