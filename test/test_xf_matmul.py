"""matmul_naive and matmul_tiled vs golden.matmul_i8 (and fused requant), bit-exact."""
import cocotb
import numpy as np
import pytest

from xf import golden, kernels
from xf.image import Image
from .helpers.runner import Launch, run_launches, write_bench_result

KINDS = ("naive", "tiled")


def _case(name, a, b, mq=0, s=0):
    return name, np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64), mq, s


def _rand(rng, m, k, n):
    return rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (k, n))


async def _check(dut, cases, bench=None):
    """Run both kernels on every case from one image; assert each output equals the golden model."""
    img, launches, want = Image(), [], {}
    for name, a, b, mq, s in cases:
        img.alloc(f"{name}.a", a.shape, "i8", data=a)
        img.alloc(f"{name}.b", b.shape, "i8", data=b)
        acc = golden.matmul_i8(a, b)
        out = golden.requant(acc, mq, s) if mq else acc
        for kind in KINDS:
            c = f"{name}.{kind}"
            img.alloc(c, out.shape, "i8" if mq else "i32")
            prog, threads = kernels.matmul(img, kind, f"{name}.a", f"{name}.b", c, mq, s)
            launches.append(Launch(prog, threads, c, timeout_cycles=200_000))
            want[c] = out
    metrics, mem = await run_launches(dut, img, launches)
    bad = [c for c in want if not np.array_equal(img.read(c, mem.memory), want[c])]
    assert not bad, f"mismatch: {bad}"
    if bench:
        for m in metrics:
            write_bench_result(f"xf_matmul_{m['name'].replace('.', '_')}", m)
    return metrics


@cocotb.test()
async def test_shapes(dut):
    rng = np.random.default_rng(0)
    await _check(dut, [_case(f"s{m}x{k}x{n}", *_rand(rng, m, k, n)) for m, k, n in
                       ((4, 8, 16), (4, 16, 8), (1, 8, 8))], bench=True)


@cocotb.test()
async def test_ragged(dut):
    """K not a multiple of the 4-wide chunk, M not a multiple of the 4-row tile."""
    rng = np.random.default_rng(1)
    await _check(dut, [_case(f"r{m}x{k}x{n}", *_rand(rng, m, k, n)) for m, k, n in
                       ((4, 7, 8), (3, 10, 4), (5, 1, 4), (2, 13, 12), (6, 5, 8))])


@cocotb.test()
async def test_extremes(dut):
    full = lambda v, sh: np.full(sh, v)
    alt = np.where(np.indices((4, 16)).sum(0) % 2, 127, -128)
    await _check(dut, [
        _case("min", full(-128, (4, 16)), full(-128, (16, 8))),
        _case("max", full(127, (4, 16)), full(127, (16, 8))),
        _case("minmax", full(-128, (4, 16)), full(127, (16, 8))),
        _case("alt", alt, alt.T.copy()),
        _case("zero", full(0, (4, 8)), full(0, (8, 16))),
        _case("zero_a", full(0, (1, 8)), np.arange(-32, 32).reshape(8, 8)),
    ])


@cocotb.test()
async def test_random_seeds(dut):
    shapes = ((4, 8, 16), (4, 16, 8), (1, 8, 8), (3, 9, 4), (2, 6, 8))
    cases = []
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        cases.append(_case(f"seed{seed}", *_rand(rng, *shapes[seed % len(shapes)])))
    await _check(dut, cases)


@cocotb.test()
async def test_fused_requant(dut):
    rng = np.random.default_rng(2)
    cases = [_case(f"q{i}", *_rand(rng, 4, 16, 8), mq=int(rng.integers(1, 1 << 12)), s=int(rng.integers(8, 24)))
             for i in range(4)]
    cases += [
        _case("q_s0", *_rand(rng, 1, 8, 8), mq=1, s=0),          # no rounding term; saturates
        _case("q_sat", np.full((4, 16), -128), np.full((16, 4), 127), mq=3, s=4),
        _case("q_pos", np.full((4, 16), 127), np.full((16, 4), 127), mq=5, s=10),
        _case("q_ragged", *_rand(rng, 3, 7, 4), mq=77, s=9),
        _case("q_s31", *_rand(rng, 2, 8, 4), mq=(1 << 13) - 1, s=31),
    ]
    await _check(dut, cases)


@cocotb.test()
async def block_size_check(dut):
    """matmul_tiled runs bit-exact at a supported THREADS_PER_BLOCK and is refused by run_launches,
    before any launch, at an unsupported one (AUDIT-4 M2). Not a test_ function: the model runs those."""
    tpb = int(dut.THREADS_PER_BLOCK.value)
    rng = np.random.default_rng(7)
    a, b = _rand(rng, 5, 9, 8)
    img = Image()
    img.alloc("a", a.shape, "i8", data=a)
    img.alloc("b", b.shape, "i8", data=b)
    img.alloc("c", [5, 8], "i32")
    launch = Launch(*kernels.matmul(img, "tiled", "a", "b", "c"), "c.tiled", timeout_cycles=200_000)
    if tpb <= 16 and (tpb % 4 == 0 or 4 % tpb == 0):
        _, mem = await run_launches(dut, img, [launch])
        assert np.array_equal(img.read("c", mem.memory), golden.matmul_i8(a, b))
    else:
        with pytest.raises(ValueError, match=f"launch 'c.tiled': blockDim {tpb} unsupported"):
            await run_launches(dut, img, [launch])
