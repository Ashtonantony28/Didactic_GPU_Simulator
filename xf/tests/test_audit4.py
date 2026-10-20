"""AUDIT-4 repros (L3, L4, M2), fixed in SG22."""
import numpy as np
import pytest

from xf import elementwise as ew, golden, kernels
from xf.asm import AsmError, assemble
from xf.image import Image
from xf.isasim import run


def test_br_without_condition_rejected():
    with pytest.raises(AsmError):
        assemble("top: BR top\nRET")


@pytest.mark.parametrize("src", [
    "BRnzp 200\nRET",
    "BRz 2\nRET",                       # one past the last instruction
    "BRn end\nRET\nend:",               # label after the last instruction
    "BRnzp -1\nRET",
    "SYNC 2\nRET",
    "SYNC end\nBRz end\nRET\nend:",
])
def test_br_target_outside_program_rejected(src):
    with pytest.raises(AsmError, match="outside"):
        assemble(src)


def test_br_target_last_instruction_accepted():
    assert assemble("BRz 1\nRET") == [0x1401, 0xF000]
    assert assemble("SYNC end\nBRnzp end\nend: RET")[:2] == [0x1002, 0x1E02]


def _img(shapes, dtype="i8"):
    img = Image()
    for name, shape in shapes.items():
        img.alloc(name, shape, dtype if name != "lut" else "i32")
    return img


@pytest.mark.parametrize("kind", ["naive", "tiled"])
@pytest.mark.parametrize("m,k,n", [(4, 0, 4), (0, 4, 4), (4, 4, 0)])
def test_matmul_rejects_empty(kind, m, k, n):
    img = _img({"a": [m, k], "b": [k, n], "c": [m, n]})
    with pytest.raises(ValueError, match="empty"):
        kernels.matmul(img, kind, "a", "b", "c")


@pytest.mark.parametrize("bind,shapes", [
    (lambda img: ew.rmsnorm(img, "x", "w", "lut", "o", 1, 0), {"x": [2, 0], "w": [0], "lut": [16], "o": [2, 0]}),
    (lambda img: ew.rmsnorm(img, "x", "w", "lut", "o", 1, 0), {"x": [0, 8], "w": [8], "lut": [16], "o": [0, 8]}),
    (lambda img: ew.softmax(img, "x", "lut", "o", 127), {"x": [0, 0], "lut": [256], "o": [0, 0]}),
    (lambda img: ew.relu(img, "x", "o"), {"x": [0], "o": [0]}),
    (lambda img: ew.transpose(img, "x", "o"), {"x": [0, 4], "o": [4, 0]}),
    (lambda img: ew.residual_add(img, "x", "w", "o", 1, 1, 0), {"x": [0], "w": [0], "o": [0]}),
])
def test_elementwise_rejects_empty(bind, shapes):
    with pytest.raises(ValueError, match="empty"):
        bind(_img(shapes))


def _tiled(m, k, n, mq, s, seed):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (k, n))
    want = golden.matmul_i8(a, b)
    img = _img({"a": [m, k], "b": [k, n]})
    img.write("a", a)
    img.write("b", b)
    img.alloc("c", [m, n], "i8" if mq else "i32")
    prog, threads = kernels.matmul(img, "tiled", "a", "b", "c", mq, s)
    return img, prog, threads, golden.requant(want, mq, s) if mq else want


@pytest.mark.parametrize("tpb,warps", [(1, 1), (2, 1), (2, 2), (4, 1), (8, 1), (8, 2), (12, 3), (16, 1), (16, 4)])
@pytest.mark.parametrize("m,k,n,mq,s", [(4, 8, 16, 0, 0), (3, 10, 4, 0, 0), (5, 1, 4, 7, 4), (6, 5, 8, 0, 0),
                                        (9, 7, 12, 3, 2), (1, 16, 16, 0, 0), (2, 13, 12, 5, 6)])
def test_matmul_tiled_any_block_size(tpb, warps, m, k, n, mq, s):
    """M2: block-size generic, bit-exact, every branch uniform, no conflicting stores."""
    img, prog, threads, want = _tiled(m, k, n, mq, s, seed=m * 100 + k * 10 + n)
    kernels.check_block(prog.max_block, tpb)
    for policy in ("lrr", "gto"):
        r = run(prog, img.words, threads, word_bits=32, addr_bits=12, threads_per_block=tpb,
                warps_per_core=warps, warp_policy=policy)
        assert not r.divergent and not r.stores_conflict
        assert np.array_equal(img.read("c", r.memory), want)


@pytest.mark.parametrize("tpb", [3, 5, 6, 20, 32, 64])
def test_matmul_tiled_refuses_block_size(tpb):
    """M2: block sizes the kernel cannot run are refused by name, never run to wrong numbers."""
    _, prog, _, _ = _tiled(4, 8, 8, 0, 0, seed=0)
    assert prog.max_block == kernels.TILED_MAX_BLOCK == 16
    with pytest.raises(ValueError, match=f"launch 'c.tiled': blockDim {tpb} unsupported"):
        kernels.check_block(prog.max_block, tpb, "c.tiled")


@pytest.mark.parametrize("tpb", [1, 3, 32, 64, 255])
def test_other_kernels_any_block_size(tpb):
    img = _img({"a": [2, 3], "b": [3, 4], "c": [2, 4], "x": [2, 4], "o": [2, 4]})
    progs = [kernels.matmul(img, "naive", "a", "b", "c")[0], ew.relu(img, "x", "o")[0]]
    for p in progs:
        assert p.max_block is None
        kernels.check_block(p.max_block, tpb)


def test_launch_takes_max_block_from_program():
    from test.helpers.runner import Launch
    _, prog, threads, _ = _tiled(4, 8, 8, 0, 0, seed=0)
    assert Launch(prog, threads).max_block == 16
    assert Launch(prog, threads, max_block=8).max_block == 8
    assert Launch(list(prog), threads).max_block is None
    naive = kernels.matmul(_img({"a": [2, 2], "b": [2, 4], "c": [2, 4]}), "naive", "a", "b", "c")
    assert Launch(*naive).max_block is None
