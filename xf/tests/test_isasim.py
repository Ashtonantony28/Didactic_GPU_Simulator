"""xf/isasim.py: every cocotb kernel test's memory result, every xf kernel vs xf.golden, directed semantics."""
import asyncio
import re
from types import SimpleNamespace

import numpy as np
import pytest

from test import (test_conv1d, test_divergent_relu, test_isa_ext, test_l1d_indepth, test_matadd, test_matmul,
                  test_relu, test_tiled_matmul, test_transpose, test_vector_dot, test_xf_attention,
                  test_xf_elementwise, test_xf_matmul, test_xf_mlp_block, test_xf_rmsnorm)
from xf import golden, kernels
from xf.asm import assemble
from xf.image import Image
from xf.isa import decode, encode
from xf.isasim import IsaSimError, run
from xf.tests.legacy import hand_encoded

LEGACY = hand_encoded()
X32 = dict(word_bits=32, addr_bits=12)


def _asm(text):
    """assemble() with `SYNC target` lowered to encode("BR", nzp=0, imm=target)."""
    a, b = (assemble(re.sub(r"\bSYNC\b", br, text)) for br in ("BRn", "BRz"))
    return [encode("BR", nzp=0, imm=x & 0xFF) if x != y else x for x, y in zip(a, b)]


def _mem(prog, data, threads, **kw):
    return run(prog, data, threads, **kw).memory


# 1. Hand-encoded cocotb kernel tests: (program, data, threads, {base: expected words})
def _l1d(w, a):
    mask, k = (1 << w) - 1, 0xA5C3E1F7 & ((1 << w) - 1)
    raw = [(0x7FFFFFF0 + i) & mask for i in range(4)]
    s = [(0x5A3C1E10 + i) & mask for i in range(4)]
    cores, out = 2, test_l1d_indepth.OUT
    p = "test_l1d_indepth."
    return {
        f"l1d_hot{w}": (LEGACY[p + "test_hot_address_hits"], [7] + [0] * 31, 4, {16: [21] * 4}, w, a),
        f"l1d_raw{w}": (LEGACY[p + "test_read_after_write_same_thread"], raw + [0] * 4, 4,
                        {0: [(v + 1) & mask for v in raw], 8: [(v + 1) & mask for v in raw]}, w, a),
        f"l1d_later_block{w}": (LEGACY[p + "test_write_read_by_later_block"], s + [0] * 3 + [k] + [0] * 12, 12,
                                {0: [k] * 4, out: s, out + 4 * cores: [k] * 4}, w, a),
        f"l1d_disable{w}": (LEGACY[p + "test_l1d_disable_equivalence"], [10, 20, 30, 40, 0, 0, 0, 0], 4,
                            {8: [10, 20, 30, 40]}, w, a),
    }


def _legacy_cases():
    m = test_matadd.DATA
    cases = {
        "matadd": (test_matadd.PROGRAM, m, 8, {16: [x + y for x, y in zip(m[:8], m[8:])]}),
        "matmul": (test_matmul.PROGRAM, test_matmul.DATA, 4, {8: [7, 10, 15, 22]}),
        "vector_dot": (test_vector_dot.PROGRAM, test_vector_dot.DATA, 8,
                       {17: [a * b for a, b in zip(test_vector_dot.A, test_vector_dot.B)]}),
        "relu": (LEGACY["test_relu.test_relu"], test_relu.DATA, 8, {8: test_relu.DATA}),
        "conv1d": (test_conv1d.PROGRAM, test_conv1d.DATA, 6, {16: [sum(test_conv1d.SIGNAL[i:i + 3]) for i in range(6)]}),
        "transpose": (test_transpose.PROGRAM, test_transpose.DATA, 8, {8: [1, 5, 2, 6, 3, 7, 4, 8]}),
        "divergent_relu": (test_divergent_relu.PROGRAM, test_divergent_relu.DATA, 8,
                           {8: [v + 1 if i % 2 == 0 else v - 1 for i, v in enumerate(test_divergent_relu.DATA)]}),
        "tiled_matmul": (test_tiled_matmul.PROGRAM, test_tiled_matmul.DATA, 4, {8: [19, 22, 43, 50]}),
        "edge_single_thread": (LEGACY["test_edge_cases.test_single_thread"], [0] * 16, 1, {0: [8]}),
        "edge_two_blocks": (LEGACY["test_edge_cases.test_two_blocks_one_core"], [0] * 32, 16, {16: list(range(16))}),
        "edge_pure_compute": (LEGACY["test_edge_cases.test_pure_compute"], [0] * 16, 1, {0: [9]}),
        "edge_unaligned": (LEGACY["test_edge_cases.test_unaligned_block"], [0] * 24, 3, {16: [0, 1, 2]}),
        "smem_roundtrip": (LEGACY["test_shared_mem_indepth.test_smem_roundtrip"], [0] * 16, 4, {0: [1, 2, 3, 4]}),
        "smem_cross": (LEGACY["test_shared_mem_indepth.test_smem_cross_thread_read"], [0] * 32, 4,
                       {16: [(i + 1) % 4 + 10 for i in range(4)]}),
        "smem_persistence": (LEGACY["test_shared_mem_indepth.test_smem_persistence"], [0] * 16, 1, {0: [84]}),
        "stress_long_loop": (LEGACY["test_stress.test_long_loop"], [0] * 32, 4, {16: [28] * 4}),
        "stress_many_threads": (LEGACY["test_stress.test_many_threads"], [0] * 128, 16, {64: [2 * i for i in range(16)]}),
    }
    cases = {k: (*v, 8, 8) for k, v in cases.items()}
    cases.update(_l1d(8, 8))
    cases.update(_l1d(32, 12))
    return cases


LEGACY_CASES = _legacy_cases()


@pytest.mark.parametrize("name", LEGACY_CASES)
def test_cocotb_kernel(name):
    prog, data, threads, want, w, a = LEGACY_CASES[name]
    mem = _mem(prog, data, threads, word_bits=w, addr_bits=a)
    for base, words in want.items():
        assert mem[base:base + len(words)] == list(words), f"{name} @ {base}"


@pytest.mark.parametrize("w,a", [(8, 8), (32, 12)])
def test_isa_ext_sra_max(w, a):
    pairs = test_isa_ext._operand_pairs(w)
    n, mask, sg = len(pairs), (1 << w) - 1, test_isa_ext._signed
    mem = _mem(assemble(test_isa_ext.SRA_MAX.format(n=n)), [x for x, _ in pairs] + [y for _, y in pairs], n,
               word_bits=w, addr_bits=a)
    assert mem[2 * n:3 * n] == [(sg(x, w) >> y) & mask for x, y in pairs]
    assert mem[3 * n:4 * n] == [max(sg(x, w), sg(y, w)) & mask for x, y in pairs]


@pytest.mark.parametrize("w,a", [(8, 8), (32, 12)])
def test_isa_ext_cmp(w, a):
    mask = (1 << w) - 1
    vals = [0, 1, 2, mask, 1 << (w - 1), (1 << (w - 1)) - 1]
    pairs = [(x, y) for x in vals for y in vals]
    n = len(pairs)
    r = run(assemble(test_isa_ext.CMP_FLAGS.format(n=n)), [x for x, _ in pairs] + [y for _, y in pairs], 4 * n,
            word_bits=w, addr_bits=a)
    assert r.memory[2 * n:3 * n] == [0b100 if x != y else 0b010 for x, y in pairs]
    assert not r.divergent


# 2. xf kernels: the cocotb tests themselves, with run_launches swapped for the model
async def _model_launches(dut, image, launches, mem_latency=0):
    words, metrics = list(image.words), []
    for i, ln in enumerate(launches):
        r = run(ln.program, words, ln.threads, word_bits=image.word_bits, addr_bits=image.addr_bits)
        assert not r.divergent and not r.stores_conflict, (ln.name, r.divergent, r.stores_conflict)
        words = r.memory
        metrics.append({"launch": i, "name": ln.name, "steps": r.steps})
    return metrics, SimpleNamespace(memory=words)


XF_MODS = (test_xf_matmul, test_xf_elementwise, test_xf_rmsnorm, test_xf_attention, test_xf_mlp_block)
XF_TESTS = [(m, n) for m in XF_MODS for n in sorted(vars(m)) if n.startswith("test_") and hasattr(getattr(m, n), "func")]


@pytest.mark.parametrize("mod,name", XF_TESTS, ids=[f"{m.__name__.split('.')[-1]}.{n}" for m, n in XF_TESTS])
def test_xf_kernel(monkeypatch, mod, name):
    for m in XF_MODS:
        monkeypatch.setattr(m, "run_launches", _model_launches, raising=False)
    asyncio.run(getattr(mod, name).func(None))


# 3. Directed semantics
@pytest.mark.parametrize("w,a", [(8, 8), (32, 12)])
def test_cmp_flags(w, a):
    mask = (1 << w) - 1
    pairs = [(x, y) for x in (0, 1, mask) for y in (0, 1, mask)]
    n = len(pairs)
    mem = _mem(assemble(test_isa_ext.CMP_FLAGS.format(n=n)), [x for x, _ in pairs] + [y for _, y in pairs], n,
               word_bits=w, addr_bits=a, threads_per_block=1)
    assert mem[2 * n:3 * n] == [2, 4, 4, 4, 2, 4, 4, 4, 2]


@pytest.mark.parametrize("w,a", [(8, 8), (32, 12)])
def test_sra_negative(w, a):
    mask = (1 << w) - 1
    pairs = [(v & mask, s) for v in (-1, -2, -(1 << (w - 1)), -96) for s in (0, w - 1, w, 255)]
    pairs.append((((1 << (w - 1)) - 1), w))   # positive >> W is 0
    n = len(pairs)
    mem = _mem(assemble(test_isa_ext.SRA_MAX.format(n=n)), [x for x, _ in pairs] + [y for _, y in pairs], n,
               word_bits=w, addr_bits=a)
    want = [x if s == 0 else mask for x, s in pairs[:-1]] + [0]
    assert mem[2 * n:3 * n] == want


def test_div_by_zero():
    prog = assemble("""
        CONST R1, #7
        CONST R2, #0
        CONST R3, #2
        DIV R4, R1, R2
        DIV R5, R1, R3
        DIV R6, R2, R2
        CONST R0, #0
        STR R0, R4
        CONST R0, #1
        STR R0, R5
        CONST R0, #2
        STR R0, R6
        RET""")
    assert _mem(prog, [9, 9, 9], 1)[:3] == [0, 3, 0]


def test_readonly_registers():
    prog = [encode("CONST", rd=r, imm=99) for r in (13, 14, 15)] + [encode("ADD", rd=15, rs=13, rt=14)] + assemble("""
        MUL R0, %blockIdx, %blockDim
        ADD R0, R0, %threadIdx
        STR R0, %blockIdx
        CONST R1, #16
        ADD R1, R1, R0
        STR R1, %blockDim
        ADD R1, R1, R1
        SUB R1, R1, R0
        STR R1, %threadIdx      ; 32 + i
        RET""")
    mem = _mem(prog, [], 8)
    assert mem[:8] == [0] * 4 + [1] * 4
    assert mem[16:24] == [4] * 8
    assert mem[32:40] == [0, 1, 2, 3] * 2


@pytest.mark.parametrize("simt", [False, True])
def test_tail_block(simt):
    prog = assemble("""
        MUL R0, %blockIdx, %blockDim
        ADD R0, R0, %threadIdx
        CONST R1, #16
        ADD R2, R1, R0
        STR R2, %blockDim
        CONST R1, #32
        ADD R2, R1, R0
        STR R2, %threadIdx
        RET""")
    mem = _mem(prog, [0xEE] * 48, 6, simt=simt)
    assert mem[16:24] == [4] * 6 + [0xEE] * 2
    assert mem[32:40] == [0, 1, 2, 3, 0, 1, 0xEE, 0xEE]


def test_smem_highest_lane_wins():
    prog = assemble("""
        CONST R0, #0
        LDS R1, R0              ; SMEM is zero at block start
        CONST R2, #1
        ADD R3, %threadIdx, R2
        STS R0, R3              ; every lane: SMEM[0] = t + 1
        LDS R4, R0
        CONST R5, #17
        STS R5, R3              ; 17 & 15 = 1
        LDS R7, R2
        MUL R8, %blockIdx, %blockDim
        ADD R8, R8, %threadIdx
        STR R8, R1
        CONST R9, #16
        ADD R9, R9, R8
        STR R9, R4
        ADD R9, R9, R9
        SUB R9, R9, R8          ; 32 + i
        STR R9, R7
        RET""")
    mem = _mem(prog, [], 7, smem_size=16)
    assert mem[:7] == [0] * 7
    assert mem[16:23] == [4, 4, 4, 4, 3, 3, 3]
    assert mem[32:39] == [4, 4, 4, 4, 3, 3, 3]


def test_store_conflict():
    prog = assemble("CONST R0, #0\nSTR R0, %threadIdx\nCONST R1, #5\nSTR R1, R1\nRET")
    r = run(prog, [], 4)
    assert r.memory[0] == 3 and r.memory[5] == 5
    assert r.stores_conflict == [(0, 1, 0)]


M1 = """
    {sync}
    CONST R1, #0
    CMP %threadIdx, R1
    BRn taken
    CONST R2, #1
    STR %threadIdx, R2
    RET
taken:
    CONST R2, #2
    STR %threadIdx, R2
    RET
done:
    RET
"""


def test_audit4_m1():
    plain, sync = _asm(M1.format(sync="")), _asm(M1.format(sync="SYNC done"))
    assert sync[0] == encode("BR", nzp=0, imm=len(sync) - 1)
    r = run(plain, [0] * 8, 4)
    assert r.memory[:4] == [1, 1, 1, 1] and r.divergent == [(0, 2)]
    r = run(sync, [0] * 8, 4)
    assert r.memory[:4] == [1, 1, 1, 1] and r.divergent == [(0, 3)]
    assert run(sync, [0] * 8, 4, simt=True).memory[:4] == [1, 2, 2, 2]
    with pytest.raises(IsaSimError, match="without SYNC"):
        run(plain, [0] * 8, 4, simt=True)


NESTED = """
    SYNC outer
    CONST R0, #0
    CONST R9, #2
    DIV R1, %threadIdx, R9      ; hi = t / 2
    MUL R2, R1, R9
    SUB R2, %threadIdx, R2      ; odd = t % 2
    CONST R3, #10
    CMP R1, R0
    BRn hi
    SYNC lo_end
    CMP R2, R0
    BRn lo_odd
    CONST R4, #1
    ADD R3, R3, R4
    BRnzp lo_end
lo_odd:
    CONST R4, #2
    ADD R3, R3, R4
lo_end:
    CONST R4, #16
    ADD R3, R3, R4
    BRnzp outer
hi:
    SYNC hi_end
    CMP R2, R0
    BRn hi_odd
    CONST R4, #4
    ADD R3, R3, R4
    BRnzp hi_end
hi_odd:
    CONST R4, #8
    ADD R3, R3, R4
hi_end:
    CONST R4, #32
    ADD R3, R3, R4
outer:
    CONST R4, #64
    ADD R3, R3, R4
    STR %threadIdx, R3
    RET
"""


def test_nested_divergence():
    prog = _asm(NESTED)
    assert run(prog, [], 4, simt=True, stack_depth=4).memory[:4] == [10 + 1 + 16 + 64, 10 + 2 + 16 + 64,
                                                                     10 + 4 + 32 + 64, 10 + 8 + 32 + 64]
    r = run(prog, [], 4)
    assert r.memory[:4] == [91] * 4 and r.divergent == [(0, 8), (0, 11)]
    with pytest.raises(IsaSimError, match="deeper than 3"):
        run(prog, [], 4, simt=True, stack_depth=3)


LOOP = """
    CONST R0, #0
    CONST R1, #1
    ADD R2, %threadIdx, R1      ; trips = t + 1
    CONST R3, #0
loop:
    SYNC done
    ADD R3, R3, R2
    SUB R2, R2, R1
    CMP R2, R0
    BRn loop
done:
    STR %threadIdx, R3
    RET
"""


def test_loop_per_lane_trips():
    prog = _asm(LOOP)
    assert run(prog, [], 4, simt=True, stack_depth=6).memory[:4] == [1, 3, 6, 10]
    assert run(prog, [], 4).memory[:4] == [1, 2, 3, 4]
    with pytest.raises(IsaSimError, match="deeper than 5"):
        run(prog, [], 4, simt=True, stack_depth=5)


def test_ret_in_divergent_region():
    prog = _asm("""
        SYNC join
        CONST R0, #0
        CONST R5, #5
        CMP %threadIdx, R0
        BRn rest
        STR R0, R5              ; thread 0 only
        RET
    rest:
        CONST R1, #7
        STR %threadIdx, R1
    join:
        CONST R2, #8
        ADD R3, %threadIdx, R2
        STR R3, R2
        RET""")
    mem = _mem(prog, [], 4, simt=True)
    assert mem[:4] == [5, 7, 7, 7]
    assert mem[8:12] == [0, 8, 8, 8]


def test_max_steps():
    prog = assemble("CONST R0, #0\nCMP R0, R0\nspin: BRz spin\nRET")
    with pytest.raises(IsaSimError, match="steps"):
        run(prog, [], 4, max_steps=100)
    assert run(assemble("NOP\nRET"), [], 8).steps == 4


def _tiled(seed, m=4, k=12, n=8):
    rng = np.random.default_rng(seed)
    a, b = rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (k, n))
    img = Image()
    img.alloc("a", a.shape, "i8", data=a)
    img.alloc("b", b.shape, "i8", data=b)
    img.alloc("c", [m, n], "i32")
    prog, threads = kernels.matmul(img, "tiled", "a", "b", "c")
    return img, prog, threads, golden.matmul_i8(a, b)


def _c(img, prog, threads, **kw):
    return img.read("c", run(prog, img.words, threads, **X32, threads_per_block=4, **kw).memory)


@pytest.mark.parametrize("warps,policy", [(1, "lrr"), (2, "lrr"), (2, "gto"), (4, "lrr"), (4, "gto")])
def test_multiwarp_matmul_tiled(warps, policy):
    img, prog, threads, want = _tiled(0)
    assert np.array_equal(_c(img, prog, threads, warps_per_core=warps, warp_policy=policy), want)


def test_multiwarp_bar_matters():
    img, prog, threads, want = _tiled(1)
    nobar = [encode("NOP") if decode(w)[0] == "BAR" else w for w in prog]
    assert nobar != prog
    # Round-robin issue keeps warps one instruction apart, which this kernel tolerates without BAR;
    # a greedy policy runs warp 0 ahead and reads SMEM words warp 1 has not staged yet.
    assert np.array_equal(_c(img, nobar, threads, warps_per_core=2), want)
    assert not np.array_equal(_c(img, nobar, threads, warps_per_core=2, warp_policy="gto"), want)
    assert np.array_equal(_c(img, prog, threads, warps_per_core=2, warp_policy="gto"), want)
