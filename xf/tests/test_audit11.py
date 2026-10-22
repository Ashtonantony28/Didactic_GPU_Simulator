"""AUDIT-11 repros (API/SDK consistency). Each is a strict xfail until fixed."""
import re
from pathlib import Path

import numpy as np
import pytest

from xf import elementwise as ew, isasim, kernels
from xf.asm import AsmError, assemble
from xf.image import Image

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.xfail(strict=True, reason="AUDIT-11 M1: the Makefile's PARAMS example has unscoped -P switches, "
                                       "which iverilog ignores without error")
def test_makefile_params_example_is_scoped():
    line = next(ln for ln in (ROOT / "Makefile").read_text().splitlines() if "PARAMS=\"" in ln and ln.startswith("#"))
    switches = re.findall(r"-P(\S+?)=", line)
    assert switches and all(s.startswith("gpu.") for s in switches), line


def _mm_img(c_dtype):
    img = Image()
    img.alloc("a", [2, 2], "i8", data=np.full((2, 2), 100))
    img.alloc("b", [2, 2], "i8", data=np.full((2, 2), 100))
    img.alloc("c", [2, 2], c_dtype)
    return img


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L1: matmul with mq=0 writes int32 accumulators into an i8 "
                                       "segment, which later binders then accept as i8")
def test_unrequantized_matmul_rejects_i8_output():
    with pytest.raises(ValueError):
        kernels.matmul(_mm_img("i8"), "naive", "a", "b", "c")


def test_unrequantized_matmul_into_i32_still_accepted():
    kernels.matmul(_mm_img("i32"), "naive", "a", "b", "c")


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L2: the assembler accepts BRp, which is never taken "
                                       "(CMP never sets P); PLAN SG22 says it rejects BRp")
def test_assembler_rejects_brp():
    with pytest.raises(AsmError):
        assemble("BRp 0")


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L3: a program longer than the 256-word program memory "
                                       "assembles; the runner then drops the tail silently")
def test_assembler_rejects_programs_past_program_memory():
    with pytest.raises(AsmError):
        assemble("NOP\n" * 256 + "RET\n")


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L4: a binder that fails on a constant leaves <out>.args "
                                       "allocated, so a corrected retry fails")
def test_failed_bind_leaves_image_unchanged():
    img = Image()
    img.alloc("x", [4], "i32")
    img.alloc("o", [4], "i8")
    with pytest.raises((ValueError, OverflowError)):
        ew.requant(img, "x", "o", 1 << 31, 0)
    assert "o.args" not in img.segments
    ew.requant(img, "x", "o", 1, 0)


def _ew_img(lut_shape, lut_dtype):
    img = Image()
    for n, s in (("x", [1, 8]), ("g", [1, 8]), ("w", [8]), ("o", [1, 8])):
        img.alloc(n, s, "i8")
    img.alloc("lut", lut_shape, lut_dtype)
    return img


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L5: binders accept LUTs that golden rejects "
                                       "(2-D rsqrt_lut, i32 silu_lut)")
@pytest.mark.parametrize("bind", [
    lambda: ew.rmsnorm(_ew_img([32, 32], "i32"), "x", "w", "lut", "o", 1, 0),
    lambda: ew.swiglu(_ew_img([256], "i32"), "g", "x", "lut", "o", 1, 0),
], ids=["rmsnorm_2d_lut", "swiglu_i32_lut"])
def test_binders_reject_luts_golden_rejects(bind):
    with pytest.raises(ValueError):
        bind()


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L6: contract 9 says errors raise IsaSimError; "
                                       "argument checks raise ValueError")
def test_isasim_argument_errors_are_isasimerror():
    with pytest.raises(isasim.IsaSimError):
        isasim.run([0xF000], [], 4, smem_size=48)


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L6: isasim runs 300 threads; the RTL thread count is 8 bits "
                                       "and the binders cap it at 255")
def test_isasim_rejects_thread_count_past_8_bits():
    with pytest.raises((ValueError, isasim.IsaSimError)):
        isasim.run([0xF000], [], 300)


@pytest.mark.xfail(strict=True, reason="AUDIT-11 L7: Image.alloc accepts a negative dimension, "
                                       "so two segments share a base")
def test_image_rejects_negative_shape():
    img = Image()
    with pytest.raises(ValueError):
        img.alloc("n", [-4], "i8")
