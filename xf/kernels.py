"""Kernel binding: assemble kernels/<name>.asm and point it at an args block in the image.

Every xf kernel starts with the ARGS prologue `CONST Rx, #hi ... CONST Ry, #lo` (words 0 and 4)
that builds the args base hi*256 + lo in a register; `program` patches those two immediates.
"""
from pathlib import Path

from .asm import assemble
from .isa import decode, encode

KERNEL_DIR = Path(__file__).resolve().parents[1] / "kernels"
TILE = 4  # matmul_tiled: blockDim and tile width


def program(name, args_base):
    words = assemble((KERNEL_DIR / f"{name}.asm").read_text())
    for i, imm in ((0, args_base >> 8), (4, args_base & 0xFF)):
        mnem, f = decode(words[i])
        if mnem != "CONST":
            raise ValueError(f"{name}: word {i} is not part of the ARGS prologue")
        words[i] = encode("CONST", rd=f["rd"], imm=imm)
    return words


def matmul(img, kind, a, b, c, mq=0, s=0):
    """Bind matmul_<kind> (naive|tiled) for c = a @ b; mq != 0 fuses requant(acc, mq, s).

    Allocates the args segment `<c>.args` and returns (program, threads).
    """
    (m, k), (k2, n) = img.segments[a]["shape"], img.segments[b]["shape"]
    if k != k2 or img.segments[c]["shape"] != [m, n]:
        raise ValueError(f"shapes {m}x{k} @ {k2}x{n} -> {img.segments[c]['shape']}")
    if not 0 <= s <= 31:
        raise ValueError(f"shift {s} outside [0, 31]")
    if kind == "naive":
        threads = m * n
    elif kind == "tiled":
        if n % TILE:
            raise ValueError(f"matmul_tiled needs N % {TILE} == 0, got N={n}")
        threads = -(-m // TILE) * n
    else:
        raise ValueError(f"unknown matmul kind {kind!r}")
    if threads > 255:
        raise ValueError(f"{threads} threads exceed the 8-bit thread count")
    args = [img.base(a), img.base(b), img.base(c), m, k, n, mq, s, (1 << (s - 1)) if s else 0]
    return program(f"matmul_{kind}", img.alloc(f"{c}.args", [len(args)], "i32", data=args)), threads
