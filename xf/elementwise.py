"""Bindings for the elementwise, RMSNorm, transpose and softmax kernels (calling convention: xf/kernels.py).

Each binding allocates the args segment `<out>.args` and returns (program, threads),
one thread per output element. Semantics are xf.golden's, bit for bit.
"""
import math

from . import golden
from .kernels import program, require_i8


def _rnd(s):
    if not 0 <= s <= 31:
        raise ValueError(f"shift {s} outside [0, 31]")
    return (1 << (s - 1)) if s else 0


def _shape(img, name):
    return img.segments[name]["shape"]


def _same(img, out, *names):
    for n in names:
        if _shape(img, n) != _shape(img, out):
            raise ValueError(f"{n} shape {_shape(img, n)} != {out} shape {_shape(img, out)}")


def _bind(img, kernel, out, args):
    threads = math.prod(_shape(img, out))
    if threads < 1:
        raise ValueError(f"{out} is empty")
    if threads > 255:
        raise ValueError(f"{threads} threads exceed the 8-bit thread count")
    return program(kernel, img.alloc(f"{out}.args", [len(args)], "i32", data=args)), threads


def requant(img, x, out, m, s):
    """out = golden.requant(x, m, s)."""
    _same(img, out, x)
    return _bind(img, "requant", out, [img.base(x), img.base(out), m, s, _rnd(s)])


def relu(img, x, out):
    _same(img, out, x)
    require_i8(img, x)
    return _bind(img, "relu", out, [img.base(x), img.base(out)])


def swiglu(img, gate, up, lut, out, m, s):
    """out = golden.swiglu_lut(gate, up, lut, m, s)."""
    _same(img, out, gate, up)
    require_i8(img, gate, up)
    if _shape(img, lut) != [256]:
        raise ValueError(f"{lut} must have 256 entries")
    return _bind(img, "swiglu", out, [img.base(gate), img.base(up), img.base(lut), img.base(out), m, s, _rnd(s)])


def residual_add(img, a, b, out, ma, mb, s):
    """out = golden.residual_add(a, b, ma, mb, s)."""
    _same(img, out, a, b)
    require_i8(img, a, b)
    return _bind(img, "residual_add", out, [img.base(a), img.base(b), img.base(out), ma, mb, s, _rnd(s)])


def rmsnorm(img, x, w, lut, out, m, s):
    """out = golden.rmsnorm_int(x, w, lut, m, s) over the rows of x (T, d)."""
    _same(img, out, x)
    require_i8(img, x, w)
    d = _shape(img, x)[-1]
    if _shape(img, w) != [d]:
        raise ValueError(f"{w} shape {_shape(img, w)} != [{d}]")
    n = math.prod(_shape(img, lut))
    kmax = int(golden.rsqrt_index(d * 128 * 128, n)[1])
    if not 0 <= s <= 30 - kmax:
        raise ValueError(f"shift {s} + {kmax} normalization steps outside [0, 30]")
    return _bind(img, "rmsnorm", out, [img.base(x), img.base(w), img.base(lut), img.base(out),
                                       d, max(kmax, 1), n - 1, m, s, 1 << s])


def transpose(img, x, out):
    """out = x.T for a 2-D x."""
    r, c = _shape(img, x)
    if _shape(img, out) != [c, r]:
        raise ValueError(f"{out} shape {_shape(img, out)} != {[c, r]}")
    return _bind(img, "transpose", out, [img.base(x), img.base(out), r, c])


def softmax(img, scores, lut, out, one):
    """out = golden.softmax_lut(scores, lut, one), causal, over square scores (T, T)."""
    _same(img, out, scores)
    require_i8(img, scores)
    t, t2 = _shape(img, scores)
    if t != t2:
        raise ValueError(f"{scores} must be square, got {[t, t2]}")
    if _shape(img, lut) != [256]:
        raise ValueError(f"{lut} must have 256 entries")
    return _bind(img, "softmax", out, [img.base(scores), img.base(lut), img.base(out), t, one])
