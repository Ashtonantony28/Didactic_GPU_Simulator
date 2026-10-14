"""Integer golden model: bit-exact semantics of the transformer kernels.

All arithmetic is int64 here; every intermediate a kernel holds in a 32-bit
register is checked to fit int32, so wrap-free results equal the hardware's.
Shifts are arithmetic (SRA), division is only applied to non-negative values.
"""
import numpy as np

I8 = (-128, 127)
I32 = (-(1 << 31), (1 << 31) - 1)


def _i32(x, what):
    x = np.asarray(x, dtype=np.int64)
    if x.size and (x.min() < I32[0] or x.max() > I32[1]):
        raise OverflowError(f"{what} overflows int32")
    return x


def _i8(x, what):
    x = np.asarray(x, dtype=np.int64)
    if x.size and (x.min() < I8[0] or x.max() > I8[1]):
        raise ValueError(f"{what} is not int8")
    return x


def clamp_i8(x):
    return np.clip(x, *I8)


def matmul_i8(a, b):
    """(M,K) int8 @ (K,N) int8 -> (M,N) int32 accumulators."""
    a, b = _i8(a, "a"), _i8(b, "b")
    return _i32(a @ b, "matmul accumulator")


def requant(acc, m, s):
    """clamp((acc*m + 2^(s-1)) >> s, -128, 127); no rounding term when s == 0."""
    if not 0 <= s <= 31:
        raise ValueError(f"shift {s} outside [0, 31]")
    acc = _i32(acc, "acc")
    prod = _i32(acc * int(m), "acc*M")
    rounded = _i32(prod + ((1 << (s - 1)) if s else 0), "acc*M + round")
    return clamp_i8(rounded >> s)


def relu(x):
    return np.maximum(_i8(x, "x"), 0)


def swiglu_lut(gate, up, silu_lut, m, s):
    """requant(silu_lut[gate+128] * up, m, s); silu_lut is 256 int8 entries."""
    lut = _i8(silu_lut, "silu_lut")
    if lut.shape != (256,):
        raise ValueError("silu_lut must have 256 entries")
    gate, up = _i8(gate, "gate"), _i8(up, "up")
    return requant(lut[gate + 128] * up, m, s)


def residual_add(a, b, ma, mb, s):
    """Scale-aligned residual: clamp((a*ma + b*mb + 2^(s-1)) >> s)."""
    a, b = _i8(a, "a"), _i8(b, "b")
    return requant(_i32(_i32(a * int(ma), "a*Ma") + _i32(b * int(mb), "b*Mb"), "a*Ma + b*Mb"), 1, s)


def rsqrt_index(ss, n):
    """Least k >= 0 with ss >> 2k < n, and idx = ss >> 2k: ss = idx * 4^k up to truncation, idx >= n//4 when k > 0."""
    idx, k = np.array(ss, dtype=np.int64), np.zeros(np.shape(ss), dtype=np.int64)
    while (big := idx >= n).any():
        idx[big] >>= 2
        k[big] += 1
    return idx, k


def rmsnorm_int(x, w, rsqrt_lut, m, s):
    """Per row of x (T,d): idx, k = rsqrt_index(sum(x^2), len), r = rsqrt_lut[idx],
    out = requant(x * w * r, m, s + k). rsqrt(4^k idx) = rsqrt(idx) / 2^k, so the LUT holds rsqrt of
    its exact index with eps and 1/d folded in, and the 2^-k goes into the requant shift."""
    x, w = _i8(x, "x"), _i8(w, "w")
    lut = _i32(rsqrt_lut, "rsqrt_lut")
    if lut.ndim != 1 or lut.min() < 0:
        raise ValueError("rsqrt_lut must be a non-negative 1-D table")
    idx, k = rsqrt_index(_i32((x * x).sum(axis=-1), "sum of squares"), len(lut))
    xwr = _i32(_i32(x * w, "x*w") * lut[idx][..., None], "x*w*r")
    out = np.empty_like(xwr)
    for kv in np.unique(k):
        out[k == kv] = requant(xwr[k == kv], m, s + int(kv))
    return out


def softmax_lut(scores, exp_lut, one, causal=True):
    """Row softmax of int8 scores (T,T) with exp_lut[d + 255], d = s - rowmax in [-255, 0].
    p = (e*one + S//2) // S over unmasked positions (j <= i when causal); masked p = 0."""
    scores = _i8(scores, "scores")
    lut = _i32(exp_lut, "exp_lut")
    if lut.shape != (256,) or lut.min() < 0:
        raise ValueError("exp_lut must be 256 non-negative entries")
    t = scores.shape[-1]
    keep = np.tril(np.ones((scores.shape[-2], t), dtype=bool)) if causal else np.ones(scores.shape, dtype=bool)
    rowmax = np.where(keep, scores, I8[0]).max(axis=-1, keepdims=True)
    e = np.where(keep, lut[np.where(keep, scores - rowmax + 255, 255)], 0)
    total = _i32(e.sum(axis=-1, keepdims=True), "exp sum")
    if (total <= 0).any():
        raise ZeroDivisionError("row with zero exp sum")
    return _i32(e * int(one) + total // 2, "e*one + S/2") // total
