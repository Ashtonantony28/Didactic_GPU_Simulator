"""Single causal attention head (nanoGPT attn(ln_1(x)), CausalSelfAttention n_head=1) on an int8 image.

    h = rmsnorm(x); q, k, v = h @ Wq, h @ Wk, h @ Wv; scores = q @ k.T (1/sqrt(hd) folded into its requant);
    p = softmax_lut(scores, exp_lut, one); o = p @ v; y = o @ Wo

Torch-free: the image layout, golden chain, bounds and launch chain. xf/export.py builds the
image from nanoGPT and writes test/fixtures/xf/attention_<i>.json.
"""
import itertools
import math

import numpy as np

from . import elementwise as ew, golden, kernels
from .image import Image

ONE = 127                       # fixed-point 1.0 of p; p <= ONE keeps p int8 for the p @ v matmul
ACTS = ("x", "h", "q", "k", "v", "kT", "scores", "p", "o", "y")
PROJ = ("q", "k", "v")


def _rnd(s):
    return (1 << (s - 1)) if s else 0


def exp_bits(t, one=ONE):
    """Largest E with every softmax intermediate in int32 for entries <= 2^E: e*one + S/2, S <= t*2^E."""
    return max(e for e in range(31) if (one + t // 2) << e <= golden.I32[1] and t << e <= golden.I32[1])


def exp_lut(score_scale, t, one=ONE):
    """exp_lut[d + 255] = round(exp(d * score_scale) * 2^E), d in [-255, 0]."""
    return np.rint(np.exp(np.arange(-255, 1) * score_scale) * (1 << exp_bits(t, one))).astype(np.int64)


def build_image(weights, scales, t, fixed_point, quantize, maxabs_scale, rmsnorm_lut):
    """weights: float norm_w (d,), w_q/w_k/w_v/w_o (d, d) as x @ W. scales: x, h, q, k, v, scores, o, y.
    The export helpers (fixed_point, quantize, maxabs_scale, rmsnorm_lut) come from xf.export."""
    sc = scales
    w = {k: quantize(v, maxabs_scale(v)) for k, v in weights.items()}
    sw = {k: maxabs_scale(v) for k, v in weights.items()}
    d, hd = weights["w_q"].shape

    rsqrt_lut, norm_s = rmsnorm_lut(sc["x"], sw["norm_w"], sc["h"], int(np.abs(w["norm_w"]).max()), d)

    def mm(name, x, out):
        return fixed_point([sc[x] * sw[name] / sc[out]], [128 * int(np.abs(w[name]).sum(axis=0).max())])

    img = Image(word_bits=32, addr_bits=12)
    img.alloc("x", [t, d], "i8", sc["x"])
    img.alloc("norm_w", [d], "i8", sw["norm_w"], w["norm_w"])
    img.alloc("rsqrt_lut", [len(rsqrt_lut)], "i32", data=rsqrt_lut)
    img.alloc("norm_M", [1], "i32", data=[1])
    img.alloc("norm_s", [1], "i32", data=[norm_s])
    img.alloc("h", [t, d], "i8", sc["h"])
    for p in PROJ:
        (m,), s = mm(f"w_{p}", "h", p)
        img.alloc(f"w_{p}", [d, hd], "i8", sw[f"w_{p}"], w[f"w_{p}"])
        img.alloc(f"{p}_M", [1], "i32", data=[m])
        img.alloc(f"{p}_s", [1], "i32", data=[s])
        img.alloc(p, [t, hd], "i8", sc[p])
    img.alloc("kT", [hd, t], "i8", sc["k"])
    (m,), s = fixed_point([sc["q"] * sc["k"] / math.sqrt(hd) / sc["scores"]], [128 * 128 * hd])
    img.alloc("scores_M", [1], "i32", data=[m])
    img.alloc("scores_s", [1], "i32", data=[s])
    img.alloc("scores", [t, t], "i8", sc["scores"])
    img.alloc("exp_lut", [256], "i32", data=exp_lut(sc["scores"], t))
    img.alloc("one", [1], "i32", data=[ONE])
    img.alloc("p", [t, t], "i8", 1 / ONE)
    (m,), s = fixed_point([sc["v"] / ONE / sc["o"]], [128 * ONE * t])
    img.alloc("o_M", [1], "i32", data=[m])
    img.alloc("o_s", [1], "i32", data=[s])
    img.alloc("o", [t, hd], "i8", sc["o"])
    (m,), s = mm("w_o", "o", "y")
    img.alloc("w_o", [hd, d], "i8", sw["w_o"], w["w_o"])
    img.alloc("y_M", [1], "i32", data=[m])
    img.alloc("y_s", [1], "i32", data=[s])
    img.alloc("y", [t, d], "i8", sc["y"])
    return img


def _c(img, name):
    return int(img.read(name)[0])


def attention_int(img, x):
    """Golden integer attention head on image constants: every intermediate, keyed by segment."""
    def rq(acc, out):
        return golden.requant(acc, _c(img, f"{out}_M"), _c(img, f"{out}_s"))

    o = {"h": golden.rmsnorm_int(x, img.read("norm_w"), img.read("rsqrt_lut"), _c(img, "norm_M"), _c(img, "norm_s"))}
    o.update({p: rq(golden.matmul_i8(o["h"], img.read(f"w_{p}")), p) for p in PROJ})
    o["kT"] = np.swapaxes(o["k"], -1, -2)
    o["scores"] = rq(golden.matmul_i8(o["q"], o["kT"]), "scores")
    o["p"] = golden.softmax_lut(o["scores"], img.read("exp_lut"), _c(img, "one"))
    o["o"] = rq(golden.matmul_i8(o["p"], o["v"]), "o")
    o["y"] = rq(golden.matmul_i8(o["o"], img.read("w_o")), "y")
    return o


def check_bounds(img):
    """Worst case over all int8 inputs of every requant numerator and softmax term fits int32."""
    t, hd = img.segments["q"]["shape"]
    one, emax = _c(img, "one"), int(img.read("exp_lut").max())
    worst = {p: 128 * int(np.abs(img.read(f"w_{w}")).sum(axis=0).max()) * _c(img, f"{p}_M") + _rnd(_c(img, f"{p}_s"))
             for p, w in (("q", "q"), ("k", "k"), ("v", "v"), ("y", "o"))}
    kmax = int(golden.rsqrt_index(img.segments["x"]["shape"][-1] * 128 * 128, len(img.read("rsqrt_lut")))[1])
    worst["h"] = (128 * int(np.abs(img.read("norm_w")).max()) * int(img.read("rsqrt_lut").max()) * _c(img, "norm_M")
                  + _rnd(_c(img, "norm_s") + kmax))
    worst["scores"] = 128 * 128 * hd * _c(img, "scores_M") + _rnd(_c(img, "scores_s"))
    worst["o"] = 128 * one * t * _c(img, "o_M") + _rnd(_c(img, "o_s"))
    worst["softmax"] = emax * one + t * emax // 2
    bad = {k: v for k, v in worst.items() if v > golden.I32[1]}
    if bad:
        raise OverflowError(f"attention bounds exceed int32: {bad}")
    return worst


def row_sum_error(lut, one, t):
    """Max |sum(p) - one| over every reachable causal row of length 1..t, by enumeration.

    softmax_lut depends only on d = s - rowmax in [-255, 0] and the row max has d = 0, so a row
    of n unmasked entries is (0, d_1..d_{n-1}): 256^(n-1) rows, two dimensions vectorized."""
    lut = np.asarray(lut, dtype=np.int64)
    worst = 0
    for n in range(1, t + 1):
        for head in itertools.product(lut, repeat=max(n - 3, 0)):
            cols = np.meshgrid(lut[255:], *([[h] for h in head]), *[lut] * min(n - 1, 2), indexing="ij")
            total = sum(cols)
            worst = max(worst, int(np.abs(sum((c * one + total // 2) // total for c in cols) - one).max()))
    return worst


def alloc_acts(img, prefix):
    """Allocate a copy of every activation segment as `<prefix><name>`; return the name map."""
    for n in ACTS:
        seg = img.segments[n]
        img.alloc(prefix + n, seg["shape"], seg["dtype"], seg["scale"])
    return {n: prefix + n for n in ACTS}


def launches(img, kind="tiled", acts=None):
    """[(output segment, (program, threads))] in launch order; `kind` picks the matmul kernel."""
    a = acts or {n: n for n in ACTS}
    c = {n: _c(img, n) for n, seg in img.segments.items() if seg["shape"] == [1]}
    mm = lambda x, w, out: kernels.matmul(img, kind, x, w, a[out], c[f"{out}_M"], c[f"{out}_s"])
    return [
        (a["h"], ew.rmsnorm(img, a["x"], "norm_w", "rsqrt_lut", a["h"], c["norm_M"], c["norm_s"])),
        *((a[p], mm(a["h"], f"w_{p}", p)) for p in PROJ),
        (a["kT"], ew.transpose(img, a["k"], a["kT"])),
        (a["scores"], mm(a["q"], a["kT"], "scores")),
        (a["p"], ew.softmax(img, a["scores"], "exp_lut", a["p"], c["one"])),
        (a["o"], mm(a["p"], a["v"], "o")),
        (a["y"], mm(a["o"], "w_o", "y")),
    ]
