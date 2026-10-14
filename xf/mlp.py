"""MLP block launch chain on an xf.export image: out = x + down(swiglu(gate(h), up(h))), h = rmsnorm(x).

Weights, LUTs and requant constants are the image's exported segments; activations are
the segments named in `acts` (default: the exported ones), so several inputs can share
one image.
"""
from . import elementwise as ew, kernels

ACTS = ("x", "h", "gate", "up", "act", "down", "out")


def alloc_acts(img, prefix):
    """Allocate a copy of every activation segment as `<prefix><name>`; return the name map."""
    for n in ACTS:
        seg = img.segments[n]
        img.alloc(prefix + n, seg["shape"], seg["dtype"], seg["scale"])
    return {n: prefix + n for n in ACTS}


def launches(img, kind="tiled", acts=None):
    """[(output segment, (program, threads))] in launch order; `kind` picks the matmul kernel."""
    a = acts or {n: n for n in ACTS}
    c = {n: int(img.read(n)[0]) for n, seg in img.segments.items() if seg["shape"] == [1]}
    mm = lambda w, x, out: kernels.matmul(img, kind, a[x], w, a[out], c[f"{out}_M"], c[f"{out}_s"])
    return [
        (a["h"], ew.rmsnorm(img, a["x"], "norm_w", "rsqrt_lut", a["h"], c["norm_M"], c["norm_s"])),
        (a["gate"], mm("w_gate", "h", "gate")),
        (a["up"], mm("w_up", "h", "up")),
        (a["act"], ew.swiglu(img, a["gate"], a["up"], "silu_lut", a["act"], c["act_M"], c["act_s"])),
        (a["down"], mm("w_down", "act", "down")),
        (a["out"], ew.residual_add(img, a["x"], a["down"], a["out"], c["res_Ma"], c["res_Mb"], c["res_s"])),
    ]
