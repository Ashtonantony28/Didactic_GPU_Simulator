"""Quantize the D3 nanoGPT MLP block (x + mlp(rmsnorm(x))) and attention head (attn(rmsnorm(x))) to int8 and write fixtures.

    NANOGPT_DIR=... python3 -m xf.export           # write test/fixtures/xf/{mlp_block,attention}_<i>.json
    NANOGPT_DIR=... python3 -m xf.export --check   # determinism, bounds, tolerance, fixtures current

Scales are symmetric per-tensor max-abs over a seeded calibration set, every
non-input activation widened by HEADROOM so held-out inputs do not saturate. Every
requant (M, s) is chosen against the worst case over all int8 inputs, so the
golden pipeline (and the kernels) can never overflow int32. The integer side
(`mlp_block_int`, `check_bounds`, xf/attention.py) needs no torch.
"""
import argparse
import math
import os
import sys

import numpy as np

from . import attention, fixture, golden
from .image import Image

N_EMBD, HIDDEN, T = 8, 16, 4
SEED, N_CALIB, N_CHECK, N_FIXTURES = 0, 4096, 200, 4
RSQRT_LEN, EPS = 1024, 1e-6
TOL_LSB = 2
# attention calibrates on 2^16 inputs: with 4096, k and v saturate on held-out inputs (16 LSB worst of 5000).
# TOL 6 covers the adversarial search (5.35 LSB, SG14); the Gaussian tail alone exceeds 3
ATTN_TOL_LSB, ATTN_N_CALIB, ATTN_N_CHECK = 6, 1 << 16, 2000
ACTS = ("x", "h", "gate", "up", "act", "down", "out")
# max-abs calibration saturates on rare held-out inputs (AUDIT-2 H1); check measures HELDOUT_N of them
HEADROOM, HELDOUT_SEED, HELDOUT_N = 1.5, 12345, 200_000
HELDOUT_TOL = ("mlp_block", "attention")
OUTLIER = np.r_[1, 1, 1, 1, 1, 4, 1, 1]
# (least-squares gain, RMS error in LSB) of each golden segment on the held-out inputs at SG16; check allows
# |gain - HEAD| <= GAIN_TOL and RMS <= HEAD * RMS_TOL (AUDIT-7 H2, contract 8)
GAIN_TOL, RMS_TOL = 0.0025, 1.1
SEGMENTS = {
    "mlp_block": {"h": (1.0021, 0.298), "gate": (1.0024, 0.409), "up": (1.0018, 0.423), "act": (1.0045, 0.369),
                  "down": (1.0037, 0.753), "out": (1.0001, 0.337)},
    "attention": {"h": (0.9997, 0.290), "q": (1.0009, 0.482), "k": (0.9998, 0.454), "v": (0.9998, 0.454),
                  "kT": (0.9998, 0.454), "scores": (0.9998, 0.542), "p": (1.0000, 0.318), "o": (0.9964, 0.443),
                  "y": (0.9958, 0.501)},
}


def fixed_point(ms, bounds, k=0):
    """Largest s in [0, 31 - k] with M_i = round(m_i * 2^s) and sum(bound_i * M_i) + 2^(s+k-1) <= INT32_MAX;
    k is the largest extra requant shift (rmsnorm_int's normalization steps)."""
    for s in range(31 - k, -1, -1):
        mi = [round(m * (1 << s)) for m in ms]
        if sum(b * m for b, m in zip(bounds, mi)) + ((1 << (s + k - 1)) if s + k else 0) <= golden.I32[1]:
            if min(mi) < 1:
                raise ValueError(f"multipliers {ms} vanish at shift {s}")
            return mi, s
    raise OverflowError(f"bounds {bounds} leave no room for multipliers {ms}")


def quantize(x, scale):
    return golden.clamp_i8(np.rint(np.asarray(x) / scale)).astype(np.int64)


def maxabs_scale(x):
    return float(np.abs(x).max()) / 127


def calib_scales(acts, sx):
    return {k: sx if k == "x" else HEADROOM * maxabs_scale(v) for k, v in acts.items()}


def rsqrt_kmax(d, n=RSQRT_LEN):
    """Normalization steps rmsnorm_int takes on the loudest int8 row."""
    return int(golden.rsqrt_index(d * 128 * 128, n)[1])


def rmsnorm_lut(sx, sw, sh, wmax, d):
    """(rsqrt_lut, s) for rmsnorm_int from x (scale sx) and weight (scale sw, max |w_q| wmax) to h (scale sh)."""
    ss = np.arange(1, RSQRT_LEN)    # rsqrt_lut[0] only meets x = 0
    r = np.r_[0, 1 / np.sqrt(sx ** 2 * ss / d + EPS) * sx * sw / sh]
    (rmax,), s = fixed_point([r.max()], [128 * wmax], rsqrt_kmax(d))
    lut = np.rint(r * (1 << s)).astype(np.int64)
    assert lut.max() == rmax
    return lut, s


def build_image(weights, scales):
    """weights: float norm_w (d,), w_gate/w_up (d,H), w_down (H,d) as x @ W. scales: per activation."""
    sc = scales
    w = {k: quantize(v, maxabs_scale(v)) for k, v in weights.items()}
    sw = {k: maxabs_scale(v) for k, v in weights.items()}
    d, hidden = weights["w_gate"].shape

    rsqrt_lut, norm_s = rmsnorm_lut(sc["x"], sw["norm_w"], sc["h"], int(np.abs(w["norm_w"]).max()), d)

    g = np.arange(-128, 128) * sc["gate"]
    silu = g / (1 + np.exp(-g))
    s_silu = maxabs_scale(silu)
    silu_lut = quantize(silu, s_silu)

    def mm(name, out):
        bound = 128 * int(np.abs(w[name]).sum(axis=0).max())
        return fixed_point([sc["h" if name != "w_down" else "act"] * sw[name] / sc[out]], [bound])

    (gate_m,), gate_s = mm("w_gate", "gate")
    (up_m,), up_s = mm("w_up", "up")
    (act_m,), act_s = fixed_point([s_silu * sc["up"] / sc["act"]], [128 * int(np.abs(silu_lut).max())])
    (down_m,), down_s = mm("w_down", "down")
    (res_ma, res_mb), res_s = fixed_point([sc["x"] / sc["out"], sc["down"] / sc["out"]], [128, 128])

    img = Image(word_bits=32, addr_bits=12)
    img.alloc("x", [T, d], "i8", sc["x"])
    img.alloc("norm_w", [d], "i8", sw["norm_w"], w["norm_w"])
    img.alloc("rsqrt_lut", [RSQRT_LEN], "i32", data=rsqrt_lut)
    for name, v in (("norm_M", 1), ("norm_s", norm_s)):
        img.alloc(name, [1], "i32", data=[v])
    img.alloc("h", [T, d], "i8", sc["h"])
    for proj, m, s in (("gate", gate_m, gate_s), ("up", up_m, up_s)):
        img.alloc(f"w_{proj}", [d, hidden], "i8", sw[f"w_{proj}"], w[f"w_{proj}"])
        img.alloc(f"{proj}_M", [1], "i32", data=[m])
        img.alloc(f"{proj}_s", [1], "i32", data=[s])
        img.alloc(proj, [T, hidden], "i8", sc[proj])
    img.alloc("silu_lut", [256], "i8", s_silu, silu_lut)
    img.alloc("act_M", [1], "i32", data=[act_m])
    img.alloc("act_s", [1], "i32", data=[act_s])
    img.alloc("act", [T, hidden], "i8", sc["act"])
    img.alloc("w_down", [hidden, d], "i8", sw["w_down"], w["w_down"])
    img.alloc("down_M", [1], "i32", data=[down_m])
    img.alloc("down_s", [1], "i32", data=[down_s])
    img.alloc("down", [T, d], "i8", sc["down"])
    for name, v in (("res_Ma", res_ma), ("res_Mb", res_mb), ("res_s", res_s)):
        img.alloc(name, [1], "i32", data=[v])
    img.alloc("out", [T, d], "i8", sc["out"])
    return img


def _c(img, name):
    return int(img.read(name)[0])


def mlp_block_int(img, x):
    """Golden integer MLP block on image constants: every intermediate, keyed by segment."""
    o = {"h": golden.rmsnorm_int(x, img.read("norm_w"), img.read("rsqrt_lut"), _c(img, "norm_M"), _c(img, "norm_s"))}
    for proj in ("gate", "up"):
        o[proj] = golden.requant(golden.matmul_i8(o["h"], img.read(f"w_{proj}")), _c(img, f"{proj}_M"), _c(img, f"{proj}_s"))
    o["act"] = golden.swiglu_lut(o["gate"], o["up"], img.read("silu_lut"), _c(img, "act_M"), _c(img, "act_s"))
    o["down"] = golden.requant(golden.matmul_i8(o["act"], img.read("w_down")), _c(img, "down_M"), _c(img, "down_s"))
    o["out"] = golden.residual_add(x, o["down"], _c(img, "res_Ma"), _c(img, "res_Mb"), _c(img, "res_s"))
    return o


def check_bounds(img):
    """Worst case over all int8 inputs of every requant numerator fits int32."""
    def rnd(s):
        return (1 << (s - 1)) if s else 0

    kmax = rsqrt_kmax(img.segments["x"]["shape"][-1], len(img.read("rsqrt_lut")))
    worst = {
        "norm": 128 * int(np.abs(img.read("norm_w")).max()) * int(img.read("rsqrt_lut").max())
        * _c(img, "norm_M") + rnd(_c(img, "norm_s") + kmax),
        "act": 128 * int(np.abs(img.read("silu_lut")).max()) * _c(img, "act_M") + rnd(_c(img, "act_s")),
        "res": 128 * (_c(img, "res_Ma") + _c(img, "res_Mb")) + rnd(_c(img, "res_s")),
    }
    for p in ("gate", "up", "down"):
        worst[p] = 128 * int(np.abs(img.read(f"w_{p}")).sum(axis=0).max()) * _c(img, f"{p}_M") + rnd(_c(img, f"{p}_s"))
    bad = {k: v for k, v in worst.items() if v > golden.I32[1]}
    if bad:
        raise OverflowError(f"requant bounds exceed int32: {bad}")
    return worst


def build_block(nanogpt_dir):
    """D3 block from nanoGPT, seeded; SwiGLU hidden overridden to HIDDEN, float64, eval."""
    import torch
    from torch import nn

    sys.path.insert(0, str(nanogpt_dir))
    from model import GPTConfig
    from models.gpt import Block

    torch.manual_seed(SEED)
    cfg = GPTConfig(block_size=T, n_layer=1, n_head=1, n_embd=N_EMBD, dropout=0.0, bias=False,
                    pos_embedding="none", norm_type="rmsnorm", activation="swiglu")
    blk = Block(cfg)
    blk.ff.gate_proj = nn.Linear(N_EMBD, HIDDEN, bias=False)
    blk.ff.up_proj = nn.Linear(N_EMBD, HIDDEN, bias=False)
    blk.ff.down_proj = nn.Linear(HIDDEN, N_EMBD, bias=False)
    for norm in (blk.ln_1, blk.ln_2):
        nn.init.uniform_(norm.weight, 0.5, 1.5)
    return blk.double().eval()


def float_acts(blk, x):
    """Float activations of x + ff(ln_2(x)) through nanoGPT's own modules."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float64)
        h = blk.ln_2(x)
        gate, up = blk.ff.gate_proj(h), blk.ff.up_proj(h)
        down = blk.ff(h)
        a = {"x": x, "h": h, "gate": gate, "up": up, "act": F.silu(gate) * up, "down": down, "out": x + down}
        assert torch.equal(a["out"], x + blk.ff(blk.ln_2(x)))
        return {k: v.numpy() for k, v in a.items()}


def export(nanogpt_dir):
    """-> (image, check cases [(x_q, golden, reference)], measured max error in output LSB)."""
    blk = build_block(nanogpt_dir)
    weights = {"norm_w": blk.ln_2.weight.detach().numpy(),
               **{f"w_{p}": getattr(blk.ff, f"{p}_proj").weight.detach().numpy().T for p in ("gate", "up", "down")}}
    calib = np.random.default_rng(SEED).standard_normal((N_CALIB, T, N_EMBD))
    sx = maxabs_scale(calib)
    acts = float_acts(blk, quantize(calib, sx) * sx)
    scales = calib_scales({k: acts[k] for k in ACTS}, sx)
    img = build_image(weights, scales)
    check_bounds(img)

    cases, worst = [], 0.0
    s_out = img.segments["out"]["scale"]
    for x in np.random.default_rng(SEED + 1).standard_normal((N_CHECK, T, N_EMBD)):
        xq = quantize(x, sx)
        g = mlp_block_int(img, xq)
        ref = float_acts(blk, xq * sx)["out"]
        worst = max(worst, float(np.abs(g["out"] * s_out - ref).max()) / s_out)
        cases.append((xq, g, ref))
    return img, cases, worst


def float_attention(blk, x):
    """Float activations of nanoGPT's attn(ln_1(x)) (CausalSelfAttention, n_head=1) on x (..., T, d)."""
    import torch
    import torch.nn.functional as F

    attn = blk.attn
    with torch.no_grad():
        x = torch.as_tensor(x, dtype=torch.float64)
        h = blk.ln_1(x)
        q, k, v = attn.c_attn(h).split(attn.n_embd, dim=-1)
        causal = torch.ones(T, T, dtype=torch.bool).tril()
        scores = q @ k.transpose(-2, -1) / math.sqrt(attn.head_dim)
        p = F.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
        o = p @ v
        y = attn.c_proj(o)
        assert torch.allclose(y, attn(blk.ln_1(x.reshape(-1, T, N_EMBD))).reshape(y.shape), rtol=0, atol=1e-12)
        a = {"x": x, "h": h, "q": q, "k": k, "v": v, "kT": k.transpose(-2, -1), "scores": scores.masked_fill(~causal, 0),
             "p": p, "o": o, "y": y}
        return {k: v.numpy() for k, v in a.items()}


def export_attention(nanogpt_dir):
    """-> (image, check cases [(x_q, golden, reference)], measured max error in y LSB)."""
    blk = build_block(nanogpt_dir)
    w = blk.attn.c_attn.weight.detach().numpy().T
    weights = {"norm_w": blk.ln_1.weight.detach().numpy(),
               **{f"w_{p}": w[:, i * N_EMBD:(i + 1) * N_EMBD] for i, p in enumerate(attention.PROJ)}}
    weights["w_o"] = blk.attn.c_proj.weight.detach().numpy().T
    calib = np.random.default_rng(SEED).standard_normal((ATTN_N_CALIB, T, N_EMBD))
    sx = maxabs_scale(calib)
    acts = float_attention(blk, quantize(calib, sx) * sx)
    scales = calib_scales(acts, sx)
    img = attention.build_image(weights, scales, T, fixed_point, quantize, maxabs_scale, rmsnorm_lut)
    attention.check_bounds(img)

    xq = quantize(np.random.default_rng(SEED + 1).standard_normal((ATTN_N_CHECK, T, N_EMBD)), sx)
    cases = [(x, attention.attention_int(img, x), ref) for x, ref in zip(xq, float_attention(blk, xq * sx)["y"])]
    s_y = img.segments["y"]["scale"]
    return img, cases, max(float(np.abs(g["y"] * s_y - ref).max()) / s_y for _, g, ref in cases)


# fixture case prefix: (export function, output segment, TOL in output LSB)
BLOCKS = {"mlp_block": (export, "out", TOL_LSB), "attention": (export_attention, "y", ATTN_TOL_LSB)}


def fixture_bytes(nanogpt_dir, tmp, block):
    fn, out, tol_lsb = BLOCKS[block]
    img, cases, worst = fn(nanogpt_dir)
    tol = tol_lsb * img.segments[out]["scale"]
    files = {}
    for i, (xq, g, ref) in enumerate(cases[:N_FIXTURES]):
        path = os.path.join(tmp, f"{block}_{i}.json")
        fixture.save(path, f"{block}_{i}", img, {"x": xq}, g, {out: ref}, tol)
        files[f"{block}_{i}.json"] = open(path, "rb").read()
    return img, cases, worst, files


def segment_stats(img, got, ref):
    """{segment: (least-squares gain, RMS error in LSB)} of each golden segment against its float activation;
    scores only where causal (the float reference zeroes the rest)."""
    causal = np.tril(np.ones((T, T), dtype=bool))
    stats = {}
    for k, g in got.items():
        s = img.segments[k]["scale"]
        q, r = g * s, ref[k]
        if k == "scores":
            q, r = q[..., causal], r[..., causal]
        stats[k] = (float((q * r).sum() / (r * r).sum()), float(np.sqrt(((q - r) ** 2).mean())) / s)
    return stats


def segment_errors(block, stats):
    """Segments whose held-out gain or RMS error leaves SEGMENTS' bounds (contract 8)."""
    assert set(stats) == set(SEGMENTS[block]), f"{block}: segments {sorted(stats)}, bounds for {sorted(SEGMENTS[block])}"
    return {k: f"gain {g:.4f} (HEAD {g0}), RMS {r:.3f} LSB (HEAD {r0})" for k, (g, r) in stats.items()
            for g0, r0 in [SEGMENTS[block][k]] if abs(g - g0) > GAIN_TOL or r > r0 * RMS_TOL}


def heldout(nanogpt_dir, block, img, gain=1):
    """-> per-input max error in output LSB, each calibrated float activation's peak in its own LSB, and
    segment_stats.
    `gain` scales the N(0,1) inputs per channel (OUTLIER: one channel x4, a non-Gaussian set)."""
    s = img.segments
    xq = quantize(np.random.default_rng(HELDOUT_SEED).standard_normal((HELDOUT_N, T, N_EMBD)) * gain, s["x"]["scale"])
    blk = build_block(nanogpt_dir)
    if block == "mlp_block":
        out, ref, got = "out", float_acts(blk, xq * s["x"]["scale"]), mlp_block_int(img, xq)
    else:
        out, ref, got = "y", float_attention(blk, xq * s["x"]["scale"]), attention.attention_int(img, xq)
    err = np.abs(got[out] * s[out]["scale"] - ref[out]).max(axis=(1, 2)) / s[out]["scale"]
    peak = {k: float(np.abs(v).max()) / s[k]["scale"] for k, v in ref.items() if k not in ("x", "p")}  # p: fixed 1/ONE
    return err, peak, segment_stats(img, got, ref)


def check(nanogpt_dir, block, a, b):
    fn, out, tol_lsb = BLOCKS[block]
    img, cases, worst, files = fixture_bytes(nanogpt_dir, a, block)
    assert fixture_bytes(nanogpt_dir, b, block)[3] == files, f"{block} export is not deterministic"
    for name, data in files.items():
        path = fixture.FIXTURE_DIR / name
        assert path.exists() and path.read_bytes() == data, f"{path} is stale; rerun python3 -m xf.export"
    assert img.addr_bits == 12 and len(img.words) <= 1 << 12
    golden_fn = mlp_block_int if block == "mlp_block" else attention.attention_int
    for x in (np.full((T, N_EMBD), -128), np.full((T, N_EMBD), 127), *np.random.default_rng(SEED + 2).integers(-128, 128, (100, T, N_EMBD))):
        golden_fn(img, x)
    if block == "mlp_block":
        bounds = check_bounds(img)
    else:
        bounds = attention.check_bounds(img)
        err = attention.row_sum_error(img.read("exp_lut"), int(img.read("one")[0]), T)
        assert err <= 1, f"softmax rows sum to one within {err} LSB, not 1"
    assert worst <= tol_lsb, f"{block}: max error {worst:.3f} LSB exceeds TOL {tol_lsb}"
    err, peak, stats = heldout(nanogpt_dir, block, img)
    assert max(peak.values()) <= 127.5, f"{block}: activations saturate on held-out inputs: {peak}"
    over = int((err > tol_lsb).sum())
    assert block not in HELDOUT_TOL or not over, f"{block}: {over} of {HELDOUT_N} held-out inputs exceed TOL, max {err.max():.3f} LSB"
    bad = segment_errors(block, stats)
    assert not bad, f"{block}: held-out segments off their float activations: {bad}"
    out_err, out_peak, _ = heldout(nanogpt_dir, block, img, OUTLIER)
    assert max(out_peak.values()) <= 127.5 and out_err.max() <= tol_lsb, \
        f"{block}: outlier-channel inputs reach {out_err.max():.3f} LSB, peak {max(out_peak.values()):.1f}"
    print(f"ok {block}: deterministic, {len(img.words)}/{1 << 12} words, bounds {bounds}, "
          f"{len(cases)} inputs within TOL, max error {worst:.3f} LSB <= {tol_lsb} LSB "
          f"= {tol_lsb * img.segments[out]['scale']:.6f}; {HELDOUT_N} held-out: max error {err.max():.3f} LSB, "
          f"{over} over TOL, peak activation {max(peak.values()):.1f} LSB, {len(stats)} segments within gain and RMS bounds; "
          f"outlier channel: max error {out_err.max():.3f} LSB")


def main(argv=None):
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify instead of writing fixtures")
    args = ap.parse_args(argv)
    nanogpt_dir = os.environ.get("NANOGPT_DIR")
    if not nanogpt_dir:
        sys.exit("set NANOGPT_DIR to a nanoGPT Extended checkout @ 03dbab8")

    for block in BLOCKS:
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            if args.check:
                check(nanogpt_dir, block, a, b)
                continue
            img, _, worst, files = fixture_bytes(nanogpt_dir, a, block)
            fixture.FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            for name, data in files.items():
                (fixture.FIXTURE_DIR / name).write_bytes(data)
            print(f"wrote {len(files)} {block} fixtures, {len(img.words)} words, "
                  f"max error {worst:.3f} LSB (TOL {BLOCKS[block][2]})")


if __name__ == "__main__":
    main()
