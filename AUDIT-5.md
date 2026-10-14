# AUDIT-5 — Correctness against the reference (round 2)

Lens: do the kernels, the integer golden model and the export faithfully implement nanoGPT Extended's
`Block` (pre-LN, RMSNorm, SwiGLU, single-head causal attention), and are the round-1 fixes (SG9–SG12)
correct, not only green. Audited at `1eadf70` against nanoGPT Extended `03dbab8`, cocotb 2.0.1,
torch 2.14.1. Kernel = golden is proven bit-exact by the `test_xf_*` targets, so a golden-level
divergence from nanoGPT is a kernel-level divergence. AUDIT-1..4 findings are not repeated.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | The attention head has no `ln_1`, so its correctness depends on input loudness: 55.8 LSB at 3× the calibration scale, vacuous below 0.1× | `xf/tests/test_audit5.py` (2 strict xfails) |
| M1 | Medium | TOL is asserted on the residual output only; under nanoGPT's own init the MLP branch is 0.011 output LSB, so deleting the MLP would pass | below |
| L1 | Low | `rsqrt_index` truncates `Σx² >> 2k`, so the normalized `h` is biased up by up to 0.19 % (mean +0.054 %) | below |

## Round-1 fixes: checked

- **SG9 (RMSNorm).** Over every reachable `Σx² ∈ [1, 8·128²]` the LUT value `lut[idx]/2^(s+k)` is
  within 0.194 % of nanoGPT's `rsqrt(mean(x²)+eps)` (eps scales by `4^k` on normalized rows, negligible
  since `sx²·Σx²/d ≥ 0.17` there). The kernel's branch-free step sets `f = 1` exactly when
  `Σx² ≥ len`, the golden's condition; `Σx² = 1024` (`[32,0,…]`) and mixed-`k` rows in one block are
  in `test_xf_rmsnorm`. `h` never saturates: the RMSNorm output is bounded by `√d·max|w|` = 4.16,
  under `127·scale(h)` = 6.05. One-hot, constant, ±127, uniform-int8, sparse and 2×-Gaussian inputs
  all stay within 1.38 LSB of nanoGPT.
- **SG10 (headroom).** `export --check` reproduces contract 8 (MLP 1.060 / 1.418 LSB, attention
  2.107 / 2.751 LSB). On the 200k held-out inputs no *integer* golden intermediate saturates either
  (largest |value|: MLP 93, attention 96; only `p = 127` on causal row 0, which is exact).
- **Generalization.** Re-exporting with seeds 1–7 keeps both blocks within TOL on 20k held-out inputs
  (MLP ≤ 1.24 LSB, attention ≤ 2.74 LSB), so the TOL is not tuned to seed 0. nanoGPT's
  `_init_weights` also stays within TOL, but see M1.
- **SG11/SG12** are test-infrastructure fixes; `make test_all` passes at `1eadf70` with
  `NANOGPT_DIR` set.
- **Semantics re-checked:** `Block.forward` is `x + attn(ln_1(x))` then `x + ff(ln_2(x))`;
  `SwiGLUMLP` is `down(silu(gate(x)) * up(x))`; `RMSNorm` is `x·rsqrt(mean(x²)+eps)·w`; softmax
  masks `j > i` in both the golden and the kernel; `1/√head_dim` is folded into `scores_M`.

## H1 — Attention without `ln_1` is correct only at the calibration loudness (High)

nanoGPT's attention never sees raw `x`: the block runs `attn(ln_1(x))`, and `ln_1` makes the
attention input scale-invariant. The project's head is calibrated and verified on raw `N(0,1)` `x`.
Its q, k, v and scores scale with `x` (scores quadratically), so a louder input saturates and a
quieter one carries no signal. AUDIT-1 checked 0.05–1.0× and found it within TOL; below about 0.1×
that check is vacuous, and above 1.25× it fails.

Max error vs nanoGPT, `rng(7).standard_normal((2000, 4, 8))·a` quantized at `scale(x)` (attention TOL 3,
MLP TOL 2 LSB):

| a | attention | over TOL | MLP block |
|---|-----------|----------|-----------|
| 1.0 | 2.01 | 0 | 1.20 |
| 1.5 | 3.20 | 1 | 1.28 |
| 2.0 | 4.40 | 5 | 1.34 |
| 3.0 | 55.81 | 258 | 1.12 |

Other int8 inputs: uniform int8 98.6 LSB (24.6 % over TOL), ±127 sign patterns 325.6 LSB (98.6 %),
constant rows 126.1 LSB. In each case q, k, v and scores peak at 1.5–6× their int8 range.
The MLP block, which has its RMSNorm in front, stays within 1.38 LSB on all of these.

Quiet inputs are the opposite problem: an all-zero `y` is within TOL of nanoGPT on 99.9 % of inputs at
a = 0.05 and on 73.8 % at a = 0.1, because the attention output is only a few `y` LSB there.

So the single head is verified only for inputs about as loud as `N(0,1)`, and it is not the attention
path of a nanoGPT block. This extends round-1 M1 (no full block). That was Medium because the halves
were each believed correct; this audit shows the attention half is not correct off its calibration scale.

Repro: `NANOGPT_DIR=... python3 -m pytest xf/tests/test_audit5.py -rxX`. The two attention cases at
a = 2, 3 are `xfail(strict=True)`. The MLP cases at a = 1.5, 2, 3 pass and act as the control.

Fix direction (SG13): run `ln_1` in front of the head, as an RMSNorm launch on the existing kernel
and `rmsnorm_int`, with the attention image gaining `ln1_w`, `rsqrt_lut`, `norm_s` and `h` segments.
Calibrate on `ln_1(x)` and use `x + attn(ln_1(x))` as the reference, which also covers M1's residual.
Then re-measure the attention TOL across input scales. This changes the SG8 attention fixture layout
and contract 8's attention entry, not contracts 1–7.

## M1 — Output-only TOL is dominated by the residual (Medium)

TOL is asserted on `out = x + down`. With the current PyTorch default init, the MLP branch is large
enough that golden mutations are caught on held-out inputs:

| mutation of the golden chain | max error | inputs over TOL |
|---|---|---|
| none | 1.39 | 0 % |
| `down = 0` | 20.41 | 100 % |
| SiLU LUT → ReLU | 7.47 | 99.7 % |
| `down` halved | 10.41 | 97.3 % |

With nanoGPT's own `_init_weights` (N(0, 0.02), `down_proj`/`c_proj` scaled by 1/√2, norm weights 1;
round-1 L2), the MLP branch peaks at **0.011 output LSB** (mean 0.0013) over 20k inputs. Any MLP chain,
including none at all, then passes TOL and every fixture check, and the MLP export reports a constant
0.34 LSB that is pure `x` requant error.

The attention side has a weaker version of the same problem. A softmax that ignores the scores
(uniform causal `p`) fails only 68.4 % of inputs, and `1/hd` in place of `1/√hd` fails only 43.3 %.
`export --check` still catches both on its 2000 + 200k inputs, but a 4-input fixture check would let
the second through with probability about 0.1.

Recommendation: also assert TOL on the branch itself (`down` against `ff(ln_2(x))`, in `down` LSB)
before any switch to nanoGPT init. Not promoted, since the current weights are discriminating.

Repro (torch + `NANOGPT_DIR`):
```
python3 -c "
import os, numpy as np, torch; from torch import nn; from xf import export as E
b = E.build_block(os.environ['NANOGPT_DIR'])
with torch.no_grad():
    for n, m in b.named_modules():
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02 / (2 ** .5 if n.endswith(('c_proj', 'down_proj')) else 1))
    b.ln_2.weight.fill_(1)
E.build_block = lambda d: b
img, _, w = E.export(os.environ['NANOGPT_DIR']); so = img.segments['out']['scale']
x = E.quantize(np.random.default_rng(12345).standard_normal((20000, 4, 8)), img.segments['x']['scale'])
print(w, np.abs(E.float_acts(b, x * img.segments['x']['scale'])['down']).max() / so)"
```
This prints `0.34…` and `0.0114…`.

## L1 — `rsqrt_index` truncates, so `h` is biased up (Low)

`idx = Σx² >> 2k` floors, so `lut[idx]` is the rsqrt of a slightly smaller `Σx²` and `h` is always
slightly too large on normalized rows (`k > 0`). Relative error is +0.033 % mean / 0.147 % max for
`Σx² ∈ [1024, 4096)` and +0.055 % / 0.194 % above 4096, against ±0.017 % for `k = 0`. At `|h|` ≈ 87
LSB that is up to 0.17 LSB of systematic bias. It is within SG9's 2^-8 criterion. Rounding the index
(`(Σx² + 2^(2k-1)) >> 2k`, clamped to `len - 1`) would centre it.

Repro: `xf/tests/test_export.py::test_rsqrt_relative_error_bounded` with `print((got / want - 1).mean())`
added (+5.4e-4 over all `Σx²`; it would be ≈ 0 if unbiased).
