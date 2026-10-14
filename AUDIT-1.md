# AUDIT-1 — Correctness against the reference (round 1)

Lens: do the kernels, the integer golden model (`xf/golden.py`, `xf/export.py::mlp_block_int`,
`xf/attention.py::attention_int`) and the export faithfully implement nanoGPT Extended's MLP half
(`x + ff(ln_2(x))`, RMSNorm, SwiGLU) and its `CausalSelfAttention` (n_head=1), not only agree with
each other. Audited at `8412641` against nanoGPT Extended `03dbab8`, cocotb 2.0.1, torch 2.14.1.
Kernel = golden is already proven bit-exact by `make test_xf_mlp_block` / `test_xf_attention`, so a
golden-level divergence from nanoGPT is a kernel-level divergence.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | Integer RMSNorm is not scale-invariant on quiet rows; MLP block exceeds TOL on in-distribution inputs | `xf/tests/test_audit1.py` (3 strict xfails) |
| M1 | Medium | No transformer block is ever run: attention lacks `ln_1` and its residual, and the two halves cannot be chained | below |
| L1 | Low | Exporter hard-codes `EPS = 1e-6` instead of reading `ln_2.eps` | below |
| L2 | Low | Weights use PyTorch default init, not nanoGPT's `GPT._init_weights` | below |
| I1 | Info | `RMSNorm` computes in float32 even after `.double()` | below |

Checked and faithful: `float_attention` is asserted equal (atol 1e-12) to nanoGPT's own
`CausalSelfAttention.forward` (SDPA `is_causal=True`, 1/√head_dim, `c_proj`); the golden and the
softmax kernel mask exactly `j > i`, the row max is taken over unmasked entries only, and
`exp_lut[d+255] = exp(d·scale(scores))` matches max-subtracted softmax; SwiGLU is
`down(silu(gate(h)) * up(h))` with the SiLU LUT indexed at the dequantized gate; the residual adds
the un-normalized `x`; `float_acts` is asserted equal to `x + blk.ff(blk.ln_2(x))`; requant rounds
half-up with an arithmetic shift as contracted. Attention stays within its TOL for inputs scaled
0.05–1.0× the calibration distribution (max 2.05 LSB over 2000 inputs each, seed 7).

## H1 — Integer RMSNorm is not scale-invariant on quiet rows (High)

`rmsnorm_int` looks up `rsqrt_lut[Σx² >> 7]` and the exporter fills each 128-wide bucket with the
rsqrt of its midpoint (`export.build_image`, `mid = (k << shift) + (2^shift - 1)/2`). For bucket 0
(Σx² < 128) every row gets rsqrt(63.5) regardless of its energy, and bucket k has a relative rsqrt
error of about 1/(4k). nanoGPT's RMSNorm output is independent of the row's magnitude, so a quiet
token (small residual-stream norm) is normalized to the wrong size.

At the `h` level (fixture image, nanoGPT RMSNorm formula on `dequant(x)` with the quantized weight):

| row `x_q` | Σx² | `h` error (h LSB) |
|-----------|-----|-------------------|
| `[1,0,0,0,0,0,0,0]` | 1 | 95.1 (golden 14, reference 109) |
| `[2,0,…]` / `[4,0,…]` | 4 / 16 | 82.3 / 54.4 |
| `[4,-2,4,5,-2,-2,5,-2]` | 98 | 13.2 |
| rows with Σx² ≥ 4096 | | ≤ 1.10 |

End to end, MLP block vs nanoGPT `x + ff(ln_2(x))` on `rng(7).standard_normal((2000, 4, 8))·a`,
quantized at `scale(x)` (TOL = 2 LSB):

| a | bucketed LUT (current) | rsqrt exact per row, same requant |
|---|------------------------|-----------------------------------|
| 1.0 (calibration distribution) | 7.77 LSB max, 0.1 % of inputs over TOL | 1.12 LSB |
| 0.7 | 4.74, 0.4 % | 1.41 |
| 0.5 | 8.03, 2.2 % | 1.19 |
| 0.3 | 14.94, 24.5 % | 1.24 |
| 0.1 | 18.00, 93.8 % | 1.14 |

The exact-rsqrt column isolates the cause: every other stage stays within TOL. The worst a = 1.0
input is #1666, whose first token is `[4,-2,4,5,-2,-2,5,-2]` (Σx² = 98, float RMS 0.13); 0.75 % of
N(0,1) rows have Σx² < 1024. PLAN contract 8's "1.76 LSB over 2000" holds for seed 1 only; the
claim that the MLP block is within TOL of nanoGPT for its input distribution does not hold.

Repro: `python3 -m pytest xf/tests/test_audit1.py -rxX` (torch-free, two `h`-level cases), and with
torch + `NANOGPT_DIR`, `test_mlp_block_within_tol_on_fresh_seed` (7.77 LSB). All three are
`xfail(strict=True)` so the fix flips them to failures until the markers are removed.

Fix direction (SG9): make the rsqrt lookup relative, e.g. normalize Σx² by an even power of two
before indexing (a short shift loop, no ISA change) or a log-spaced LUT, so the relative error is
bounded for every Σx² ≥ 1; then re-measure TOL on a fresh seed and on scaled inputs.

## M1 — No transformer block is ever run (Medium)

nanoGPT's `Block.forward` is `x = x + attn(ln_1(x)); x = x + ff(ln_2(x))`. The project verifies
`ff`-half and `attn` separately: `attention_<i>.json` feeds raw `x` to `CausalSelfAttention` (no
`ln_1`, no residual), and the two images have different input scales (attention `scale(x)` =
0.042127, MLP 0.037260) with no requant between `y` and the MLP's `x`. So the deliverable
"a real transformer forward pass" stops at two disconnected halves; the attention half of a block,
and the chain into the MLP half, have no golden chain, kernel launch list or reference check.
Every piece needed exists (RMSNorm, residual-add kernels), so this is composition, not new kernels.
Not promoted to a sub-goal at Medium; recommended as the next stretch after SG9.

Repro: `python3 -c "from xf import fixture as f; print(*(f.load(f.FIXTURE_DIR/n)['image'].segments['x']['scale'] for n in ('attention_0.json','mlp_block_0.json')))"`
→ `0.0421268… 0.0372595…`; `grep -n ln_1 xf/*.py` → only the weight init at `xf/export.py:159`.

## L1 — `EPS` is hard-coded (Low)

`export.EPS = 1e-6` is folded into `rsqrt_lut`; nanoGPT's value is `blk.ln_2.eps`. They agree at
`03dbab8` (`RMSNorm(dim, eps=1e-6)`), but a change upstream would silently diverge, and
`--check` would only notice through the tolerance. Repro: `grep -n "EPS" xf/export.py`.

## L2 — Weights are not nanoGPT-initialized (Low)

`build_block` instantiates `Block` directly, so Linear weights keep PyTorch's Kaiming-uniform init and
norm weights are drawn from U(0.5, 1.5). A `GPT` model applies `_init_weights` (N(0, 0.02), `c_proj`
scaled by 1/√(2·n_layer), norm weights 1). The graph is nanoGPT's, but the per-tensor max-abs
quantization is only exercised on uniform weights, never on the Gaussian (heavier-tailed)
distribution nanoGPT produces. Repro: `grep -n "init" xf/export.py`;
`model.py:146` `_init_weights`.

## I1 — Reference RMSNorm runs in float32 (Info)

`RMSNorm.forward` casts to float32 before the rsqrt, so the "float64" reference carries ~6e-8
relative error in `ln_2`. Negligible against a 0.036 output LSB; noted so later audits do not read the
reference as exact float64.
