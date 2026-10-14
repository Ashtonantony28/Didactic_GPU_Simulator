# AUDIT-6 — Numerical tolerance (round 2)

Lens: do contract 8's tolerances (MLP block 2 output LSB, attention 3 output LSB) bound the error
against nanoGPT on inputs the tests do not sample, now that SG9 (exact-index rsqrt) and SG10
(`HEADROOM` = 1.5) are merged? Audited at `1eadf70` against nanoGPT Extended `03dbab8`, torch
2.14.1, with the committed fixture images (identical to a fresh export; `export --check` passes).
Errors are measured as in `export --check`: golden integer chain vs nanoGPT on `dequant(x_q)`, in
output LSB. "Unsaturated" means every float activation stays within ±127.5 of its own LSB.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | Attention TOL is not a bound: unsaturated inputs reach 10.66 LSB, from int8 rounding of `q` and `k` | `xf/tests/test_audit6.py` (2 strict xfails) |
| L1 | Low | The "held-out" set `rng(12345)` tuned `HEADROOM`, so `export --check` no longer measures unseen inputs | below |

AUDIT-2 M3 (attention 3.31 LSB unsaturated at 1M Gaussian inputs) is the Gaussian tail of H1 and is
folded into it.

Checked and sound:
- **MLP TOL = 2 LSB holds on every input tried.** Fresh seeds 2026 and 777, 1M inputs each: max 1.544
  and 1.595 LSB, peak activation 98.3 and 100.6 LSB. N(0, σ²) for σ = 0.01, 0.03, 0.1, 0.3, 0.5, 1.25,
  1.5, 2, 3, unit-variance Laplace, Student-t (df 3), one channel ×4, one-hot rows and uniform int8
  (200k each): max 1.542 LSB, none over TOL. Hill-climbing 4096 int8 inputs for 3000 steps (two
  seeds) toward the largest unsaturated error reaches 1.72 LSB; climbing toward the largest
  activation reaches 104.4 LSB, so no int8 input was found that saturates the block. RMSNorm bounds
  ‖h/w‖ = √8, which caps `h`, `gate`, `up` analytically at 87, 88, 87 LSB for any input.
- **SG9 rsqrt.** For every Σx² in [1, 8·128²], `rsqrt_lut[idx] / 2^k` is within 0.19 % of the
  exact `1/sqrt(scale(x)²·Σx²/d + eps)` factor (index truncation alone ≤ 0.39 % in Σx²). Quiet
  inputs at σ = 0.01 and 0.03 stay at 1.15 and 1.40 LSB.
- **SG10 headroom.** No MLP or attention activation saturates on the 2M fresh-seed Gaussian inputs
  (attention peak 107.4 and 110.1 LSB).
- **Requant multipliers.** Every `M` is ≥ 6667, so `round(m·2^s)` adds < 1e-4 relative error.

## H1 — Attention TOL is not a bound (High)

`scores = q·kᵀ/√hd` is computed from int8 `q` and `k`. Rounding each to its LSB leaves ±0.5 LSB per
element, and in the dot product that error is multiplied by the other operand: Σ (δq·k + q·δk) over
hd = 8 terms. When `q` and `k` are large and the rounding signs line up, the score error is several
`scores` LSB, softmax turns it into a large `p` error, and `p @ v` carries it into `o` and `y`. The
Gaussian test distribution rarely lines the signs up, so the sampled maxima (2.75 LSB over 200k)
look like a bound; they are not.

The input below saturates nothing (largest activation `k` at 100.9 LSB) and the golden chain is
10.66 y LSB from nanoGPT, 3.6× TOL:

```
x = [[60,-18,-43,4,-78,28,41,47], [-105,-24,30,-22,28,-6,-37,-23],
     [6,56,65,-101,-21,-80,-36,-50], [-31,83,116,71,25,38,-62,-116]]
```

| stage | peak (own LSB) | max error (own LSB) |
|---|---|---|
| q, k, v | 87.6, 100.9, 89.4 | 0.83, 0.68, 0.61 |
| scores, from rounded q and k alone (row 3) | 34.5 | 1.93, −0.66, −2.85, −1.80 |
| p, row 3 | — | 0.543 vs 0.481, 0.354 vs 0.418 (8 p LSB) |
| o | 70.8 | 10.25 |
| y | 50.9 | 10.66 |

Turning on one rounding at a time in a float emulation (weights already quantized) attributes it:
`q`/`k` rounding alone gives 6.47 LSB on this input; `v`, `scores`, `p`, `o` and `y` rounding give
0.32, 0.43, 0.27, 0.61 and 0.50. All of them together give 8.45; weight quantization makes up the
rest. SG10's headroom widens the `q`/`k` LSB and makes it worse: the same input with `HEADROOM` = 1.0
gives 6.14 LSB (but then saturates `k`).

It is not confined to adversarial inputs. With nothing saturated, natural non-Gaussian inputs cross
TOL far more often than the Gaussian tail does:

| inputs (200k) | over TOL | max | peak activation |
|---|---|---|---|
| N(0,1) held-out (contract 8) | 0 | 2.75 LSB | 97.3 |
| one channel ×4, rest N(0,1) | 61 | 4.49 LSB | 123.7 |
| one-hot rows, uniform int8 value | 2 | 3.10 LSB | 69.7 |

Repro (torch + `NANOGPT_DIR`, both `xfail(strict=True)`; each asserts the input is unsaturated):

```
NANOGPT_DIR=... python3 -m pytest xf/tests/test_audit6.py -rxX --runxfail
# test_attention_tol_unsaturated_worst: 10.66 LSB
# test_attention_tol_outlier_channel:   61 of 200000 inputs, max 4.49 LSB
```

SG13 (AUDIT-5 H1) will put `ln_1` in front of the head and recalibrate, which bounds every row's RMS
and should shrink this. A float emulation of that chain (nanoGPT `ln_1`, output rounded to int8,
scales recalibrated with the same headroom on 65,536 normalized inputs, then this golden attention
chain) stays at 2.52 LSB on the 200k held-out inputs, and the same hill climb over the raw input
reaches only 2.83 LSB, flat from step 1000 to 2000. So SG14 runs after SG13 and starts by
re-measuring on the new chain.

Fix direction (SG14, after SG13): either give `scores` more precision than int8 `q`/`k` (for example, wider
`q`/`k` operands for the scores matmul, or folding `Wq·Wkᵀ` so only one operand is rounded), or
derive the attention TOL from a worst-case error bound rather than a Gaussian sample and re-record
it. Either changes contract 4 (matmul operands), the attention layout or contract 8's TOL, so under
the plan's rules it goes to the user before the build starts.

## Also found: attention inputs at 1.25× scale saturate (AUDIT-5 H1, SG13)

Found independently and already filed as AUDIT-5 H1 (SG13); the numbers here add to it. The attention
head runs on raw `x`, with no `ln_1` in front, and its scales are calibrated on N(0,1). Any int8 input is representable, but inputs only modestly louder than the
calibration set saturate `scores` (200k inputs per row; each row a different draw):

| inputs | over TOL | max | peak `scores` |
|---|---|---|---|
| N(0, 1.25²) | 2 | 3.81 LSB | 160.8 |
| N(0, 1.5²) (repro below) | 35 | 30.6 LSB | 175.5 |
| N(0, 2²) | 1043 | 50.0 LSB | 268.6 |
| Student-t df 3, unit variance | 6 | 21.1 LSB | 148.1 |
| uniform int8 | 48,865 | 114.9 LSB | 460.4 |

The MLP block, which normalizes first, stays under 1.6 LSB on all of these. With `ln_1` in front
the attention activations are bounded analytically (rows with weighted RMS 1: `q` 44, `k` 52, `v`
50, `scores` 31 LSB), so composing the full block removes the saturation but leaves `scores` using a
quarter of its range under the current scales, which SG13's recalibration on `ln_1` outputs restores.

Repro:
```
NANOGPT_DIR=... python3 -c "
import os, numpy as np; from xf import export as E, attention as A, fixture
img = fixture.load(fixture.FIXTURE_DIR/'attention_0.json')['image']; s = img.segments
xq = E.quantize(np.random.default_rng(99).standard_normal((200000, 4, 8)) * 1.5, s['x']['scale'])
ref = E.float_attention(E.build_block(os.environ['NANOGPT_DIR']).attn, xq * s['x']['scale'])
e = np.abs(A.attention_int(img, xq)['y'] * s['y']['scale'] - ref['y']).max((1, 2)) / s['y']['scale']
print((e > 3).sum(), e.max(), np.abs(ref['scores']).max() / s['scores']['scale'])"
```

## L1 — The held-out set is no longer held out (Low)

`HEADROOM` = 1.5 was chosen because it gives 0 over TOL on `rng(12345)` (AUDIT-2 H1), and SG9/SG10
were accepted on the same 200k inputs, which `export --check` and `test_audit2.py` still use. The
measurement is now in-sample. On untouched seeds the MLP still holds (above); attention has one
input over TOL at seed 777 (3.15 LSB, unsaturated, the H1 mechanism) and none at seed 2026 (max
2.78). Recommend `export --check` measure on a seed nothing was tuned on.

Repro (about a minute):
```
NANOGPT_DIR=... python3 -c "
import os, numpy as np; from xf import export as E, attention as A, fixture
img = fixture.load(fixture.FIXTURE_DIR/'attention_0.json')['image']; s = img.segments; attn = E.build_block(os.environ['NANOGPT_DIR']).attn
x = np.random.default_rng(777).standard_normal((1000000, 4, 8)); e = []
for c in np.split(E.quantize(x, s['x']['scale']), 5):
    e.append(np.abs(A.attention_int(img, c)['y'] * s['y']['scale'] - E.float_attention(attn, c * s['x']['scale'])['y']).max((1, 2)) / s['y']['scale'])
e = np.concatenate(e); print((e > 3).sum(), e.max())"
```
