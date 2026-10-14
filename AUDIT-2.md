# AUDIT-2 — Numerical tolerance (round 1)

Lens: are the frozen tolerances (contract 8: MLP block 2 output LSB, attention 3 output LSB), the
softmax row-sum bound, the int32 overflow sizing and the calibration actually sound? Audited at
`2f9c807` against nanoGPT Extended `03dbab8`, torch 2.14.1. All measurements use the committed
fixture images (identical to a fresh export) and the same reference as `export --check`: nanoGPT on
`dequant(x_q)`, on held-out inputs `rng(12345).standard_normal((N, 4, 8))`, disjoint from the
calibration (seed 0) and check (seed 1) sets.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | Max-abs calibration leaves a saturation tail: MLP reaches 13.1 LSB, attention 8.3 LSB on held-out inputs | `xf/tests/test_audit2.py` (2 strict xfails) |
| M1 | Medium | The TOL evidence is a 200/2000-input maximum and cannot see a 1-in-2000 tail | below |
| M2 | Medium | TOL excludes input quantization; against nanoGPT on the float input the MLP error reaches 23 LSB | below |
| M3 | Medium | Attention exceeds TOL without any saturation (3.31 LSB in 1M inputs) | below |

AUDIT-1 H1 (bucketed rsqrt LUT, SG9) is the second cause of MLP TOL violations; it is not repeated
here, but H1's MLP test only passes once both SG9 and SG10 land.

Checked and sound:
- **Softmax row sum ≤ 1 LSB.** Re-proved independently of `attention.row_sum_error`: every reachable
  causal row (lengths 1–4, max at 127, all 1 + 256 + 65,536 + 16,777,216 tails) pushed through
  `golden.softmax_lut` itself with the exported LUT gives max |Σp − 127| = 0, 1, 1, 1.
- **exp LUT.** E = 23 is the largest exponent keeping `e·one + S/2` in int32 at T = 4
  (`test_exp_bits_tight`); all 256 entries are distinct and non-zero (1037 … 8,388,608), so no
  `d` underflows to an all-zero row.
- **int32 sizing.** Every `check_bounds` term is 128 × (max column abs-sum, or LUT max) × M plus the
  rounding term, the exact worst case for sign-matched int8 operands. The tightest are attention
  `scores` (1,875,116,032, 87 % of INT32_MAX) and MLP `act` (1,804,562,560); `export --check` and
  the ±128/127/random sweeps pass. The `o` bound (128·one·T) is loose, since Σp ≤ 128.

## H1 — Max-abs calibration leaves a saturation tail (High)

Scales are `max|a|/127` over the calibration set, so any held-out input whose float activation
exceeds the calibration maximum clamps at ±127. The clamp error is unbounded in LSB, and contract 8's
TOLs do not hold on the calibration distribution itself.

**MLP block** (4096 calibration inputs), 200,000 held-out inputs, TOL = 2 LSB:

| | value |
|---|---|
| inputs over TOL | 88 (0.044 %), max 13.08 LSB, 99.99th percentile 4.73 LSB |
| over-TOL rows with a saturated activation | 32 rows, worst 13.08 LSB (input 87179, row `[-17,121,19,4,3,34,-3,-21]`) |
| over-TOL rows from AUDIT-1 H1 (Σx² < 1024, unsaturated) | 54 rows, worst 6.76 LSB |
| inputs with a saturated activation | h 14, gate 24, up 4, act 21, down 5, out 43 |

Calibrating on more inputs only shrinks the tail: 65,536 inputs leave 1 saturated over-TOL row
(3.69 LSB); 2^20 leaves none in 200k, at the price of a coarser `scale(x)`. The remaining over-TOL
rows at 2^20 (59, worst 7.24 LSB) are all AUDIT-1 H1.

**Attention head** (65,536 calibration inputs), 200,000 held-out inputs, TOL = 3 LSB: 4 inputs over
TOL, 8 over 2.5 LSB. The four: 8.30, 5.51, 3.44 LSB (`y` saturated) and 3.90 LSB (`v` saturated).
Inputs with a saturated activation: k 3, v 1, scores 1, o 4, y 5. Over 1,000,000 held-out inputs:
34 over TOL, worst 47.74 LSB (`y` saturated), 30 of the 34 with `y`, `o` or `v` saturated. The SG8 move from 4096 to 65,536
did not remove the tail, it made it rarer than the 5000-input measurement could detect (M1).

Headroom works where more calibration does not. Attention with every non-input scale multiplied by
1.25: max 3.67 LSB, 1 over TOL; by 1.5: max 2.75 LSB, 0 over TOL. Headroom widens each LSB, so the
absolute TOL grows with it and must be re-recorded.

Repro (torch + `NANOGPT_DIR`, both `xfail(strict=True)`):

```
NANOGPT_DIR=... python3 -m pytest xf/tests/test_audit2.py -rxX --runxfail
# test_mlp_tol_on_heldout:       88 of 200000 inputs, max 13.08 LSB
# test_attention_tol_on_heldout: 4 of 200000 inputs, max 8.30 LSB
```

Fix direction (SG10): calibrate every non-input activation with headroom (a fixed factor, or a
high percentile times a factor) chosen so no held-out activation saturates, re-export, and measure
TOL on ≥ 200k held-out inputs in `export --check`.

## M1 — The TOL evidence cannot see the tail (Medium)

`export --check` measures the MLP block on 200 inputs and attention on 2000; contract 8 quotes
maxima over 2000 and 5000. At the measured violation rates (4.4e-4 MLP, 2e-5 attention) a 200-input
check misses the MLP violations with probability 0.9996^200 ≈ 92 %, and a 5000-input check misses
the attention ones with probability ≈ 90 %. The quoted maxima (1.76, 2.11 LSB) are sample maxima,
not bounds; at 200k they are 13.08 and 8.30. Folded into SG10's acceptance (check ≥ 200k inputs).

Repro: `grep -n "N_CHECK" xf/export.py`, then the H1 tests.

## M2 — TOL excludes input quantization (Medium)

The reference runs on `dequant(x_q)`, so TOL measures the kernels' error given an already-int8
input, as contract 8 says. Against nanoGPT on the float input itself (input rounding and clipping at
±127·scale(x) included), the MLP block on the same 200k inputs reaches 23.19 LSB, median 1.00 LSB,
131 inputs over 2 LSB. The residual adds `x` directly, so half an input LSB (0.52 output LSB) is in
every output before the block contributes anything. This is a scope statement, not a bug, but the
goal "within quantization tolerance of PyTorch" reads as end to end. Recommend contract 8 also
record the end-to-end figure once SG9/SG10 land.

Repro:
```
python3 -c "
import os, numpy as np; from xf import export as E, fixture
img = fixture.load(fixture.FIXTURE_DIR/'mlp_block_0.json')['image']; blk = E.build_block(os.environ['NANOGPT_DIR'])
x = np.random.default_rng(12345).standard_normal((200000, 4, 8)); s = img.segments
e = np.abs(E.mlp_block_int(img, E.quantize(x, s['x']['scale']))['out'] * s['out']['scale'] - E.float_acts(blk, x)['out']).max((1, 2)) / s['out']['scale']
print(e.max(), np.median(e), (e > 2).sum())"
```

## M3 — Attention exceeds TOL without saturation (Medium)

Excluding saturated inputs, the attention head's worst error is 2.90 LSB over 200k held-out inputs
and 3.31 LSB over 1,000,000 (four unsaturated inputs over TOL: 3.31, 3.23, 3.17, 3.11 LSB). The
error is the p = 1/127 resolution compounded through `o` and `y`, so TOL = 3 is undersized for the
far tail even after H1 is fixed, and headroom that widens the `o` or `y` LSB moves it either way.
SG10 is accepted at 200k inputs; at 1M it needs either a re-measured TOL (a contract 8 value) or
more `p` resolution.

Repro (the 200k-worst unsaturated input; prints the error in y LSB, then the largest float
activation in its own LSB, below 127.5 so nothing saturates):
```
NANOGPT_DIR=... python3 -c "
import os, numpy as np; from xf import export as E, attention as A, fixture
img = fixture.load(fixture.FIXTURE_DIR/'attention_0.json')['image']; s = img.segments
x = np.array([[43,28,-43,-55,45,28,-40,12],[22,22,4,-32,7,3,-16,26],[-35,-17,-26,18,-39,-73,-16,-48],[10,43,-18,-20,-35,15,2,35]])
f = E.float_attention(E.build_block(os.environ['NANOGPT_DIR']).attn, x * s['x']['scale'])
print(np.abs(A.attention_int(img, x)['y'] * s['y']['scale'] - f['y']).max() / s['y']['scale'],
      max(np.abs(f[k] / s[k]['scale']).max() for k in ('q','k','v','scores','o','y')))"
# 2.904…  98.28…
```
The 1M figures come from `test_attention_tol_on_heldout` with `HELDOUT_N = 1_000_000`, keeping the
inputs whose float `q, k, v, scores, o, y` all stay within ±127.5 LSB.
