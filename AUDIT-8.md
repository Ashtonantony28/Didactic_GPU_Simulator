# AUDIT-8 — ISA and kernel consistency (round 2)

Lens: same as [AUDIT-4](AUDIT-4.md) (`docs/isa.md`, `xf/isa.py`, `xf/asm.py`, the RTL decoder/ALU/scheduler,
every `kernels/*.asm`, and the register, args, SMEM and launch conventions between kernels, binders and
the golden model). The focus is code added since round 1: the SG9 rmsnorm kernel (PR #18, merged) and the SG13
`ln_1` launch in front of the attention head (PR #20, audited on its branch at `2b11eec`). Audited at
`6ea801e` + `2b11eec`, with iverilog 12.0, sv2v 0.0.13 and cocotb 2.0.1.

Method:
- **Independent ISA model.** A new instruction-level model, written from `docs/isa.md` alone. It runs lockstep blocks
  with per-thread NZP and 32-bit wrap. It flags divergent branches, reads of uninitialised registers, LDS reads with
  no earlier STS in the same block, and every LDR/STR outside the image's segments (STR outside the launch's output
  segment counts as a violation).
- **Chains.** The model ran the MLP and SG13 attention chains through the real binders: 4 fixtures, all-zero and
  all −128 inputs, plus 36 random int8 inputs. Naive matmul ran at 2, 4 and 8 threads per block, tiled at 4. Result:
  bit-exact against `mlp_block_int` / `attention_int`, with 0 divergent branches, 0 uninitialised reads, 0 stale SMEM
  reads and 0 out-of-segment accesses.
- **rmsnorm fuzz.** A random fuzz of `ew.rmsnorm` against `golden.rmsnorm_int` used d ∈ {1…63}, LUT lengths 1–4000,
  random LUT/M/s and 1, 2, 4 or 8 threads per block. All 1130 accepted cases are bit-exact. The other 65 cases are
  ones where golden raises `OverflowError` (see L1).
- **RTL vs model.** A differential cocotb fuzz (not committed) ran 40 random 92-instruction programs: random
  ADD/SUB/MUL/DIV/SRA/MAX over 32-bit memory operands and edge words (0, 31–33, 2^31−1, 2^31, 2^32−1), on 8
  threads. RTL and model agree on every register in three configurations: default; `PIPELINED=1 ICACHE_LINES=16`;
  and `DCACHE_LINES=16 COALESCE=1 NUM_CORES=1`.
- **RTL runs.** `make test_xf_rmsnorm` passes at `THREADS_PER_BLOCK` 2 and 8. `make test_xf_attention` passes on the
  SG13 branch.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| L1 | Low | Binders do not check input dtypes. An i32 `x` (rmsnorm) or `gate` (swiglu) makes the kernel read past its LUT, where golden raises | `xf/tests/test_audit8.py` (2 strict xfails) |
| L2 | Low | The `rmsnorm.asm` header names args word 5 `kmax`. The binder stores `max(kmax, 1)` because the loop is do-while | below |

There are no critical or high findings, so this audit adds no SG rows.

Round-1 items, re-assessed:
- **AUDIT-4 M1** (thread 0's branch decides for the block, undocumented). Still open, still Medium. SG9's rmsnorm
  adds one more kernel that relies on block-uniform branching. It complies: the `d` and `kmax` trip counts are
  per-launch constants, and the per-row `k` is computed without branches. The model saw no divergence.
- **AUDIT-4 M2** (`matmul_tiled` only works at 4 threads per block). Unchanged. SG13's chain uses it the same way.
- **AUDIT-4 L1–L4** are unchanged. L4 (do-while loops) also covers the rmsnorm `kmax` loop, which is why the binder
  writes `max(kmax, 1)`. MeDiC=1 disabling L1-D is unchanged.

Checked and consistent:
- **No ISA change since round 1.** `git diff c58ff7c 2b11eec -- src docs/isa.md docs/decisions xf/isa.py xf/asm.py`
  is empty. The only kernel change is `kernels/rmsnorm.asm` (SG9). The runner change is limited to the
  `MEM_LATENCY` default (SG12).
- **rmsnorm kernel vs contract 4.** The `norm` loop computes `f = min(max(ss − lutmax, 0), 1)` with two signed
  MAX operations, so `f = 1` exactly when `ss ≥ len`. That is `rsqrt_index`'s step. `ss >>= 2f` uses SRA on a
  non-negative value. The shift accumulates `s + k`, and the rounding term is `(2^s · 2^k) >> 1`, which is 0 when
  `s + k = 0`, matching `golden.requant`. The binder's `s ≤ 30 − kmax` keeps `2^(s+k)` positive in int32. `kmax`
  comes from the binder's `d·128²`, the largest int8 Σx², and `rsqrt_index` is monotone, so `kmax` trips always
  bring `ss` below `len`. The args header, the binder's list, and the kernel's word offsets agree (words 0–9).
- **SG13 chain.** `launches` puts the rmsnorm launch first, on `x → h`, with the fixture's `norm_w`,
  `rsqrt_lut`, `norm_M` and `norm_s`. `attention_int` and `check_bounds` (`worst["h"]` with `rnd(norm_s + kmax)`)
  use the same constants. `alloc_acts` copies `h`, so five extra input sets still fit (1847-word fixture). The PLAN
  layout line, the 1847-word count and `tol = 3·scale(y) = 0.041765` match the fixtures. The MLP uses `ln_2`
  and the attention head uses `ln_1`, so each image has its own LUT.

## L1 — Binders accept non-int8 inputs that index past a LUT (Low)

`ew.rmsnorm` derives `kmax` from `d·128²`. `ew.swiglu` indexes `silu_lut[gate + 128]`. Both assume int8
inputs, and golden checks that assumption (`_i8` raises). The binders check shapes and LUT lengths, never the
input segment's dtype. With an i32 segment, the kernel silently reads another segment instead of failing:

- `x = 1000` in a `[1, 8]` i32 segment with a 1024-entry LUT. Σx² = 8·10⁶ is still ≥ 1024 after 4 steps, so the
  LDR at pc 58 reads word 2594, far past `rsqrt_lut` (words 16–1039). Output is all 0.
- `gate = 300` (i32) reads word 436, past `silu_lut` (words 8–263).

Every exported image passes int8 activations, so the current chains cannot reach this. It is the binder-level
counterpart of the overflow preconditions that `check_bounds` guards for the exporter: 65 of the 1195 fuzz cases
had golden raising `OverflowError` while the binder accepted the arguments and the kernel wrapped.

Repro: `python3 -m pytest xf/tests/test_audit8.py -rxX` (both strict xfail). Fix: have `_same`/`_bind` require
`dtype == "i8"` for the int8 operands (`x`, `gate`, `up`, `a`, `b`, `scores`), as golden does.

## L2 — rmsnorm header mislabels args word 5 (Low)

`kernels/rmsnorm.asm` lists the args as `x, w, lut, out, d, kmax, lutmax, M, s, 2^s`.
`elementwise.rmsnorm` stores `max(kmax, 1)` in word 5, because the `norm` loop is do-while: `kmax = 0`, which
happens when `len > d·128²`, would wrap the counter and run about 2^32 trips (AUDIT-4 L4). The header should say
`max(kmax, 1)`, so a hand-written binder does not pass 0.

Repro: `sed -n 6p kernels/rmsnorm.asm`, then `grep -n "max(kmax, 1)" xf/elementwise.py`.
