# AUDIT-7 — Test coverage (round 2)

Lens: would the cocotb and pytest suites catch real bugs in the RTL, kernels, export and golden model?
Do the round-1 coverage fixes (SG11, SG12) kill the mutants AUDIT-3 described, and is the code added
since (SG9 RMSNorm, SG10 calibration headroom) covered? Audited at `ab89cb9`, iverilog 12.0,
sv2v 0.0.13, cocotb 2.0.1, torch 2.14.1, nanoGPT Extended `03dbab8`. `make test_all` passes at HEAD
with `NANOGPT_DIR` set (424 passed, 4 xfailed in pytest). AUDIT-1..6 findings are not repeated.

Method: 23 hand-placed mutants, one plausible bug each, run against the targets that cover the
mutated code. An exporter mutant is followed by `python3 -m xf.export` (any exporter change has to
regenerate the fixtures, or `test_export_matches_fixtures` fails), then pytest and `export --check`.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | Without torch and `NANOGPT_DIR`, the merge gate checks tolerance on 4 fixtures per block only: a full revert of SG10 passes `make test_all` | below |
| H2 | High | Tolerance is a per-element max on the final output only, so systematic gain bugs inside either chain pass every check, `export --check` included | below |
| M1 | Medium | The L1-D hit path runs in one matrix configuration; read-hit and write-hit tag mutants pass the L1-D tests and 8-bit regression | below |
| L1 | Low | AUDIT-3 M1 and L2 are still open: the overflow guards and `fixed_point`'s rounding term can be broken with no test failing | below |

Checked and covered (mutant killed):
- **SG11.** Both AUDIT-3 H1 mutants (write hit leaves the line stale; a fill never sets `valid`) fail
  `make test_l1d_indepth PARAMS=-Pgpu.DCACHE_LINES=16` and `make test_xf_l1d` with the same params.
  Ignoring MeDiC's bypass (`if (!(MEDIC && warp_low_utility[i]))` → `if (1)`) fails
  `test_l1d_indepth` under `DCACHE_LINES=16 MEDIC=1`.
- **SG12.** The AUDIT-3 H2 I-cache tag mutant fails `make regression PARAMS=-Pgpu.ICACHE_LINES=16`,
  which `matrix_icache` runs.
- **SG9 kernel and binding.** Each of these fails `make test_xf_rmsnorm`: `lutmax = n` instead of
  `n - 1` (the `Σx² = 1024` row), one normalization trip fewer (`max(kmax - 1, 1)`), the rounding term
  dropped, the rounding term not scaled by `2^k`, and the upper clamp at 126.
- **SG10 and SG9 export, with `NANOGPT_DIR` set.** Headroom 1.0 everywhere (5 pytest failures), no
  headroom on `v` (`test_attention_tol_on_heldout`), softmax temperature 1.1 (same), and the rsqrt LUT
  built with `d + 1` (11 failures). No headroom on `act` passes pytest and is caught only by
  `export --check`'s saturation assert (see H1).

## H1 — The merge gate drops every held-out tolerance check without torch (High)

The Dispatch protocol's full suite is `make test_all` and `make -j4 matrix`. Neither sets or requires
`NANOGPT_DIR`, and neither runs `python3 -m xf.export --check`. Without them, pytest reports
`413 passed, 13 skipped, 2 xfailed` and exits 0 (15 skipped once AUDIT-6's `test_audit6.py` landed). Round 1 (AUDIT-3 M2) counted 5 skips. Since then the
torch-gated set has grown to every test that checks more than the 4 committed fixtures per block:
`test_audit1` (MLP at 4 input scales, the SG9 acceptance), `test_audit2` (200k held-out, the SG10
acceptance), `test_audit5` (louder inputs) and both fixture-freshness checks. The cocotb chain tests
check the same 4 fixtures, and their golden values are regenerated together with the exporter.

So an exporter regression whose 4 fixtures still land within TOL passes the whole merge gate. Each
of these passes torch-free pytest after re-export:

| mutant (after `python3 -m xf.export`) | torch-free pytest | pytest with `NANOGPT_DIR` | `export --check` |
|---|---|---|---|
| `HEADROOM = 1.0` (SG10 reverted) | pass | 5 failed | fails (saturation) |
| no headroom on `v` | pass | 1 failed | fails (saturation) |
| softmax temperature ×1.1 (`exp_lut`) | pass | 1 failed | fails (86 of 200k over TOL) |
| no headroom on `act` | pass | **pass** | fails (`act` peaks at 133.7 LSB) |

The last row shows a second gap. Contract 8's "no activation saturates on the 200k held-out inputs"
is asserted only in `export.check`, and pytest never calls it, even with torch.

Repro:
```bash
sed -i 's/HEADROOM, HELDOUT_SEED, HELDOUT_N = 1.5,/HEADROOM, HELDOUT_SEED, HELDOUT_N = 1.0,/' xf/export.py
NANOGPT_DIR=... python3 -m xf.export                 # regenerate fixtures
env -u NANOGPT_DIR make test_all                     # passes
NANOGPT_DIR=... python3 -m pytest xf/tests -q        # 5 failed
git checkout xf/export.py test/fixtures
```

Fix direction (SG15): make `test_all` fail when torch or `NANOGPT_DIR` is missing rather than skip,
run `python3 -m xf.export --check` from it, and move the held-out saturation assert into a pytest
(or call `export.check` from one) so it runs with the rest. Then name `NANOGPT_DIR` in the Dispatch
protocol's full-suite command. Test-only.

## H2 — Output-only tolerance lets gain bugs inside the chains through (High)

Every tolerance check (fixtures, `test_audit1/2/5`, `export --check`) bounds the per-element max error
of the final output, `out` or `y`. The MLP has 0.6 LSB of slack against TOL = 2 (1.39 LSB over 20k
held-out inputs), and `out = x + down` is dominated by the residual (AUDIT-5 M1). So a bug that skews
an intermediate by a few percent moves the output's max error by a fraction of an LSB and passes.
Three plausible one-line exporter bugs pass every check, `export --check` on 200k inputs included:

| mutant (after re-export) | effect | output max err (20k) | least-squares gain of the bugged segment |
|---|---|---|---|
| none (HEAD) | | `out` 1.39, `y` 2.27 | `act` 1.0044, `down` 1.0036, `out` 1.0001, `o` 1.0008, `y` 0.9998 |
| SiLU LUT indexed one gate LSB off (`np.arange(-127, 129)`) | `silu(g + 1 LSB)` | `out` 1.66 | `act` **1.0339**, `down` **1.0344** |
| `o` requant uses `1/(ONE+1)` for `p` | `o`, `y` 0.8 % low | `y` 2.50 | `o` **0.9923**, `y` **0.9913** |
| residual `x` multiplier ×1.005 | `out` 0.5 % high | `out` 1.38 | `out` **1.0049** |

Gain is `Σ(dequant·ref) / Σ(ref²)` against nanoGPT's float activation for that segment
(`export.float_acts` / `float_attention`). RMS error separates the SiLU case too (`down` 0.754 → 1.008
LSB). Max error per segment does not separate it: `act` and `down` already reach 3.8 and 4.4 LSB at HEAD.

Repro (each one, then `NANOGPT_DIR=... python3 -m xf.export && NANOGPT_DIR=... python3 -m pytest xf/tests -q
&& NANOGPT_DIR=... python3 -m xf.export --check`, which all pass):
```bash
sed -i 's/g = np.arange(-128, 128) \* sc\["gate"\]/g = np.arange(-127, 129) * sc["gate"]/' xf/export.py
sed -i 's|fixed_point(\[sc\["v"\] / ONE / sc\["o"\]\]|fixed_point([sc["v"] / (ONE + 1) / sc["o"]]|' xf/attention.py
sed -i 's|fixed_point(\[sc\["x"\] / sc\["out"\], |fixed_point([1.005 * sc["x"] / sc["out"], |' xf/export.py
```

Fix direction (SG16): in `export --check` and a pytest, compare every non-input golden segment of both
blocks against its float activation on the held-out inputs. Assert gain within a measured bound
(e.g. `|gain − 1| ≤ 1 %` for intermediates, `≤ 0.25 %` for `out` and `y`) and RMS error within a bound
frozen from HEAD, recorded in contract 8 next to TOL. Each of the three mutants above must fail.
This adds bounds to contract 8, so per the Rules it goes to Ashton before the build. It should land after
SG13 and SG14 (AUDIT-6 H1), which change the attention chain.

## M1 — The L1-D hit path is exercised by one matrix configuration (Medium)

MeDiC never installs a line (SG11 open issue), so under `MEDIC=1` the cache is a pass-through. The
matrix's `all` configuration includes `MEDIC=1`, so `dcache` is the only configuration with cache hits,
and the cache never meets ICACHE, PIPELINED or COALESCE with hits. Two tag mutants of
`src/l1d_cache.sv` pass `test_l1d_indepth`, `test_xf_l1d` and the 8-bit `regression` under
`DCACHE_LINES=16`, because no directed test touches two addresses that share a line:

| mutant | passes (DCACHE_LINES=16) | caught by |
|---|---|---|
| a read hit ignores the tag | `test_l1d_indepth`, 8-bit `regression` | `test_xf_matmul` (all 5 tests) |
| a write-through updates a valid line whatever its tag | `test_l1d_indepth`, `test_xf_l1d`, 8-bit `regression` | `test_xf_matmul`, `test_xf_mlp_block` |

Both fail `matrix_dcache` only.

Repro:
```bash
sed -i 's/^\( *\)&& tag_arr\[idx_of(consumer_read_address\[i\])\] == tag_of(consumer_read_address\[i\])) begin/\1) begin/' src/l1d_cache.sv
make test_l1d_indepth PARAMS=-Pgpu.DCACHE_LINES=16 && make regression PARAMS=-Pgpu.DCACHE_LINES=16   # pass
make test_xf_matmul PARAMS=-Pgpu.DCACHE_LINES=16                                                   # fails
git checkout src/l1d_cache.sv
```

Recommendation: a directed test that reads `a`, then `a + DCACHE_LINES`, then writes `a + DCACHE_LINES` and
re-reads `a`, and an `all` configuration with `MEDIC=0`.

## L1 — Round-1 discriminator gaps still open (Low)

With torch and `NANOGPT_DIR`, these still pass all of pytest (424 passed, 4 xfailed):
`export.check_bounds` and `attention.check_bounds` never raising (`if bad:` → `if False:`; AUDIT-3 M1),
and `fixed_point` ignoring the rounding term (AUDIT-3 L2). Since SG9 that term also carries the
`kmax` normalization shift, so it now guards the RMSNorm requant against int32 overflow.
