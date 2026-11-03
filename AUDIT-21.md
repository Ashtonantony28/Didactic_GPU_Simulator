# AUDIT-21 — SG28 race exclusion and the warps2lat config (final audit)

Lens, all together: (1) is SG28's cross-warp `STR` race exclusion sound, or can a wrong RTL pass because words are
left out; (2) is SG28's failing `test_xf_rmsnorm` at `warps2lat` (and the other targets left out of `warps2lat`) a real
RTL/model bug or a test assumption about 4 threads per block; (3) do `docs/isa.md`, ADR-016, the PLAN contract and the
`xf.isasim` docstrings say what the code does. Findings already recorded in AUDIT-1 to AUDIT-20 are not repeated.

Audited at `829b8a3` (main) with iverilog 12.0, sv2v, cocotb 2.0.1. `src/`, the existing tests, the `xf` library, the
Makefile and the docs are unchanged. Builds lived under `build/a21/` and mutants under `build/mut/` (git-ignored); at
most four simulations at a time. Time-boxed to about two hours; no full matrix run.

## Findings

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| M1 | Medium | `memory_diff` leaves a raced word out whatever value the RTL wrote, so the exclusion accepts results that no order of the racing stores can produce. On a lane-order mutant, H1's word 100 holds 1 or 5 (legal: 4 or 8) and `test_cross_warp::test_race_flagged_and_excluded` passes. The mutant is still caught elsewhere (below), so no wrong RTL passes the matrix today, but the race test itself is blind on the words it exists for | `test_audit21.py::test_m1_raced_word_must_hold_a_legal_outcome` (strict xfail); mutant `rev` below |
| L1 | Low | At one lane per warp (`warps4gto`: TPB 4, W 4) `test_isa_diff`'s `RACED` covers all 12 words `str_conflict{8,32}` writes, so those two fixtures compare nothing there but `warp_issues_total`. Before SG28 they were compared word for word and passed (AUDIT-18: the fixture never lands in the contention window) | `test_audit21.py::test_l1_str_conflict_keeps_a_compared_word_at_one_lane_per_warp` (strict xfail, 8 and 32) |
| L2 | Low | `race_words` over-approximates. `docs/isa.md` and the `xf/isasim.py` docstring say it holds "the words whose final value depends on" a race, but a word with one possible final value is flagged: warp 0 stores 1 then 2, warp 1 stores 2; every order ends 2 (a warp's STRs complete in program order), yet the word is raced and left out. The exact-set guard keeps this visible, so it costs coverage, not soundness | `test_audit21.py::test_l2_deterministic_word_not_raced` (strict xfail, lrr and gto) |
| L3 | Low | Item 2: the `warps2lat` exclusions are test assumptions, not RTL or model bugs, and `test_xf_rmsnorm` does not fail at all. At `829b8a3` it passes at `warps2lat` (2/2, 53 s), as does `test_xf_matmul` (6/6, 193 s), so the Makefile comment "the other xf modules assume 4 threads per block or run for minutes at L10" is wrong for those two, and SG28's reported rmsnorm failure does not reproduce | `test_audit21.py::test_l3_warps2lat_runs_rmsnorm` (strict xfail); `::test_rmsnorm_model_at_warps2lat_shape_matches_golden` (control); table below |

There are no critical or high findings. No configuration in this audit produced a wrong result from the GPU: every RTL
run that a test or scratch harness could compare against `xf.isasim` at `warps2lat` matched outside the declared race
words.

### M1 — the exclusion is value-blind (Medium)

`test/helpers/runner.memory_diff` drops every word in `model.race_words` from the comparison:

```python
return [a for a, (g, w) in enumerate(zip(rtl, model.memory)) if g != w and a not in model.race_words]
```

The race leaves the word's value unspecified among the outcomes of the racing stores, not arbitrary. In H1 (two warps,
every lane stores `threadIdx + 1` to word 100) each `STR` still obeys ADR-016 within the warp, so word 100 must end as
4 (warp 0's lane 3) or 8 (warp 1's lane 3).

Mutant `rev` (`build/mut/rev/src/controller.sv`): the write branch of the channel scan uses consumer
`NUM_CONSUMERS-1-j` in place of `j`, so writes claim channels from the highest consumer down and the lowest lane of one
`STR` wins. Build `warps2lat` (`-Pgpu.THREADS_PER_BLOCK=8 -Pgpu.WARPS_PER_CORE=2 -Pgpu.ICACHE_LINES=16
-Pgpu.PIPELINED=1`, `MEM_LATENCY=10`):

- Observed: `test_cross_warp::test_race_flagged_and_excluded` **passes**. Its log has word 100 = 1 for cases
  (2,1,1), (4,3,3), (3,1,2), (5,2,4) and 5 for (1,2,1), (1,1,1); the model says 4 or 8.
- Expected: a value outside {4, 8} at a raced word is a mismatch.

The same mutant fails `test_cross_warp::test_bar_separated_rounds` (word 100 = 1, model 4) and
`test_divergence::test_str_conflict` (`10 words differ, first at 0: RTL 1, model 4`) in the same config, and every W = 1
config through the `str_conflict` fixtures, so the matrix catches it. A fault that only shows when stores race (for
example one that corrupts the data of a store overtaken by another warp's) would not be caught by any test that uses
the exclusion. The fix is cheap for `STR`-only races: have `xf.isasim` record, per raced word, the set of values the
racing stores can leave (each warp's last store in the epoch, after lane resolution) and have `memory_diff` require the
RTL word to be in it. Words reached through taint (140..147 here) can be checked against the RTL's own raced word.

Repro (no simulator): `test_m1_raced_word_must_hold_a_legal_outcome` replays H1 (`xf.tests.test_audit18._h1_replay`),
sets the RTL image's word 100 and 140..147 to the mutant's 1, and asserts `memory_diff` reports it; today it returns
`[]`. RTL command:

```
make BUILD=build/mut/rev/w2l GPU_SRC="$(ls build/mut/rev/src/*.sv | grep -v warp_scheduler_tb)" \
  PARAMS="-Pgpu.THREADS_PER_BLOCK=8 -Pgpu.WARPS_PER_CORE=2 -Pgpu.ICACHE_LINES=16 -Pgpu.PIPELINED=1" compile
MEM_LATENCY=10 COCOTB_TEST_MODULES=test.test_cross_warp COCOTB_RESULTS_FILE=build/mut/rev/w2l/r.xml \
  vvp -M $(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/mut/rev/w2l/sim.vvp
# -> test_race_flagged_and_excluded PASS, test_bar_separated_rounds FAIL
```

### L1 — `str_conflict` is vacuous at one lane per warp (Low)

`RACED["str_conflict{8,32}"] = {8..11, 24..31}` applies when `THREADS_PER_BLOCK // WARPS_PER_CORE == 1`, which in the
matrix is `warps4gto`. The fixture writes exactly those 12 words (`xf.isasim.run` at W = 1 vs the data image), so the
comparison set is empty. At one lane per warp there is no within-`STR` conflict to check, so nothing in ADR-016 is lost,
but the "every fixture is compared at every warp count" statement in ADR-016 no longer holds for these two. Mutant
`rev` passes `test_isa_diff` (37/37) and `test_multiwarp` (6/6) at `warps4gto`; that is expected (at one lane per warp
lane order is cross-warp order, which SG28 made unspecified), and it is caught at W = 1.

### L2 — `race_words` flags deterministic words (Low)

`xf.isasim.run` marks a word raced when, since the last `BAR`, more than one warp and more than one value have stored to
it. The value set mixes warps, so warp 0 storing 1 then 2 and warp 1 storing 2 is a "race" though only 2 can remain.
`test_l2_deterministic_word_not_raced` (`SAME_FINAL`, TPB 2, W 2) shows `store_races == [(0, pc, 50)]` and
`race_words == {50}` under both policies. Expected per the docs' wording: not raced. Because `memory_diff` requires the
caller to name every raced word, a test author sees it; the cost is a needlessly excluded word. The H1 and fixture
races are real races, so no existing test is affected.

### L3 / item 2 — why the targets left out of `warps2lat` fail (Low)

Each target was run alone on the `warps2lat` build (`build/a21/w2l`, `MEM_LATENCY=10`):

| Target | Result | Root cause |
|---|---|---|
| `test_isa_diff` (8-bit) | 0/37 | Guard: `THREADS_PER_BLOCK differs` (fixtures are recorded at TPB 4). Not a bug |
| `test_xf_isa_diff` (32-bit) | 0/54 | Same guard |
| scratch `a21_isadiff` (the same fixtures, model run at the DUT's TPB 8, W 2, replaying the RTL order) | **37/37 and 54/54 match**, no race flagged | Confirms no RTL/model disagreement behind the guard |
| `test_l1d_indepth`, `test_xf_l1d` | 3/4: `test_write_read_by_later_block` fails, `block 2 thread 0: stale read 0x10` (8-bit) / `0x5a3c1e10` (32-bit) | The test launches `4 * (NUM_CORES + 1)` threads assuming blocks of 4, so "block NUM_CORES starts after an earlier block finished". At TPB 8 that is 12 threads in 2 blocks on 2 cores, running concurrently; the read is a cross-block race (out of scope in `docs/isa.md`). Not a bug |
| `test_multiwarp::test_barrier` | fails `[4, 3, 2, 1, 0, 0, ...] == [4, 3, 2, 1, 4, 3, ...]` after `check_against_model` passed | Expected list `4 - t % 4` assumes TPB 4. At TPB 8, threads 4..7 read SMEM `(3 - t) & 63` = 63..60, never written, so 0 is right; RTL equals the model. Not a bug |
| `test_xf_isa_ext` | 2/3: `test_cmp_unchanged` fails `CMP 0xffffffff, 0x0: got 000 want 100` | `CMP_FLAGS` indexes pairs by `%blockIdx` and launches `4 * n` threads, so at TPB 8 only the first `n / 2` pairs run; pair 18 is never written (000). Not a bug |
| `test_xf_rmsnorm` | **2/2 pass** (1.79 ms sim, 53 s) | No 4-thread assumption: each thread sums its own row from `blockIdx * blockDim + threadIdx`. The `xf.isasim` model at TPB 8, W 2 also matches `golden.rmsnorm_int` on every `test_synthetic` case (control test). SG28's failure does not reproduce at `829b8a3`; the builder's checkout (`/home/claude/sg28`) has identical `src/`, `test/`, `xf/` and Makefile, and its matrix log was overwritten by the final run, so the earlier failure cannot be recovered |
| `test_xf_matmul` | 6/6 pass (193 s) | Could join `XF_warps2lat` |
| `test_xf_attention` | 6/6 pass (407 s) | Runs for minutes at L10, as the comment says; no 4-thread assumption |

So the answer to item 2 is: test assumptions about 4 threads per block (four targets) plus an exact-match guard (two),
and a non-reproducing report (rmsnorm). None is a wrong GPU result. Coverage at `warps2lat` could take
`test_xf_rmsnorm` and `test_xf_matmul` as they are, and the ISA fixtures if `test_isa_diff` ran the model at the DUT's
TPB (the fixture's own `memory` check only applies at W = 1 anyway).

## Soundness of the race exclusion (item 1)

- **False positives (words left out that should be compared).** Two kinds: a deterministic word flagged as raced (L2),
  and taint over-approximation (any ALU result with a tainted operand is tainted, `SUB R, R, R` or `MUL` by 0
  included). Both are bounded by `memory_diff`'s exact-set guard and the 16-word cap: a test fails if the model's
  race set grows or changes, so the set left out is always the one the author wrote down. The larger soundness gap is
  that a left-out word is not checked at all (M1).
- **False negatives (a race not flagged).** None found for `STR`/`STR` within a block. Checked by reading `run` and
  `_taint`: the per-epoch `stored` map is cleared only when every live warp is at `BAR`; a warp's earlier
  `STR` completes before its next issue, so `RET`-then-`BAR` cases are ordered; values derived from a raced word are
  tainted through `LDR`, ALU, `STS`/`LDS` and address registers; an untainted store after a `BAR` clears the taint.
  The gaps all fall outside the stated rule and fail loudly rather than pass: dependence through branches
  (documented), cross-block races (documented as out of scope), and `LDR`/`STR` races between warps without a `BAR`
  (not landed: `a21_ldst` ran 251 of its 300 planned runs, all 150 with warp 0 loading while warp 1 stores and 101 of the reverse, with 0 mismatches; it was stopped at the time box because each `run_kernel` call in one cocotb test starts another clock and the runs slowed; AUDIT-18 also saw none; `docs/isa.md` states no `LDR`/`STR` cross-warp rule). A false negative makes `memory_diff` report a mismatch, so it cannot hide a wrong RTL.
- **`memory_diff` guards.** Exact-set match and the cap both hold (`test_memory_diff_control`). `test_divergence` calls
  it with no raced words, so any race in a divergence kernel fails.
- **`RACED` in `test_isa_diff`.** Correct for both fixtures (matches `race_words` at W 4, TPB 4) and applied only at one
  lane per warp; at two or more lanes per warp the pairs are within one warp and the model flags nothing (L1 for the
  coverage cost).
- **Mutants** (`build/mut/<m>/src`, compiled with `GPU_SRC=`):

| Mutant | `warps2lat` (`test_cross_warp`, `test_divergence`, `test_matadd`) | `warps4gto` (`test_isa_diff`, `test_multiwarp`) |
|---|---|---|
| `rev`: writes claim from the highest consumer down (lowest lane of one `STR` wins) | race test **passes** (M1); rounds test fails; `test_divergence::test_str_conflict` fails | 37/37 and 6/6 pass (expected at one lane per warp, L1); caught at W = 1 |
| `drop`: consumer `j % 8 == 7` (lane 7 of each 8-lane block) writes to the last word instead of its address | both `test_cross_warp` tests fail (word 255: RTL 7, model 0); `test_matadd` fails | not run (caught by every regression kernel) |

## Docs vs code (item 3)

- `docs/isa.md`, ADR-016 and the `xf/isasim.py` docstring agree on the rule (two warps, one word, no `BAR`) and on what
  `store_races` and `race_words` hold, except that `race_words` is broader than "depends on" (L2), and the docstring adds
  "of different values" which `docs/isa.md` omits (harmless: equal values give one outcome).
- ADR-016's memory-ordering paragraph matches `test/helpers/memory.py` (reads before writes in one call, fixed
  per-channel `latency`). AUDIT-18 M1 is closed.
- ADR-016 says `memory_diff` "leaves exactly those words out of the RTL comparison and requires the caller to name
  them, at most 16": true. It does not say the left-out words are unchecked (M1).
- The Makefile comment on `XF_warps2lat` is wrong for `test_xf_rmsnorm` and `test_xf_matmul` (L3). The comment on
  `T_warps2lat` ("l1d_indepth, isa_diff and multiwarp assume 4 threads per block") is accurate (table above).
- PLAN.md marks SG28 DONE with PR #50: consistent with `main`.

## Method

- Read CLAUDE.md, PLAN.md (SG28 row), AUDIT-18, `docs/isa.md`, ADR-016 and `git show 829b8a3`.
- Built `warps2lat` (8- and 32-bit) in `build/a21/w2l` and ran each excluded target alone, plus `test_xf_rmsnorm`,
  `test_xf_matmul` and `test_xf_attention`.
- Scratch cocotb modules in `build/a21/` (not under `test/`): `a21_isadiff.py` (every ISA fixture, model at the DUT's
  TPB, replaying the RTL order) and `a21_ldst.py` (H1's program with warp 0 loading word 100 while warp 1 stores it and
  the reverse; delays 1..5 × 1..5, hog phase 1..6; 251 of 300 runs completed, 0 mismatches).
- Mutants `rev` and `drop` of `src/controller.sv` under `build/mut/`.
- Python checks of `xf.isasim` race flagging (`test_audit21.py`).

## Verification

- `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q`: `852 passed, 55 xfailed`.
- `python3 bench/run.py --check-citations`: `citations: OK`.
