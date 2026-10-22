# AUDIT-12 — Test coverage (mutation)

Lens: which behaviour would no test catch if it broke? "Tests" here means the cocotb modules under `test/`, the
pytest suite under `xf/tests`, `harness_selftest` and the matrix configurations. I covered the modules AUDIT-9 did not
mutate:
- RTL: the data and program memory controllers (claims, accumulators, coalescing, the MeDiC priority pass),
  `l1d_cache` and MeDiC bypass, the I-cache, shared memory/LDS/STS, the warp scheduler inside the core, the
  dispatcher, the contract-12 counters, and ALU/LSU at `WORD_BITS=32`;
- Python: the `xf` golden model and model, the `export.py` checks (contract 8), `bench/run.py` and
  `test/helpers/runner.py`.

I checked against PLAN contracts 5, 8, 9 and 12, `docs/isa.md`, ADR-009 and ADR-012/014. Audited at `44a396c` (main)
with iverilog 12.0, sv2v, cocotb 2.0.1 and torch, with `NANOGPT_DIR=/home/claude/nanogpt`. All builds went under
`build/mut/` and `build/a12/`. `git diff --stat src` is empty.

Method:
- **RTL mutants (50).** Each one is a one-line edit (two lines for L09) to a copy of `src/`, under
  `build/mut/<id>/src`. It is built with `GPU_SRC`/`WS_SRC` pointing at the copy, and each configuration in its own
  `build/mut/<id>/<cfg>`. For each configuration the mutant affects, I ran that configuration's `matrix_run` contents
  one target at a time, in matrix order, stopping at the first failure:
  - the 8-bit suite: `regression`, `test_l1d_indepth`, `test_isa_diff`, `test_controller`, then `multiwarp_run`
    (`divergence_run` for `simt`);
  - the xf suite: `regression_xf` and all 9 `test_xf_*` (minus the long chains at `warps4gto`);
  - `test_medic` where `MEDIC=1`.

  Where a mutant is live at default parameters, I also ran the `test_all` modules that are not in `matrix_run`:
  `test_shared_mem_indepth`, `test_edge_cases`, `test_stress`, `test_launches`, `test_param_sweep`, `test_isa_ext`,
  `divergence_run`, `test_bench_sweep` and `test_audit10`. `test_multiwarp`'s off-matrix builds (`w4_lrr`,
  `w4_gtlrr`) and the standalone `test_warp_scheduler_*` ran where relevant. At most two simulations ran at once.
- **Python mutants (13).** Each is a one-line edit in a copy of the repo's Python side (`build/mut/py/<id>`). Then
  `XF_REQUIRE_NANOGPT=1 NANOGPT_DIR=... python3 -m pytest xf/tests -q -x` ran, plus `python3 -m xf.export --check`
  for export/golden mutants. For model and runner mutants I also ran RTL targets from the copy.
- **Probes** (scratch cocotb module, not committed). These cover:
  - a high-address load at 12 bits;
  - SMEM read before write across blocks;
  - exact I-cache counts on an 8-word loop;
  - two launches through the I-cache;
  - an STS conflict.

  I ran them at HEAD in the default, icache, coalesce, dcache and cores1 configurations, at 8 and 32 bits (all pass).
  Then I ran each against the surviving mutants.
- **Coverage.** `coverage.py` is not installed, so line and branch coverage of the Python packages was skipped.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | HEAD default build: `test_controller`, `test_l1d_indepth`, `test_isa_diff`, `regression`, every `test_xf_*`, `regression_xf` | all pass (xf chain 18 min) |
  | 50 RTL mutants as above (`build/mut/driver.py`) | 41 killed, 9 survived |
  | 13 Python mutants (`build/mut/pydriver.py`) | 6 killed, 7 survived |
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` | 759 passed, 24 xfailed (6 new) |
  | `python3 bench/run.py --check-citations` | `citations: OK` |

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | No test has two lanes load addresses that agree in the low 8 bits in the same cycle at the xf widths. Coalescing on `addr[7:0]` (K08) survives `matrix_coalesce` and `matrix_all` in full, and it returns another address's data | `test_audit12.py::test_coalescing_probe_runs_at_xf_addresses`; K08 below |
| H2 | High | No test checks an I-cache hit or miss count. Hits and misses swapped (I02) and hits never counted (I04) survive `matrix_icache` (I02 also `matrix_all`, xf included). `runner.check_counters` and `test_controller` check only that the total equals the per-core sum | `test_audit12.py::test_icache_counts_are_asserted`; I02/I04 below |
| H3 | High | STS lane order is pinned only in the model. In the RTL, the lowest lane winning an STS conflict (S02) survives every default-config target, all xf tests included. The model's "highest lane wins" (`test_isasim::test_smem_highest_lane_wins`) is never compared with the RTL: no ISA fixture tells the two orders apart | `test_audit12.py::test_isa_fixtures_pin_sts_lane_order`; S02 below |
| M1 | Medium | GTLRR inside the core is not gated. W03 (core feeds `warp_progressing = warp_ready`) and W04 (GTLRR's first pass ignores progress, i.e. LRR) are killed only by `make test_multiwarp` (`multiwarp_w4_gtlrr`), which neither `test_all` nor `matrix` calls. `test_warp_scheduler_gtlrr` asserts only the issue count. `bench/phase-11-all.json` has `w2gtlrr`/`w4gtlrr` rows | `test_audit12.py::test_gtlrr_core_is_gated` |
| M2 | Medium | Contract 8's segment bounds are not pinned. Setting `GAIN_TOL, RMS_TOL = 0.25, 11` (P01, 100× and 10× looser) passes pytest with nanoGPT and `export --check`, because `test_audit7` reads the bounds from `xf.export` | `test_audit12.py::test_contract8_segment_bounds_pinned` |
| M3 | Medium | MeDiC's threshold is unpinned, and `test_medic` infers installs from the priority signal. L07 (`warp_low_utility = h <= 2`) bypasses fills that `warp_high_utility` still calls high-utility. The monitor counts those fills as installs, the hot lanes still beat `MEDIC=0`, and the test passes. The bench numbers move: hot hits 18 → 16, misses 38 → 40, 781 → 784 cycles | `test_audit12.py::test_medic_observes_the_bypass_decision` |
| L1 | Low | `run_launches`' `check_block` refusal (contract 5, SG22) is untested. With the loop deleted (P13), every test passes. `make test_xf_matmul PARAMS=-Pgpu.THREADS_PER_BLOCK=32` then runs `matmul_tiled` and reports mismatches instead of refusing | P13 below |
| L2 | Low | `bench/run.py`'s writers (`run_xf`, `write_xf_doc`, `run_phase11`) are untested; only `check_citations`/`resolve`/`expand` are. P12 swaps the L1-D hit/miss columns of `docs/xf-bench.md` and passes everything | P12 below |
| L3 | Low | The model's GTO greedy step is unpinned (contract 9). P07 (`gto` always picks the lowest ready warp) passes pytest, `multiwarp_w4_gto`, `test_isa_diff` at W=4 GTO and `divergence_w4c1gto` | P07 below |
| L4 | Low | The MLP half of `export --check`'s outlier-channel assertion has no other test. Deleting the assertion (P03) passes pytest and `export --check`. The attention half is covered by `test_audit6` | P03 below |

The three High rows follow the brief's rule: each mutant survives a configuration the matrix runs and changes
observable data or counters. H3 has the least practical impact of the three: no kernel stores to one SMEM word from
several lanes, and `docs/isa.md` does not specify the order. Only `xf/isasim.py`'s docstring and its unit test do.

Closed since earlier audits (killed now): the AUDIT-7 M1 write-hit tag mutant (L02, by `test_xf_isa_diff` at dcache
and medic). The read-hit tag mutant (L03, by `test_isa_diff`). AUDIT-3 L1's DIV-by-zero (A03, by `test_isa_diff`).
The SG18 claim and counter reverts (K06, K07). The SG19 priority mutant (K04).

## Mutant kill summary

**RTL: 41 killed, 9 survived.** Of the survivors, 2 are equivalent (K03, A06), and 7 are H1 (K08), H2 (I02, I04),
H3 (S02), M1 (W03, W04) and M3 (L07).

| Group | Killed (by) | Survived |
|---|---|---|
| Controller K01–K10 | K01, K07 (`regression`); K06, K10 (`test_l1d_indepth`, default and latency); K02 (`test_xf_rmsnorm` counter check at `all`; survives `coalesce`); K04, K05, K09 (`test_medic`) | K03 (equivalent), K08 |
| L1-D/MeDiC L01–L11 | L01, L05, L11 (`test_l1d_indepth`); L03 (`test_isa_diff`); L02, L04 (`test_xf_isa_diff`); L08 (`regression`); L05, L06, L09, L10 (`test_medic`) | L07 |
| I-cache I01–I05 | I01 (`regression`); I03 (`test_l1d_indepth`); I05 (`test_xf_matmul`) | I02, I04 |
| SMEM S01–S04 | S01 (`multiwarp_run`); S03 (`regression_xf`); S04 (`test_xf_isa_diff`) | S02 |
| ALU/LSU A01–A06, U01 | A01 (`test_isa_ext`); A02, A03 (`test_isa_diff`); A04, A05 (`test_xf_isa_diff`); U01 (`test_xf_elementwise`) | A06 (equivalent) |
| Dispatcher D01–D03 | D01 (`test_isa_diff`); D02 (`test_controller`, cores4); D03 (`regression`) | — |
| Counters X01–X04 | X01 (`test_isa_diff`, cores4); X02 (`test_l1d_indepth` via `check_counters`, cores4); X03 (`multiwarp_run`, warps4gto; survives warps2); X04 (`test_isa_diff`, warps2) | — |
| Warp scheduler W01–W06 | W01 (`multiwarp_run`, warps2); W02, W05 (`multiwarp_run`, warps4gto); W06 (`multiwarp_run`, warps4gto; survives warps2) | W03, W04 (killed only by off-gate `multiwarp_w4_gtlrr`) |

Equivalent survivors:
- **K03** (READ_RELAYING waits for the owner's valid only). Every rider drops its valid in the same cycle as the
  owner, because LSUs drop on `mem_read_ready` and the L1-D drops on its fill.
- **A06** (MAX `>=`). Equal operands give the same result.

**Python: 6 killed, 7 survived.**
- Killed:
  - P04 (`rsqrt_index` `>`): `test_audit1`.
  - P05 (no rounding at `s = 1`), P06 (unmasked softmax rowmax), P11 (rmsnorm `k` capped): `test_golden` / `test_export`.
  - P08 (model STS lowest wins): `test_isasim::test_smem_highest_lane_wins`.
  - P09 (model STR order): `test_isasim::test_store_conflict`.
- Survived:
  - Findings: P01 (M2), P03 (L4), P07 (L3), P12 (L2), P13 (L1).
  - Not findings (the property is still checked elsewhere): P02 (`HELDOUT_TOL = ()`, guarded by `test_audit2`) and
    P10 (the extreme-input sweep in `export.check` deleted, guarded by `check_bounds` and
    `test_export::test_no_overflow_on_any_int8_input`).

## Reproducing a mutant

Every RTL repro below has the same shape. HEAD's `matrix` is green, so the command passes there without `GPU_SRC`,
and it still passes with the mutant:

```
mkdir -p build/mut/<id> && cp -r src build/mut/<id>/src     # then apply the edit to build/mut/<id>/src/<file>
G="$(ls build/mut/<id>/src/*.sv | grep -v warp_scheduler_tb)"
make BUILD=build/mut/<id>/<cfg> PARAMS="<CFG_cfg>" GPU_SRC="$G" matrix_run     # = matrix_<cfg> on the mutant
```

## H1 — coalescing never meets 12-bit addresses (High)

K08, `controller.sv`, coalescing pass:

```
-  && consumer_read_address[k] == consumer_read_address[j]) begin
+  && consumer_read_address[k][7:0] == consumer_read_address[j][7:0]) begin
```

Under K08, `make … PARAMS=-Pgpu.COALESCE=1 GPU_SRC="$G" matrix_run` passes in full, and so does `matrix_run` with
`CFG_all` (`test_medic` included). The 8-bit suite cannot see it, because the addresses are 8 bits. The only directed
coalescing test (`test_controller`) runs only in that suite, with addresses 0–15. No xf kernel has two lanes load `a`
and `a + 256k` in the same cycle.

A probe kernel at the xf widths, where thread `t` loads `1024·t + 5` twice and stores both loads, shows the effect.
At HEAD it stores `[10, 10, 11, 11, 12, 12, 13, 13]` (the model's result). Under K08 it stores 10 everywhere, with
6 of 10 loads "coalesced":

```
    CONST R1, #32
    MUL R1, R1, R1
    MUL R0, %threadIdx, R1     ; 1024 t
    CONST R2, #5
    ADD R0, R0, R2
    LDR R3, R0
    LDR R5, R0
    CONST R4, #64
    ADD R4, R4, %threadIdx
    ADD R4, R4, %threadIdx
    STR R4, R3
    CONST R6, #1
    ADD R4, R4, R6
    STR R4, R5
    RET
```

Fix: run `test_controller` in the xf suite too, with a probe whose lanes load addresses that are equal mod 256
(for example the program above).

## H2 — I-cache counts unchecked (High)

I02, `fetcher.sv`: `if (fetch_was_hit)` → `if (!fetch_was_hit)` (hits and misses swapped). I04, `fetcher.sv`, I-cache
hit branch: `fetch_initiated <= 1'b1` → `1'b0` (hits never counted). Under each, `matrix_run` with
`PARAMS=-Pgpu.ICACHE_LINES=16` passes, and so do the `test_all` modules at that build. I02 also passes `matrix_run`
with `CFG_all`.

Contract 12 asks only that `icache_*_total` be the sum over cores, and both mutants keep that. The bench (phase 3,
`warps_icache` in phase 11) reports these numbers. A 1-block, 8-word loop (14 fetches, 8 distinct PCs) gives 6 hits /
8 misses at HEAD, 8 / 6 under I02 and 0 / 8 under I04.

Fix: assert exact hit and miss counts for a small loop, for example `misses == distinct PCs` and
`hits + misses == warp_issues_total` at one warp.

## H3 — STS lane order not compared with the RTL (High)

S02, `shared_memory.sv`: iterate the write ports from the highest lane down, so the lowest lane wins. It passes the
whole default-config set: the 8-bit matrix suite, the `test_all`-only modules and every xf test.
`test_audit12::test_isa_fixtures_pin_sts_lane_order` reruns the 81 ISA fixtures on a model with that edit (W = 1, 2
and 4), and no memory changes. A 4-lane `STS R0, %threadIdx` to SMEM[0] then `LDS`/`STR` stores 3 at HEAD and 0
under S02.

Fix: add an STS-conflict fixture to `xf.isa_fixtures` (and state the rule in `docs/isa.md`).

## M1 — GTLRR is outside the gate (Medium)

W04, `warp_scheduler.sv`, GTLRR first pass: `if (!found && warp_ready[j] && warp_progressing[j])` →
`if (!found && warp_ready[j])`. W03, `scheduler.sv`: `.warp_progressing(warp_ready & last_issued)` →
`.warp_progressing(warp_ready)`.

Both pass `make test_warp_scheduler_gtlrr`, `warps2` and `warps4gto`, and both fail
`make multiwarp_w4_gtlrr GPU_SRC="$G"` (`test_policy_order`, `test_policy_ages`, `test_age_saturation`). That target belongs to `make test_multiwarp`. `test_all`
runs the `test_multiwarp` module only on the default (W=1) build, and no `CFG_*` sets `WARP_POLICY=2`.

Fix: add `multiwarp_w4_gtlrr` to `test_all`, or give `matrix` a GTLRR configuration. Also assert an issue-order
property in `test_warp_scheduler_gtlrr`.

## M2 — contract 8 bounds can be loosened silently (Medium)

P01, `xf/export.py`: `GAIN_TOL, RMS_TOL = 0.0025, 1.1` → `0.25, 11`. `NANOGPT_DIR=… pytest xf/tests` (756 passed) and
`python3 -m xf.export --check` both pass. `TOL_LSB`/`ATTN_TOL_LSB` are pinned through the committed fixtures' `tol`.
`GAIN_TOL`, `RMS_TOL` and the `SEGMENTS` table are not pinned anywhere outside `export.py`.

Fix: assert the contract-8 values in a test.

## M3 — MeDiC threshold and the monitor (Medium)

L07, `l1d_cache.sv`: `assign warp_low_utility[lu] = (h < MEDIC_THRESHOLD)` → `(h <= MEDIC_THRESHOLD)`. It passes
`make test_medic GPU_SRC="$G"` and the medic 8-bit suite.

The monitor in `test_medic` decides install or bypass from `warp_high_util`, which does not change. Its
"low-utility lane installed" check therefore never fires, and the `WARMUP` count still reads 2 on streaming lanes.
The test log under L07 gives `hot_hits 16`, `dcache_misses 40`, `cycles 784`, `install [2, 2, 1, 1, 2, 2, 3, 3]`. At
HEAD (`bench/phase-11-all.json#medic/medic1/L0`) the same run gives 18, 38, 781 and `install [2, 2, 2, 2, 2, 2, 3, 3]`.

Fix: classify from the cache itself (`valid_arr`/`tag_arr` after the fill, or `warp_low_util`), and pin the hot lanes'
MEDIC=1 hit count.

## L1–L4

- **L1, P13.** In `test/helpers/runner.py`, delete `for ln in launches: check_block(...)`. pytest passes and
  `make test_launches` passes. Then `make test_xf_matmul PARAMS=-Pgpu.THREADS_PER_BLOCK=32` fails with
  `AssertionError: mismatch: ['s4x8x16.tiled', …]` where HEAD refuses with `check_block`'s `ValueError`. No test calls
  `run_launches` with an unsupported block size. A fake-DUT pytest can check this (the check runs before the clock
  starts).
- **L2, P12.** In `bench/run.py::write_xf_doc`, swap `r['dcache_hits']`/`r['dcache_misses']`. pytest passes. SG23's
  "two runs equal" check would not catch a consistently wrong writer either.
- **L3, P07.** In `xf/isasim.py`, change the `gto` order `[cur, *range(W)]` to `[*range(W)]`. It passes pytest, and
  from the mutated copy it passes `make multiwarp_w4_gto`, `make test_isa_diff PARAMS="-Pgpu.WARPS_PER_CORE=4
  -Pgpu.WARP_POLICY=1"` and `make divergence_w4c1gto`. Contract 9 says the policy stays on a warp until BAR or RET;
  nothing checks the "stay" half.
- **L4, P03.** In `xf/export.py::check`, `assert max(out_peak…) <= 127.5 and out_err.max() <= tol_lsb` →
  `assert True or …`. pytest and `export --check` pass. `test_audit6::test_attention_tol_outlier_channel` covers
  attention only.

Observations (no row):
- `test_shared_mem_indepth` runs only in `test_all`, never in the matrix.
- The coalescing counter checks in `check_counters` are the only thing that killed K02. They held.
- `coverage.py` is absent, so no line or branch figures are given.
