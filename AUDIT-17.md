# AUDIT-17 — Test coverage (mutation), round 5

Lens: which behaviour would no test catch if it broke? "Tests" means the cocotb modules under `test/`, the pytest
suite under `xf/tests`, `harness_selftest` and the matrix configurations. This round:
1. confirms that SG25 and SG26 closed AUDIT-13 SM1/SM2/B02 and AUDIT-14's store-conflict cases;
2. mutates SG25's replay harness (`runner` trace, `xf.isasim.run_trace`), the memory model's write order, and SG27's
   `PARAMS` checks;
3. seeds mutants in code no earlier round mutated:
   - `pc.sv`, `dcr.sv` and `gpu.sv`, which AUDIT-9/12/13 never touched;
   - `registers.sv` reset and the `core.sv` lane enables, beyond D10/D11/B01/B04/C*;
   - the no-I-cache fetcher handoff and the LSU SMEM write data;
   - `test/helpers` (memory model, runner trace, `check_counters`);
   - `xf/export.py` `check()`;
   - `bench/run.py`'s `iv`, `run_one`, `_sweep`, `_medic` and `P11`.

I audited at `e9c9eef` (main) with iverilog 12.0, sv2v, cocotb 2.0.1, torch and `NANOGPT_DIR=/home/claude/nanogpt`.
All builds went under `build/mut/`. `git diff --stat src test xf/*.py Makefile docs` is empty. At most two jobs ran
at once.

Method:
- **RTL mutants** (`build/mut/mutants17.py`, driver `build/mut/driver.py`, copied from AUDIT-13). Each mutant is a
  one-line edit to a copy of `src/` under `build/mut/<id>/src`, built with `GPU_SRC` pointing at the copy. The
  driver runs the plan's targets one at a time and stops at the first failure. `FULL` is:
  - the 8-bit suite: `test_isa_diff`, `regression`, `test_l1d_indepth`, `test_controller`, `multiwarp_run`;
  - every `test_xf_*` and `regression_xf`;
  - the `test_all`-only modules, `test_multiwarp` and `divergence_run`.
- **Python, Makefile and helper mutants** (`build/mut/pydriver17.py`). Each is an edit in a copy of the repo
  (`build/mut/py/<id>`). Then these run in order:
  - `XF_REQUIRE_NANOGPT=1 pytest xf/tests -x`, `xf.export --check`, `xf.isa_fixtures --check` and
    `bench/run.py --check-citations`, as relevant;
  - for helper mutants, RTL targets run from the copy.

  Higher-order mutants H1–H4 pair a weakened `check_counters` assertion with the RTL counter mutant that assertion
  killed in AUDIT-12/13.
- **Probes** (`build/mut/probe/probe17.py`, scratch). These are STR lane conflicts confined to each block, compared
  with `xf.isasim` (`run_trace` at W > 1).

Commands actually run:

| Command | Result |
|---|---|
| `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` (HEAD, before) | 825 passed, 29 xfailed |
| 4 closure RTL mutants (SM1, SM2, B02, ST1) | 4 killed |
| 16 fresh RTL mutants (N01–N16) | 9 killed, 7 survived (2 equivalent) |
| 20 harness/helper/Makefile mutants (RT, MW, CK, PC) | 15 killed, 5 survived (RT5 equivalent) |
| 4 higher-order (H1–H4) | 4 killed |
| 16 export/bench mutants (E01–E11, BW1–BW5) | 3 killed, 13 survived (none new, below) |
| STR-conflict probe at HEAD: TPB 4/8/16, TPB 8 with medic, coalesce+dcache at latency 10, W=2 at latency 3 | all match the model |
| `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` (with `test_audit17.py`) | 828 passed, 38 xfailed (3 new controls, 9 new strict xfails) |
| `python3 bench/run.py --check-citations` | `citations: OK` |

## Findings

No Critical, High or Medium findings: no surviving mutant changes observable results in a matrix-covered
configuration for a program whose behaviour docs/isa.md defines.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| L1 | Low | The per-block reset values of GPRs and NZP are unpinned. N11 leaves R12 uncleared (it keeps the previous block's value), and N02 resets NZP to `z`. Both survive `FULL` at default. A model that starts every GPR at 0x5A, or NZP at `z` or `n`, changes no ISA fixture: no fixture reads a register before writing it or branches before its first CMP. `xf.isasim` zeroes both, but docs/isa.md says nothing about either (it states the SMEM reset only) | `test_audit17.py::test_isa_fixtures_read_an_unwritten_register`, `::test_isa_fixtures_branch_before_cmp` |
| L2 | Low | `gpu.sv`'s three `$fatal` parameter guards are untested. Deleting the `WORD_BITS != DATA_MEM_DATA_BITS` (N09), `WARP_POLICY` (N10) or `WARPS_PER_CORE` (N16) guard survives every target run. No test builds an invalid configuration. A mismatched `-Pgpu.WORD_BITS=32` without `DATA_MEM_DATA_BITS` would then build silently | `test_audit17.py::test_gpu_parameter_guard_is_tested` |
| L3 | Low | Four of `runner.check_counters`' five assertions have no test of their own. If any of them is deleted (CK1–CK4), pytest still passes; only the coalesced assertion has one (`test_audit12` control). The assertions are not load-bearing today: H1–H4 show that each RTL counter mutant they killed in AUDIT-12/13 (K07, X02, CC2, CC1) is still killed with the assertion gone, by `test_l1d_indepth`, `test_isa_diff` or `test_controller` | `test_audit17.py::test_check_counters_assertion_is_tested` (control `::test_control_coalesced_check_is_tested`) |

## Closure (confirmed)

| Item | Mutant | Killed by |
|---|---|---|
| AUDIT-13 H1 | SM1 (STS index unmasked), SM2 (LDS index unmasked) | `test_isa_diff`, default (`smem_wrap8`: 6 words differ; SM2 reads X) |
| AUDIT-13 H2 | B02 (8-bit `total_blocks` rounding) | `test_isa_diff`, default (`launch8_253`: 249 words differ) |
| AUDIT-14 H1 / SG25 | ST1: MeDiC priority pass re-enabled for writes (`(!PRIORITY_ENABLE \|\| pass == 1)` → `can_consider`) | `test_medic` at medic and all (`mem[200] 3 != 4`). It survives `test_isa_diff`/`test_xf_isa_diff` there: the `str_conflict` fixtures run on a cold L1-D, so no lane has priority |
| SG25 memory model | MW1 (writes applied channel N-1 down, the pre-SG25 order) | `test_isa_diff`, default (`str_conflict8`) |
| | MW2/MW3 (stale-valid guard dropped for writes/reads) | `test_isa_diff`, via `check_counters` request count |
| SG25 replay | RT1 (`run_trace` returns `ready[0]`, ignoring the trace) | pytest (`test_audit14`); alone through the RTL (RT1r): `test_isa_diff` W=2 (`issue 20 picks warp 0, ready are [1]`) |
| | RT2 (leftover trace accepted), RT3 (pick not checked against ready) | pytest (`test_run_trace_rejects_illegal_orders`) |
| | RT4 (runner records a warp every cycle) | `test_isa_diff` W=2 |
| | RT5 (runner samples the trace before `ReadOnly`) | survives W=2 and 4 and `divergence_run`; equivalent: `issue`, `selected_warp` and `block_id` are all read one edge later, consistently |
| SG27 PARAMS | PC1 (`unscoped` empty), PC2 (`compile_xf` without `check_params`), PC3 (first word only), PC4 (MEDIC=0 guard deleted), PC5 (PARAMS prepended in `test_medic`), PC6 (`-P%.%` accepted), PC7 (`divergence_%` drops PARAMS) | pytest (`test_audit16`) |

AUDIT-14's store-conflict cases also agree at HEAD outside the matrix. A block-local STR-conflict probe (all lanes
→ word `blockIdx`; lane pairs → `32 + 8·blockIdx + t/2`) matches the model in all of these:
- `THREADS_PER_BLOCK` 4, 8 and 16;
- TPB 8 with `MEDIC=1`;
- TPB 8 with `COALESCE=1 DCACHE_LINES=16` at latency 10;
- TPB 8 at W=2 and latency 3.

The `str_conflict8` fixture itself is valid only at TPB 4: at TPB 8 its pair words overlap across blocks, which is a
cross-block race the model scopes out. `test_isa_diff` refuses a TPB mismatch, so this is not a finding.

## Mutant kill summary

**RTL: 13 killed, 7 survived.** Of the survivors, 2 are equivalent (N12, N13). The other 5 are L1 (N02, N11) and L2
(N09, N10, N16).

| ID | File | Edit | Result |
|---|---|---|---|
| SM1 / SM2 / B02 | shared_memory / dispatch | AUDIT-13 verbatim | killed, `test_isa_diff` |
| ST1 | controller.sv | write claims join the priority pass | killed, `test_medic` (medic, all) |
| N01 | pc.sv | taken iff `(nzp & cond) == cond` | killed, `test_bench_sweep` (`branch_relu`'s `BRnzp`), a `test_all`-only module. No ISA fixture's result depends on a multi-bit condition (only `relu_branch` uses one, `BRzp`, and its outcome does not matter) |
| N02 | pc.sv | NZP resets to `010` | **survived** `FULL` (L1) |
| N03 | dcr.sv | 7-bit thread count | killed, `test_isa_diff` (`launch8_253`) |
| N04 | gpu.sv | `warp_issues_total` sums core 0 × N | killed, `test_isa_diff` |
| N05 | gpu.sv | `bar_stall_cycles_total` sums core 0 × N | killed, `multiwarp_run` (warps2, warps4gto) |
| N06 | gpu.sv | `icache_hits_total` sums core 0 × N | killed, `test_isa_diff` (icache, `check_counters`) |
| N07 | gpu.sv | data controller priority from `warp_low_util` | killed, `test_medic` |
| N08 | gpu.sv | core `telemetry_reset` = per-block reset | killed, `test_isa_diff` (default, icache) |
| N09 / N10 / N16 | gpu.sv | `WORD_BITS`, `WARP_POLICY`, `WARPS_PER_CORE` guard → `if (0)` | **survived** (L2) |
| N11 | registers.sv | R12 not cleared on reset | **survived** `FULL` (L1) |
| N12 / N13 | core.sv | `active = i <= thread_count` (W=1 / W>1) | survived default `FULL` / warps2, warps4gto; equivalent: the phantom lane's ALU and LSU stay disabled (`i < thread_count`), the warp PC comes from lane 0, and `exists` masks it under the stack |
| N14 | fetcher.sv | no-I-cache `FETCHED` leaves only on DECODE | killed, `test_isa_diff` at pipelined (timeouts) |
| N15 | lsu.sv | STS data `rt[7:0]` | killed, `test_xf_isa_diff` |

**Python, Makefile and helpers: 22 killed, 18 survived.**
- Killed: RT1–RT4, MW1–MW3, PC1–PC7 and H1–H4 (above), E05 (scores causal mask dropped from `segment_stats`, by
  pytest), E11 (vanishing multiplier allowed, by pytest) and BW1 (`iv` emits unscoped `-P`, by
  `test_audit16::test_bench_medic_params_are_appended`).
- Survived: RT5 (equivalent) and CK1–CK4 (L3). The other 13 are not findings, because the property is still checked
  elsewhere or the gap is already recorded:
  - E01 (determinism assertion deleted) and E02 (stale-fixture assertion deleted): `test_export::test_export_matches_fixtures`.
  - E03 (saturation bound 1e9): `test_export::test_heldout_activations_unsaturated`.
  - E04 (`worst <= TOL` deleted): the stricter held-out check and `test_audit2`.
  - E06 (row sum ≤ 2): `test_attention` asserts `row_sum_error ≤ 1`.
  - E08 (RMS bound `r0 + RMS_TOL`) and E09 (one-sided gain): `test_audit7` applies its own predicate to the same stats.
  - E10 (norm rounding term without `kmax`): `test_export::test_no_overflow_on_any_int8_input` and golden's int32
    checks.
  - E07 (`heldout` ignores `gain`, so the outlier check repeats the Gaussian set): the MLP half is AUDIT-12 L4,
    still open. The attention half is covered by `test_audit6::test_attention_tol_outlier_channel`.
  - BW2 (phase-11 rows' cycles zeroed), BW3 (`_medic` drops `MEM_LATENCY`), BW4 (`run_one` reports pass) and BW5
    (warp builds at TPB 4): the bench writers are untested, which is AUDIT-12 L2 (still open).

## Reproducing a mutant

RTL (passes at HEAD without `GPU_SRC`; for the survivors it also passes with it):

```
mkdir -p build/mut/<id> && cp -r src build/mut/<id>/src     # then apply the edit to build/mut/<id>/src/<file>
G="$(ls build/mut/<id>/src/*.sv | grep -v warp_scheduler_tb | tr '\n' ' ')"
make BUILD=build/mut/<id>/default GPU_SRC="$G" matrix_run    # = matrix_default on the mutant
```

Python, Makefile or helper mutants: copy the repo to `build/mut/py/<id>`, apply the edit there, and run
`NANOGPT_DIR=/home/claude/nanogpt XF_REQUIRE_NANOGPT=1 python3 -m pytest xf/tests -q -x` from the copy.

## L1 — GPR and NZP reset values (Low)

N11, `registers.sv`:

```
-            for (i = 0; i < 13; i = i + 1) registers[i] <= 0;
+            for (i = 0; i < 12; i = i + 1) registers[i] <= 0;
```

N02, `pc.sv`:

```
-            nzp     <= 3'b0;
+            nzp     <= 3'b010;
```

Command: `make BUILD=build/mut/N11/default GPU_SRC="$G" matrix_run` passes at HEAD and under each mutant. Every
`test_all`-only module, `test_multiwarp` and `divergence_run` pass too.

Under N11, R12 holds the previous block's value on the same core. A program that reads R12 before writing it gives
different memory at HEAD and under N11. The same holds for a program that executes `BRz` before any `CMP` under N02.
The model zeroes both (`xf/isasim.py` lines 71–72), but docs/isa.md does not say so.

Fix: either state the reset values in docs/isa.md and add an ISA fixture that reads R0–R12 and branches before its
first CMP, or declare them undefined.

## L2 — parameter guards (Low)

N09, `gpu.sv`: `initial if (WORD_BITS != DATA_MEM_DATA_BITS) $fatal` → `initial if (0) $fatal`. N10 and N16 make
the same edit to the `WARP_POLICY` and `WARPS_PER_CORE` guards. The command
`make BUILD=build/mut/N09/default GPU_SRC="$G" test_isa_diff test_param_sweep test_edge_cases test_launches test_xf_isa_diff`
passes at HEAD and under each mutant.

Matrix configurations are valid, so no matrix result changes. Without the guard, however,
`-Pgpu.WORD_BITS=32` alone builds a GPU with 32-bit registers and an 8-bit data bus, and nothing reports it.

Fix: a pytest case that builds each invalid configuration (`make -n` is not enough; compile and run `vvp`) and
expects the `$fatal` text.

## L3 — `check_counters` assertions (Low)

In `test/helpers/runner.py`:
- CK1 turns `assert metrics["data_mem_requests"] == reads + writes` into `assert True or …`;
- CK2 does the same to the per-core I-cache sum;
- CK3 makes `WARPS_PER_CORE == 1` `== 0`;
- CK4 deletes the `ICACHE_LINES or not hits` assertion.

`XF_REQUIRE_NANOGPT=1 pytest xf/tests -q -x` passes at HEAD and from each mutated copy.

These assertions are not the only guard. In the higher-order runs:

| Run | Killed by |
|---|---|
| H1 = CK1 + K07 (write requests uncounted) | `test_l1d_indepth` |
| H2 = CK2 + X02 (cores4) | `test_l1d_indepth` |
| H3 = CK3 + CC2 | `test_isa_diff` (steps) |
| H4 = CK4 + CC1 | `test_controller` (exact I-cache counts, SG24) |

So the risk is only that a future change silently weakens them.

Fix: unit cases like `test_audit12::test_control_check_counters_rejects_a_wrong_coalesced_count` for the other four.

## Observations (no row)

- ST1 shows that the `str_conflict` fixtures do not exercise MeDiC's priority path. Only `test_medic`'s SG25
  assertion does, and it is in the matrix only at medic and all.
- N01 shows that multi-bit `BR` conditions (`BRnzp`, `BRnz`, …) appear in no ISA fixture whose result depends on
  them. Only `test_bench_sweep` (`test_all`, default config) kills it. Because `pc.sv` is configuration-independent,
  this is not a matrix gap today.
