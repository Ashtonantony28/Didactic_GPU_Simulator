# AUDIT-20 — API/SDK consistency (round 5)

Lens: do the public surfaces agree with each other and with their docs? This round repeats AUDIT-16's scope and
adds everything merged since `1f9021a`:
- **SG25:** ADR-016, the STR rule in `docs/isa.md`, the RET row, ADR-009's reads-only priority, ADR-014's harness
  paragraph, `runner.run_kernel(trace=...)`, `xf.isasim.run_trace` and the callable `warp_policy`, and the text of
  contracts 9 and 12;
- **SG26:** the `smem_wrap{8,32}` and `launch{8,32}_{253,255}` fixtures and the relu 253/255 cases, checked against
  the binder limits and the docs;
- **SG27:** the `PARAMS` scope check and `test_medic`'s append semantics, checked against every documented command.

I also re-checked the AUDIT-16 lows and the AUDIT-11 lows.

Audited at `9770a90` (main) with iverilog 12.0, sv2v and cocotb 2.0.1, with `NANOGPT_DIR` set for pytest. Builds went
under `build/a20/`. A fuzz campaign was running alongside, so I ran only single targets at `-j1`.

Method:
- **Commands.** I extracted every `make …` and `python3 bench/run.py …` command from README, CLAUDE.md, ROADMAP,
  `docs/` (ADRs and phase reports included) and the `bench/run.py` docstring. I resolved each one with `make -n`
  against the Makefile and `bench/run.py`. All of them resolve. `make -n matrix_medic` stops only because `-n` skips
  its `mkdir`.
- **SG27.** I passed `make compile` and `bench/run.py`'s `iv()` the forms that a user can type and that start with
  `-Pgpu.`:
  - unparsable values, such as `=GTO`, `=16k`, an empty value or no `=`;
  - hierarchical names;
  - a doubled prefix.

  For each form, I compared what iverilog elaborates on a two-module design and on `gpu.v`.
- **SG25.** I compared `isasim.run`'s callable path and `run_trace` with the module docstring, ADR-014/016 and
  contract 9, and probed them with policies that return a held, finished or out-of-range warp. I compared the
  `docs/isa.md` STR paragraph with the `STR` branch, and ADR-009 with `controller.sv` and `test_medic.py`'s monitor.
  I checked that `scheduler.sv` drives `issue`/`selected_warp` at `WARPS_PER_CORE = 1` too, so `trace=` works at
  every warp count.
- **SG26.** I checked the fixture comments against their programs (the `SMEM_WRAP` address arithmetic, `459 & 255 =
  203`, wrapping to word `11 − t`). I checked the launch fixtures and the relu cases against `_bind`'s 255-thread
  limit.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` | 839 passed, 49 xfailed (with `test_audit20.py`) |
  | `python3 bench/run.py --check-citations` | `citations: OK` |
  | `python3 -m xf.isa_fixtures --check` | `ok: 91 ISA fixtures current, skipped []` |
  | `make test_medic BUILD=build/a20/medic PARAMS="-Pgpu.DCACHE_LINES=16"` | exit 0; the runs log `MEDIC=0 … hot_hits 14`, `MEDIC=1 … hot_hits 18` (AUDIT-16 M1 fixed) |
  | `make test_matadd BUILD=build/a20/dbl PARAMS="-Pgpu.-Pgpu.ICACHE_LINES=16"` | exit 0, 1/1 PASS, every `ICACHE_LINES` record in the `.vvp` is 0 (M2) |
  | `make compile BUILD=build/a20/m PARAMS=…` for `-Pgpu.ICACHE_LINES`, `-Pgpu.cores[0].core_instance.SMEM_SIZE=32` | exit 0 each; the first prints `error: invalid value …`, the second prints nothing (M1) |

There are **no critical or high findings**, so this audit adds no SG rows. The two mediums are what SG27 left of the
"PARAMS that silently build the wrong GPU" class.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| M1 | Medium | SG27's check (`filter-out -Pgpu.%`) passes any switch that starts with `-Pgpu.`. iverilog still ignores two kinds of these and exits 0: a value it cannot parse (`-Pgpu.WARP_POLICY=GTO`, `=16k`, an empty value, no `=`), where it prints an error, and a hierarchical name (`-Pgpu.cores[0].core_instance.SMEM_SIZE=32`, `-Pgpu.gpu.X=…`), where it prints nothing. In both cases `make compile` exits 0, the default GPU is built and the test passes | `test_audit20.py::test_compile_rejects_params_iverilog_ignores` (5 cases); controls `::test_control_iverilog_ignores_these_forms_with_exit_0`, `::test_control_valid_scoped_param_applies` |
| M2 | Medium | `bench/run.py`'s `iv()` puts `-Pgpu.` in front of every `--params` word without checking it. Passing the Makefile's form, `--params "-Pgpu.ICACHE_LINES=16"`, gives `-Pgpu.-Pgpu.ICACHE_LINES=16`. That passes the scope check, and iverilog drops it without a message. Every test passes on the default GPU, and the rollup records `"params": "-Pgpu.ICACHE_LINES=16"`. `run_one` hides make's output when a test passes, so M1's iverilog errors are never shown either | `test_audit20.py::test_bench_params_in_make_form_are_not_dropped`; control `::test_control_bench_iv_doubles_the_prefix` |
| L1 | Low | SG25's trace API is not in the contracts, and `run` does not check a callable policy's choices. Contract 9 still says "`warp_policy` is `"lrr"` or `"gto"`", and no contract names `run_trace` or `run_kernel(trace=...)`; only `_run`'s docstring documents `trace`. The module docstring says a callable "picks the next warp from the ready … warp indices", but `run` does not enforce it. Picking a warp held at `BAR` runs it past the barrier without an error (memory `[1,1,3,3]` where LRR gives `[3,3,3,3]`). Picking a finished or out-of-range warp raises `IndexError`, although contract 9 says errors raise `IsaSimError` | `test_audit20.py::test_contract9_names_the_trace_api`, `::test_callable_policy_pick_is_validated[held,done]`; control `::test_control_run_trace_replays_a_recorded_order` |
| L2 | Low | `docs/isa.md` says that when several lanes of one `STR` write the same word, "`xf.isasim` lists these stores in `stores_conflict`". The model lists only stores with different values: four lanes that store 7 to word 0 give `[]` | `test_audit20.py::test_same_value_str_conflict_listed_or_documented`; control `::test_control_str_conflict_highest_lane_and_listed` |
| L3 | Low | ADR-009's Tests paragraph still says `test_medic`'s monitor "fails the run if … a low-utility lane gets a channel while a high-utility lane waits". Since SG25, priority and the monitor cover reads only (ADR-009 §Priority, ADR-016, `test_medic.py` line 105). A waiting high-utility write may let a low-utility read through | `test_audit20.py::test_adr009_test_paragraph_says_reads` |

## M1 — the scope check passes `-Pgpu.` switches iverilog ignores

The Makefile comment says: "iverilog drops an unscoped -PNAME=V without error, so compile rejects any -P not scoped to
gpu." That is accurate, but scoping is not the only way to get an ignored switch. On a design `gpu #(A = 1)` with a
`sub #(S = 64)` instance:

| Switch | iverilog exit | Message | `A` elaborated |
|---|---|---|---|
| `-Pgpu.A=2` | 0 | none | 2 |
| `-Pgpu.A=GTO`, `-Pgpu.A=16k`, `-Pgpu.A=`, `-Pgpu.A` | 0 | `error: invalid value specified for defparam: gpu.A` | 1 |
| `-Pgpu.s.S=8`, `-Pgpu.gpu.A=2`, `-Pgpu.-Pgpu.A=2` | 0 | none | 1 (and `S` stays 64) |
| `-Pgpu.a=2` (misspelt name) | 2 | `parameter 'a' not found` | — |

On the real design:

```
make compile BUILD=build/a20/m PARAMS="-Pgpu.ICACHE_LINES"; echo $?                       # 0, "invalid value" on stderr
make compile BUILD=build/a20/m PARAMS="-Pgpu.cores[0].core_instance.SMEM_SIZE=32"; echo $?  # 0, no message
```

Observed: exit 0, and `make test_<x>` then passes with default-config metrics.
Expected: compile fails, as it does for unscoped and misspelt switches.

Plausible inputs:
- `WARP_POLICY=GTO`: CLAUDE.md and ROADMAP name the policies "0 LRR, 1 GTO, 2 GTLRR". The build silently stays LRR,
  and `gpu.sv`'s `WARP_POLICY ∉ {0,1,2}` `$fatal` never sees the value.
- A hierarchical `SMEM_SIZE`: CLAUDE.md says that `SMEM_SIZE` is a `core`, not a `gpu`, parameter, which invites
  this form.

No documented command uses either form, so this is Medium, not High (compare AUDIT-11 M1 with AUDIT-16 H1).

Fix: in `check_params`, also reject any `-P` word that does not match `-Pgpu.NAME=VALUE`, with `NAME` a single
identifier and `VALUE` a plain integer. Alternatively, run iverilog with its stderr checked, for example
`… 2>&1 | tee …; ! grep -q 'error:'`.

## M2 — `bench/run.py --params` doubles a `-Pgpu.` prefix

`iv(params) = " ".join(f"-Pgpu.{p}" for p in params.split())`. The bench docstring uses `NAME=V`, but the Makefile
comment, CLAUDE.md's "Pass them via `iverilog -Pgpu.NAME=VALUE`" and every `make … PARAMS=` example use `-Pgpu.NAME=V`.
A user who copies the second form into `--params` gets:

```
python3 -c "import runpy; print(runpy.run_path('bench/run.py')['iv']('-Pgpu.ICACHE_LINES=16'))"
# -Pgpu.-Pgpu.ICACHE_LINES=16
make test_matadd BUILD=build/a20/dbl PARAMS="-Pgpu.-Pgpu.ICACHE_LINES=16"   # exit 0, PASS, ICACHE_LINES = 0
```

Observed: the bench run reports every kernel PASS, with I-cache hits 0, and writes the requested params into the
rollup JSON.
Expected: `--params` either accepts the `-Pgpu.` form or rejects anything that is not `NAME=INT`.

`run_one` keeps make's output only on failure, so the iverilog errors from M1 never reach a bench user either. Fix:
strip a leading `-Pgpu.` in `iv()` or reject it, and validate each word against `^[A-Z_]+=\d+$`.

## L1 — trace API outside the contracts; callable picks unchecked

What SG25 added:
- `xf.isasim.run(warp_policy=<callable>)`;
- `xf.isasim.run_trace(trace, program, data, threads, **kw)`;
- `test/helpers/runner.run_kernel(..., trace=dict)`.

The ADRs (014, 016), the isasim module docstring and the test docstrings describe these. PLAN contract 9 does not; its
`warp_policy` sentence is now wrong. Contract 5 and `run_kernel` (which has no docstring) do not mention `trace`,
and `run_launches` does not take it. `run_trace` itself checks every pick and the trace length. A callable passed to
`run` directly is trusted:

```python
P = assemble("CONST R0, #0\nBAR\nSTS R0, %threadIdx\nLDS R1, R0\nSTR %threadIdx, R1\nRET")
run(P, [], 4, warps_per_core=2, warp_policy=pick_warp0_on_issue_3).memory[:4]   # [1, 1, 3, 3], no error
run(P, [], 4, warps_per_core=2).memory[:4]                                      # [3, 3, 3, 3]
run(P, [], 4, warps_per_core=2, warp_policy=lambda b, r: 0)                     # IndexError: list index out of range
```

Fix: in `run`, raise `IsaSimError` when a callable returns an index that is not in `ready`, and so drop the
duplicate check from `run_trace`. Add the callable form and `run_trace` to contract 9, and `trace=` to the runner's
docs (with AUDIT-11 L8's `-> dict` fix).

## L2 — `stores_conflict` and same-value stores

The STR branch records a conflict only when `seen.get(a, v) != v`. That is a reasonable definition, because the
result does not depend on the winner. But `docs/isa.md` says "these stores" (every same-word `STR`), and the field
comment says only `(block, pc, addr)`. Fix: write "lanes that store different values to one word" in `docs/isa.md`
and in the field comment. This also closes the rest of AUDIT-16 L1 (below).

## L3 — ADR-009 Tests paragraph

ADR-016 changed `controller.sv` so that writes skip the priority pass, and ADR-009 §Status and §Priority were updated.
§Tests still describes the monitor as checking any lane. `test_medic.py`'s module docstring has it right: "never
gives a channel to a write or a low-utility read while a high-utility read waits". §Context item 3 ("high-utility
warps' requests go first") describes the paper and can stay. Fix: write "a high-utility read" in §Tests.

## Earlier findings, re-checked at HEAD

| Finding | Status | Description accurate? |
|---|---|---|
| AUDIT-16 H1 / AUDIT-11 M1 (unscoped `-P`) | fixed (SG27, `ce22326`); the usage comment is scoped and `compile`/`compile_xf` reject unscoped `-P` | — (residue: M1, M2 above) |
| AUDIT-16 M1 (`test_medic` replaces `PARAMS`) | fixed; `PARAMS` are appended, `MEDIC=0` is rejected; re-run above logs MEDIC=0 then MEDIC=1 | — |
| AUDIT-16 L1 (`stores_conflict` scope, `sts_conflict` comment) | **partly fixed**. The isasim module docstring and `docs/isa.md` now say that `stores_conflict` lists STR cases, and the `isa_fixtures` docstring no longer says that conflicting cases are skipped. Still open: the field comment `# (block, pc, addr)`, contract 9, and the `STS_CONFLICT` comment "stores its thread id" (it stores `blockIdx*blockDim+threadIdx`) | the xfail reason ("neither contract 9, isasim nor isa_fixtures says so") is now partly stale; the assertions (field comment, fixture comment) are still accurate |
| AUDIT-16 L2 (I-cache counter semantics) | open (xfail) | yes; `architecture.md`, ADR-012 and contract 12 unchanged |
| AUDIT-16 L3 (`test_all` descriptions) | open (xfail) | yes; README and CLAUDE.md lines unchanged |
| AUDIT-11 L1 (mq=0 into i8) | open (xfail) | yes |
| AUDIT-11 L2 (`BRp`) | open (xfail) | yes; PLAN SG22 line 621 still says the assembler rejects `BRp`, and `docs/isa.md` still gives `BRzp` |
| AUDIT-11 L3 (program > 256 words) | open (xfail) | yes |
| AUDIT-11 L4 (failed bind leaves `<out>.args`) | open (xfail) | yes; a thread-limit refusal allocates nothing (`test_control_relu_binder_takes_255_and_refuses_256`), a constant-range refusal still does |
| AUDIT-11 L5 (binder vs golden LUT checks) | open (xfail) | yes |
| AUDIT-11 L6 (`ValueError` vs `IsaSimError`; >255 threads) | open (xfail) | yes; L1 above adds `IndexError` from a callable policy |
| AUDIT-11 L7 (negative `alloc` shape) | open (xfail) | yes |
| AUDIT-11 L8 (`run_kernel -> dict`, missing contract-12 keys) | open | yes; SG25 added `trace=` without fixing the annotation or the module docstring |
| AUDIT-11 L9 (CLAUDE.md duplicate rows, `DATA_MEM_*`, `.PHONY bench`, ROADMAP "8- and 32-bit") | open | yes; all four unchanged |
| AUDIT-11 L10 (documented bench commands overwrite frozen JSON; `--params` ignored under `--phase 11`/`--xf`) | open | yes |

## Checked and consistent

- **The STR rule.** `docs/isa.md`, ADR-016, the isasim docstring and the `STR` branch agree that the highest lane
  wins. `controller.sv` gives writes no priority pass (line 125, `!PRIORITY_ENABLE || pass == 1`), and
  `memory.py` applies a cycle's writes from channel 0 up. ADR-009 §Status and §Priority say "reads". The ADR-016
  claims all check out:
  - `test_medic_str_conflict` and `test_divergence::test_str_conflict` exist;
  - the MeDiC numbers (18/28, 94 requests, 781/1071/1813) match `docs/final-report.md`;
  - `str_conflict{8,32}` are current.
- **The RET row.** "End the issuing warp …; the block finishes when all its warps have" matches the model (a warp is
  done when its stack empties), `architecture.md` ("ends the warp on `RET`") and ADR-011/014.
- **ADR-014's harness paragraph.** It matches `test_isa_diff`: `trace = {} if warps > 1`, `run_trace` at W > 1,
  `run` at W = 1, and `warp_issues_total == steps`. No doc still describes the old `lrr == gto` gate or
  `isa_diff_skipped`.
- **The runner's trace.** It records `issue`/`selected_warp`, which `scheduler.sv` drives at every warp count, keyed
  by the core's `block_id`, which matches the model's `b`. `run_trace` checks picks against the ready set and the trace
  length (the control test replays a recorded LRR order and rejects one extra issue).
- **Contract 12 text.** Unchanged, and still consistent with `check_counters` and ADR-012. That `warp_issues_total`
  equals `steps` is stated in ADR-012 and `architecture.md`.
- **SG26.** The `SMEM_WRAP` comment matches its program at both widths (`65t` wraps to word `t`, `459 − t` (8-bit
  `203 − t`) wraps to `11 − t`). The launch fixtures launch 253 and 255 threads, at or below `_bind`/`matmul`'s
  255-thread limit; the model reproduces them. The relu 253/255 cases are inside that limit, and 256 is refused
  before any allocation. The LAUNCH comment's "block count needs 9 bits" means the `thread_count + THREADS_PER_BLOCK − 1`
  intermediate (B02), not the block count itself (64). That is loose wording but not misleading.
- **SG27 commands.** Every documented `make` and `bench/run.py` command still resolves:
  - `make test_l1d_indepth PARAMS="-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1"`;
  - `test_isa_diff PARAMS="-Pgpu.WARPS_PER_CORE=4"` and `test_xf_matmul PARAMS=-Pgpu.THREADS_PER_BLOCK=8`;
  - `test_medic`, `test_multiwarp`, `test_divergence`, `regression_xf`, `harness_selftest` and the
    `test_warp_scheduler_*` targets;
  - the phase 3/4/5/9/11, `--xf` and `--check-citations` bench commands.

  Every `-P` in the docs is `gpu.`-scoped, except the warp-scheduler testbench's own `WS_PARAMS`, which bypass
  `compile`. `matrix_run` re-enters `test_medic` with a full `MEDIC=1` configuration, and the `override` keeps it.
