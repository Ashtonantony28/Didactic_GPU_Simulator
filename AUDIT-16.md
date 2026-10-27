# AUDIT-16 — API/SDK consistency (round 4)

Lens: do the public surfaces agree with each other and with their docs? This round repeats AUDIT-11's scope and
adds everything merged since `aa27256`:
- SG24: `test_xf_controller`, the I-cache assertions in `check_counters`, the `sts_conflict{8,32}` fixtures, and
  the STS rule in `docs/isa.md` and ADR-007;
- AUDIT-9, -10, -12, -13 and -14, as documentation and test surfaces.

I re-checked every AUDIT-11 finding. SG25's STR rule (AUDIT-14 H1, in progress) is out of scope.

Audited at `66f1dad` (main) with iverilog 12.0, sv2v 0.0.13 and cocotb 2.0.1, with `NANOGPT_DIR` set for pytest.
Builds went under `build/a16/`. The machine was running a matrix, so I ran only single targets at `-j1`.

Method:
- **Commands.** I extracted every `make …` and `python3 …` command from README, CLAUDE.md, ROADMAP, PLAN, the
  final report, `docs/architecture.md`, the ADRs, the phase reports and the docstrings, and resolved each against
  the Makefile and `bench/run.py`. All of them resolve. The new `make test_xf_controller` and `test_controller`
  targets resolve too.
- **Parameters.** I compared the parameter tables with `src/gpu.sv`. I compared how `PARAMS` behaves on each
  configured target: `test_xf_%`, `multiwarp_%`, `divergence_%`, `matrix_%` and `test_medic`.
- **SG24 surfaces.** I compared `check_counters` with contract 12, ADR-012 and the `architecture.md` telemetry
  table. I compared the STS rule text with `shared_memory.sv` and `xf.isasim`, and the `sts_conflict` fixture
  comment with its program and memory. I compared the `isa_fixtures` docstring with `Result.stores_conflict`.
- **AUDIT-11 status.** I ran `xf/tests/test_audit11.py`: all 10 strict xfails still xfail. I also re-read each
  finding against the code at HEAD.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` | 775 passed, 35 xfailed (with `test_audit16.py`) |
  | `python3 bench/run.py --check-citations` | `citations: OK` |
  | `python3 -m xf.isa_fixtures --check` | `ok: 83 ISA fixtures current, skipped []` |
  | `python3 -m xf.asm kernels/relu.asm` | prints binary words |
  | `make test_xf_controller BUILD=build/a16/xc` | 5/5 PASS |
  | `make test_medic BUILD=build/a16/medic_partial PARAMS="-Pgpu.DCACHE_LINES=16"` | exit 0, both runs log `MEDIC=0` (M1) |
  | `iverilog … -PICACHE_LINES=16 -PCOALESCE=1 build/a16/m1/gpu.v` | exit 0, every `ICACHE_LINES` in the `.vvp` is 0 (H1) |

There is **one high finding**, H1, which re-rates AUDIT-11 M1. It needs an SG row.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | Re-rates **AUDIT-11 M1** (still open, accurately described). The Makefile's usage line, `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`, builds the default GPU and exits 0. The lens rule is that a documented command giving a silently wrong result is high, so Medium was too low. Nothing in `compile`/`compile_xf` rejects the unscoped form | `test_audit16.py::test_compile_rejects_unscoped_params`; also `test_audit11.py::test_makefile_params_example_is_scoped` |
| M1 | Medium | `make test_medic PARAMS=…` **replaces** the target's `-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1` instead of appending to it. With `PARAMS="-Pgpu.DCACHE_LINES=16"` both builds are MEDIC=0, the "MEDIC=1" run takes the baseline branch, and the target passes (exit 0) without exercising MeDiC. The same happens through `python3 bench/run.py --params "DCACHE_LINES=16" medic` | `test_audit16.py::test_medic_params_are_appended`; control `::test_control_medic_default_params` |
| L1 | Low | `Result.stores_conflict` flags only `STR` conflicts. An `STS` conflict, such as the `sts_conflict` program, gives `[]`. Contract 9, the field comment `# (block, pc, addr)` and the `isa_fixtures` docstring ("store-conflicting are not written") do not say this, although a fixture named `sts_conflict` is written. The fixture comment says each lane "stores its thread id", but it stores the global id `blockIdx*blockDim+threadIdx` (memory `[3,3,3,3,7,7,7,7]`) | `test_audit16.py::test_stores_conflict_scope_documented`; control `::test_control_sts_conflicts_are_not_flagged` |
| L2 | Low | The I-cache counter semantics that SG24 now enforces are undocumented. At `ICACHE_LINES=0`, `icache_misses_total` counts every fetch (`bench/phase-11-all.json#kernels/default/matmul/L0`: hits 0, misses 41 = `warp_issues`). `check_counters` asserts hits + misses == `warp_issues_total` at W=1. Contract 12, ADR-012 and `architecture.md` say only "summed over all cores", and `check_counters`' docstring still cites only contract 12 | `test_audit16.py::test_icache_counter_semantics_documented` |
| L3 | Low | The `make test_all` descriptions drift from the recipe. README omits `harness_selftest` and `isa_fixtures --check`; CLAUDE.md omits `isa_fixtures --check`. The Makefile comment lists both | `test_audit16.py::test_test_all_descriptions_match_makefile` |

## H1 — Makefile usage line passes unscoped parameters (AUDIT-11 M1, re-rated High)

Status: open. `Makefile` line 6 is unchanged: ``# Pass parameters via DEFINES, e.g. `make test_matadd
PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`.`` AUDIT-11's description is still accurate, except for a count: the
`.vvp` now has five `ICACHE_LINES` parameter records, all 0, where AUDIT-11 counted three.

```
make gpu_v BUILD=build/a16/m1
iverilog -o build/a16/m1/a.vvp -s gpu -g2012 -PICACHE_LINES=16 -PCOALESCE=1 build/a16/m1/gpu.v; echo $?   # 0
grep '"ICACHE_LINES"' build/a16/m1/a.vvp      # every record +C4<0…0>
```

Observed: exit 0, and the default GPU is built. `make test_matadd` with that `PARAMS` passes with default-config
metrics.
Expected: the documented example builds an I-cache plus coalescing GPU, or the build fails.

Why High: the line is the Makefile's own usage documentation, and following it gives a passing run whose numbers
belong to another configuration. That is the lens's definition of high. AUDIT-11 rated it Medium because every
other doc uses the scoped form, but that does not make the result less silent.

Fix:
- correct the comment to `PARAMS="-Pgpu.ICACHE_LINES=16 -Pgpu.COALESCE=1"` and "via PARAMS";
- have `compile`/`compile_xf` reject any `PARAMS` word that matches `-P` without a `.`, for example
  `$(if $(filter-out -P%.%,$(filter -P%,$(PARAMS))),$(error unscoped -P in PARAMS: use -Pgpu.NAME=V))`.

## M1 — `test_medic` overrides `PARAMS`

```make
test_medic: PARAMS = -Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1
test_medic: compile
	$(MAKE) … BUILD=$(BUILD)/medic0 PARAMS="$(subst MEDIC=1,MEDIC=0,$(PARAMS))" compile
```

A command-line `PARAMS` beats a target-specific assignment in GNU make. Every other configured target
(`test_xf_%` per contract 6, `multiwarp_%`, `divergence_%`) appends the user's `PARAMS` to its own.

Repro:

```
make test_medic BUILD=build/a16/medic_partial PARAMS="-Pgpu.DCACHE_LINES=16"; echo $?
```

Observed: both runs log `MEDIC=0 {'hot_hits': 14, … 'bypass': [0, 0, 0, 0, 0, 0, 0, 0] …}`, `TESTS=1 PASS=1` twice,
and exit 0. The second run hits `if not medic: … return`, so none of the MeDiC assertions run: bypass counts,
`hot_hits >= baseline`, priority violations, `decided`.
Expected: either the run is at MEDIC=1 against a MEDIC=0 baseline, or the target fails.

Cases where the override fails loudly or is harmless:
- With `PARAMS` that omit `DCACHE_LINES`, the test fails on `DCACHE_LINES >= 16`.
- `matrix_medic`/`matrix_all` and `bench/run.py --phase 11` pass a full `MEDIC=1` configuration.

Fix:
- set `MEDIC_PARAMS := -Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1` and build with `$(MEDIC_PARAMS) $(PARAMS)`, keeping
  the `subst` for the baseline;
- add `assert medic` (or `MEDIC_EXPECT`) to the second run, so that two MEDIC=0 runs cannot pass.

## L1 — `stores_conflict` scope and the `sts_conflict` fixture comment

`xf/isasim.py` appends to `res.stores_conflict` only in the `STR` branch. The `STS` branch writes lanes in order
and records nothing. As a result, `xf/isa_fixtures.py`'s "Cases the model flags as … store-conflicting are not
written" excludes `STR` conflicts only. The `STS` conflict fixtures are written by design, as SG24 intended. A user
who calls `run(...).stores_conflict` to vet a program for write races does not learn about SMEM races.

The `STS_CONFLICT` comment ("stores its thread id") is also inaccurate: `R2 = blockIdx*blockDim + threadIdx`.
The comment's claim that `BAR` makes the result interleaving-independent is AUDIT-14 L1 and is not re-reported.

Fix: name the field's scope in contract 9 and the field comment ("STR only; STS conflicts resolve to the highest
lane"), and write "global thread id" in the fixture comment.

## L2 — I-cache counters

`fetcher.sv` increments `icache_misses` on every fetch when there is no cache. SG24's `check_counters` makes two
behaviours part of the harness contract:
- `hits + misses == warp_issues_total` at `WARPS_PER_CORE = 1`;
- no hits without an I-cache.

`docs/architecture.md` says "`check_counters` checks the request and I-cache counters". However, its telemetry row,
ADR-012's counter list and contract 12 define the I-cache counters only as "summed over all cores". The bench JSON
rows for builds without an I-cache therefore show `icache_misses` equal to the instruction count with no documented
meaning. The final report prints "—" for those rows. Fix: add "every fetch is a hit or a miss; with
`ICACHE_LINES = 0` every fetch is a miss" to ADR-012/`architecture.md`, and cite it in `check_counters`.

## L3 — `test_all` descriptions

The Makefile recipe runs, in order:
1. every module at default parameters;
2. the xf modules;
3. the warp-scheduler targets, `test_medic` and `harness_selftest`;
4. the pytest suite;
5. `export --check`;
6. `isa_fixtures --check`.

The README line says "every test module at default params, xf pytest, export --check". The CLAUDE.md line adds
`harness_selftest` but not `isa_fixtures --check`. Neither line is wrong about what passes, but a user who reads
either one cannot tell that stale ISA fixtures fail `test_all`.

## Earlier findings, re-checked at HEAD

| Finding | Status | Description accurate? |
|---|---|---|
| AUDIT-11 M1 | open | yes (re-rated: H1 above) |
| AUDIT-11 L1 (mq=0 into i8) | open (xfail) | yes |
| AUDIT-11 L2 (`BRp`) | open (xfail) | yes; PLAN SG22 still says the assembler rejects it |
| AUDIT-11 L3 (program > 256 words) | open (xfail) | yes |
| AUDIT-11 L4 (failed bind leaves `<out>.args`) | open (xfail) | yes |
| AUDIT-11 L5 (binder vs golden LUT checks) | open (xfail) | yes; `softmax` likewise checks only the `exp_lut` shape, not non-negativity (same class, not a new finding) |
| AUDIT-11 L6 (`ValueError` vs `IsaSimError`, 300 threads) | open (xfail) | yes |
| AUDIT-11 L7 (negative `alloc` shape) | open (xfail) | yes |
| AUDIT-11 L8 (`run_kernel -> dict`, missing contract-12 keys) | open | yes |
| AUDIT-11 L9 (CLAUDE.md duplicate parameter rows, `DATA_MEM_*` not in tables, `.PHONY bench`, ROADMAP "8- and 32-bit") | open | yes |
| AUDIT-11 L10 (documented bench commands overwrite frozen JSON; `--params` ignored under `--phase 11`/`--xf`) | open | yes |

## Checked and consistent

- **SG24 STS rule.** `docs/isa.md` and ADR-007 match `shared_memory.sv`: its write loop runs in lane order, so the
  last enabled lane wins. They also match `xf.isasim`'s STS branch. Within a warp at W>1 the rule is per `STS`
  issue, as ADR-007 says ("across warps, the warp that issues its STS last wins").
- **`sts_conflict{8,32}`.** Both fixtures are current (`isa_fixtures --check`), and the model reproduces
  `[3,3,3,3,7,7,7,7]`.
- **`test_xf_controller`.** It imports the five `test_controller` tests. It runs in `test_all` (as an `XF_TESTS`
  module) and in every matrix configuration: `XF_warps4gto` and `XF_simt` remove only attention, MLP and bench.
  `make test_xf_controller` passes, 5/5.
- **`check_counters`' new assertions** are guarded by `WARPS_PER_CORE == 1` and read `ICACHE_LINES` from the DUT,
  as the SG24 commit describes.
- **Commands.** Every documented `make` target and `python3` command resolves:
  - `test_xf_<name>`, the `test_warp_scheduler_*` targets, `multiwarp_*`/`divergence_*`, `harness_selftest`,
    `regression_xf` and `compile_xf`;
  - `bench/run.py --phase N/--xf/--check-citations`;
  - `xf.export [--check]`, `xf.isa_fixtures [--check]` and `xf.asm file.asm`.

  `--check-citations` is OK.
- **Parameters.** The `gpu.sv` parameter list, defaults and documented `$fatal`s are as AUDIT-11 recorded. No
  doc other than the Makefile line uses an unscoped `-P`.
- **Contracts 3, 4, 5, 7, 9 and 12.** The signatures are unchanged since AUDIT-11, because no `xf/` library
  file other than `isa_fixtures.py` has changed since `aa27256`.
