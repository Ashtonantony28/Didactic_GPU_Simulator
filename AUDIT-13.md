# AUDIT-13 — Test coverage (mutation), round 4

Lens: which behaviour would no test catch if it broke? "Tests" means the cocotb modules under `test/`, the pytest
suite under `xf/tests`, `harness_selftest` and the matrix configurations. This round:
1. confirms that SG24 closed AUDIT-12 H1–H3;
2. seeds mutants in areas neither AUDIT-9 nor AUDIT-12 mutated:
   - block assignment and `%blockIdx`/`%blockDim`;
   - the LSU and controller under `MEM_LATENCY` > 0;
   - BAR counting at `WARPS_PER_CORE` > 1 with the stack off;
   - SMEM masking;
   - the ALU at `WORD_BITS=32`;
   - the new `check_counters` assertions;
   - the xf binders and golden at edge shapes;
   - the `xf.isasim` model, run through the RTL differential.

I audited at `4444c25` (main) with iverilog 12.0, sv2v, cocotb 2.0.1, torch and `NANOGPT_DIR=/home/claude/nanogpt`.
All builds went under `build/mut/`, `build/a13head/` and `build/mut/probe/`. `git diff --stat src` is empty.

Method:
- **RTL mutants (28 new, plus the 4 AUDIT-12 reruns).** Each mutant is a one-line edit (two lines for B02) to a copy
  of `src/` under `build/mut/<id>/src`. It is built with `GPU_SRC` pointing at the copy, one build per configuration
  under `build/mut/<id>/<cfg>`. The driver is `build/mut/driver.py`, copied from AUDIT-12's. It runs the targets in
  matrix order, one at a time, and stops at the first failure:
  - the 8-bit suite: `regression`, `test_l1d_indepth`, `test_isa_diff`, `test_controller`, `multiwarp_run`;
  - the xf suite: `regression_xf`, then the `test_xf_*` modules (the long chains are left out where the matrix
    leaves them out);
  - for mutants live at default parameters, the `test_all`-only modules: `test_shared_mem_indepth`,
    `test_edge_cases`, `test_stress`, `test_launches`, `test_param_sweep`, `test_isa_ext`, `divergence_run`,
    `test_bench_sweep` and `test_audit10`.

  At most two simulations ran at once. A container restart cut one run short (U11 at default), and I reran it in
  full.
- **Python mutants (17 binder/golden, 5 model).** Each is a one-line edit in a copy of the repo
  (`build/mut/py/<id>`). The driver is `build/mut/pydriver13.py`. It runs `XF_REQUIRE_NANOGPT=1 pytest xf/tests -x`,
  `xf.export --check` and `xf.isa_fixtures --check`. I then ran the five model mutants that change behaviour again
  with **only** the RTL differential, together with a sixth, Y06: `test_isa_diff` and `test_xf_isa_diff` at W = 1, 2 and 4 (GTO), from the
  mutated copy. That answers whether the RTL comparison by itself catches them.
- **Probes** are in a scratch cocotb module (`build/mut/probe/probe13.py`, not committed) that compares RTL memory
  with `xf.isasim`. They cover concurrent stores, store-then-load, STS to SMEM words 64–71, and 252, 253 and 255
  threads. I ran each at HEAD (all pass) and under the relevant mutants.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` (HEAD, before) | 764 passed, 21 xfailed |
  | AUDIT-12 K08/I02/I04/S02 reruns (`build/mut/closure.py`) | all 4 killed |
  | 28 RTL mutants as above | 21 killed, 7 survived (2 equivalent) |
  | 22 Python mutants | 18 killed, 4 survived (Y19 equivalent, Q17 not a finding) |
| 5 model mutants through the RTL differential alone (Y06, Y09, Y10, Y11, Y15) | Y10 killed; the other 4 pass (pinned by pytest) |
  | HEAD `make BUILD=build/a13head/def test_isa_diff`; HEAD `divergence_run` at `SIMT_STACK=1 THREADS_PER_BLOCK=8` | 34/34 pass; 6/6 pass |
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` (with `test_audit13.py`) | 767 passed, 27 xfailed (6 new) |
  | `python3 bench/run.py --check-citations` | `citations: OK` |

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | No RTL test uses an SMEM address ≥ 64. Dropping STS to words 64–255 instead of wrapping them (SM1), or reading them unmasked (SM2, returns X), survives every target at default parameters, xf and the `test_all`-only modules included. docs/isa.md and the model mask LDS/STS to 64 words, but no ISA fixture reaches word 64 | `test_audit13.py::test_isa_fixtures_pin_smem_wrap`; SM1 below |
| H2 | High | No RTL test launches 253–255 threads (the largest ISA fixture launches 144). B02, an 8-bit `total_blocks` rounding in `dispatch.sv`, survives every default-config target. Under B02 a 253- or 255-thread launch ends in 2 cycles with nothing written. The binders accept up to 255 threads | `test_audit13.py::test_a_fixture_launches_253_threads_or_more`; B02 below |
| M1 | Medium | No test bounds a cycle count. U02 makes the LSU finish a store without waiting for `mem_write_ready`. It survives `matrix_run` at default, latency, icache, cores1, cores4, dcache, pipelined and all (xf long chains excluded at the last six). There it changes only cycles: the raw probe drops from 263 to 251 cycles at default and from 215 to 197 at `all`. It loses stores at `THREADS_PER_BLOCK=8`, which only `make test_divergence` (`divergence_t8`) catches, outside `test_all` and the matrix. BC5 (BAR release a cycle late) survives warps2 and warps4gto the same way. `bench/phase-11-all.json` and the final report quote these cycle counts | `test_audit13.py::test_some_test_bounds_cycles`; U02 below |
| L1 | Low | The elementwise binders' 255-thread refusal is unpinned. With Q03 (`threads > 256`), pytest, `export --check` and `isa_fixtures --check` pass. A 256-element relu then binds, the 8-bit DCR sees 0 threads, and the model writes nothing. Only the matmul refusal is tested (`test_kernels::test_matmul_rejects`, which kills Q02) | `test_audit13.py::test_elementwise_thread_limit_pinned` |
| L2 | Low | `matmul_tiled`'s `N % 4` refusal is tested only at N = 6. Q13 (refuse only when N > 4) passes everything. (4×3)@(3×2) then binds 2 threads, and the model's result differs from `golden.matmul_i8` (also at N = 1) | `test_audit13.py::test_tiled_small_n_refusal_pinned` |
| L3 | Low | At W > 1, `test_isa_diff` drops every interleaving-dependent fixture from the RTL comparison without limit (`isa_diff_skipped` asserts only `warps > 1 or not skipped`). HEAD skips none. Under the model mutant Y06 (BAR releases when any warp arrives), 7 fixtures become interleaving-dependent. Run only through the RTL differential, Y06 then **passes** `test_isa_diff`/`test_xf_isa_diff` at W = 2 and 4: 31 cases run instead of 34 at 8 bits. Only `test_isasim::test_multiwarp_bar_matters` catches it | `test_audit13.py::test_isa_diff_skip_set_bounded_at_several_warps` (control: `test_control_no_fixture_skipped_at_head_and_bar_mutant_skips`) |

The two High rows follow the brief's rule: each mutant survives a configuration the matrix runs (default), and
changes data memory. Neither behaviour is used by a shipped kernel. This matches AUDIT-12 H3: `matmul_tiled` stages
16 words, and the largest xf launch is 64 threads.

**AUDIT-12 H1–H3 are closed.** I copied the edits verbatim from `/home/claude/audit12/build/mut/mutants.py` and
applied them to copies of HEAD's `src/`. The results are identical to AUDIT-12's `build/mut/<id>/src` (`diff -r`).
| Mutant | Config | Killed by | Message |
|---|---|---|---|
| K08 | coalesce, all | `test_xf_controller` (`test_aliased_addresses`) | `stored [16, 16, 16, 16, 3088, …]` |
| I02 | icache, all | `test_controller` (`test_icache_counts`) | `assert 6 == 8` (misses) |
| I04 | icache | `test_controller`, via `check_counters` | `8 I-cache hits + misses != 14 warp issues` |
| S02 | default | `test_isa_diff` (`sts_conflict8`) | `8 words differ … RTL 0x0, model 0x3` |

## Mutant kill summary

**RTL: 21 killed, 7 survived.** Of the survivors, 2 are equivalent (U11, SM3). The other 5 are H1 (SM1, SM2), H2
(B02) and M1 (U02, BC5). U06 and BC2 are killed in one configuration and equivalent in the other.

| Group | Mutant (edit) | Result |
|---|---|---|
| Dispatch / `%blockIdx` / `%blockDim` | B01 `registers[13] <= block_id[4:0]` | killed, `test_isa_diff` (`isa_ext_cmp8`, 36 blocks) |
| | B02 `total_blocks` rounded in 8 bits | **survived** default, all targets (H2) |
| | B04 `registers[14] <= 4` | killed at `THREADS_PER_BLOCK=8` by `divergence_run`; equivalent at 4 |
| | B10 dispatch while `blocks_dispatched <= total_blocks` | killed, `test_l1d_indepth` (request count) |
| LSU / controller latency | U02 LSU store `WAITING` ignores `mem_write_ready` | **survived** default, latency, icache, cores1, cores4, dcache, pipelined, all; killed at T=8 by `divergence_t8` (M1) |
| | U06 controller `WRITE_WAITING` ignores `mem_write_ready` | killed at latency (`regression`); equivalent at latency 0 |
| | U07 controller `READ_WAITING` ignores `mem_read_ready` | killed, `regression` (default and latency) |
| | U11 `WRITE_RELAYING` does not wait for the consumer's valid to drop | survived latency and default; equivalent (same argument as AUDIT-12 K03; probes give identical cycles and data) |
| BAR, W>1, stack off | BC1 count stalls on the release cycle | killed, `multiwarp_run` (warps2, warps4gto) |
| | BC2 `waiting \| warp_bar[w]` | killed warps4gto (`multiwarp_run`); equivalent at W=2 (two waiting warps always release) |
| | BC4 `bar_release = &warp_bar` (absent/done warps block) | killed, `multiwarp_run` (timeout) |
| | BC5 warps not ready on the release cycle | **survived** warps2 and warps4gto (timing only, +1 cycle per barrier; M1) |
| | BC8 warp 0 ignores BAR | killed, `multiwarp_run` / `regression` |
| | BC9 `warp_live` `<=` | killed, `regression` |
| SMEM | SM1 STS index unmasked | **survived** default, all targets (H1) |
| | SM2 LDS index unmasked | **survived** default, all targets (H1) |
| | SM3 LSU SMEM read address `rs[5:0]` | survived; equivalent (`SMEM_SIZE` is fixed at 64, AUDIT-11) |
| ALU at 32 bits | AL01 SRA by `rt[7:0]`; AL02 MAX unsigned at 32; AL03 MUL by `rt & 0x7FFFFFFF`; AL04 DIV 0 for `rt > 65535` | killed, `test_xf_isa_diff` |
| | AL05 SRA by ≥ W gives 0 | killed, `test_isa_diff` |
| | AL06 ADD −1 when both operands are negative at 32 | killed, `test_xf_elementwise` |
| Counters / `check_counters` | CC1 fetches count as hits without an I-cache | killed, `regression` (new `ICACHE_LINES or not hits` assertion) |
| | CC2 RET counted as an issue at W=1 | killed, `regression` (new `fetches == issues` assertion) |
| | CC3 re-issue after a yield counted | killed, `test_isa_diff` (`warp_issues_total != model steps`) |
| Stack-off divergence | DV1 W=1 warp follows lane 1 | killed, `test_isa_diff` |
| | DV2 W>1 warp follows its second lane | killed, `test_isa_diff` / `regression` |

**Python: 18 killed, 4 survived.**
- Killed:
  - by pytest: Q01, Q02, Q04–Q12, Q14, Q16, Q18 (binder and golden edge shapes, below), and the model mutants Y09
    (LRR pointer never advances), Y10 (RET not counted as a step), Y11 (SMEM mask 63) and Y15 (stack-off branch
    taken if any lane takes it).
- Survived:
  - Findings: Q03 (L1), Q13 (L2).
  - Not findings:
    - Y19 (RET leaves `warp.bar` set): equivalent, because done warps leave `live`.
    - Q17 (`golden.matmul_i8` stops range-checking `b`): every caller already passes int8 through `require_i8` or
      `_i8`, so valid inputs give unchanged results. Contract 4 asks only for int32 intermediate checks.
- Model mutants run through the RTL differential alone: Y10 is killed (`warp_issues_total 40 != model steps 38`).
  Y09, Y11, Y15 and Y06 pass at W = 1, 2 and 4. The fixtures exercise none of those behaviours (LRR order beyond
  barrier-safe programs, SMEM > 64, divergent stack-off branches, which are excluded from fixtures). For Y06 the
  fixtures are skipped (L3). All four are pinned by `xf/tests/test_isasim.py`, so none is a finding beyond L3.

Python mutant edits:
| ID | File | Edit | Result |
|---|---|---|---|
| Q01 | kernels.py | tiled `threads = (m // TILE) * n` | killed |
| Q02 / Q03 | kernels.py / elementwise.py | `threads > 256` | killed / **survived** |
| Q04 | elementwise.py | transpose output shape `[r, c]` | killed |
| Q05 | elementwise.py | softmax square check removed | killed |
| Q06 / Q07 | elementwise.py | rmsnorm `kmax` unclamped / LUT bound `n` | killed |
| Q08 | elementwise.py | `_rnd` no rounding at `s = 1` | killed |
| Q09 | golden.py | clamp to −127 | killed |
| Q10 / Q11 | golden.py | softmax `(S+1)//2` / zero-sum check `< 0` | killed |
| Q12 | kernels.py | `check_block` rejects `b` dividing 4 | killed |
| Q13 | kernels.py | `n % TILE and n > TILE` | **survived** |
| Q14 | elementwise.py | rmsnorm shift bound `31 - kmax` | killed |
| Q16 | image.py | allocator allows one word past memory | killed |
| Q17 | golden.py | `matmul_i8` does not check `b` | survived (not a finding) |
| Q18 | kernels.py | args prologue high byte `& 7` | killed |

## Reproducing a mutant

Every RTL repro has the same shape. The command passes at HEAD without `GPU_SRC`. It also passes with the mutant
(survivors), or fails (kills). Note the `tr`: AUDIT-12's `G=$(ls …)` keeps newlines, which break the make
command line.

```
mkdir -p build/mut/<id> && cp -r src build/mut/<id>/src     # then apply the edit to build/mut/<id>/src/<file>
G="$(ls build/mut/<id>/src/*.sv | grep -v warp_scheduler_tb | tr '\n' ' ')"
make BUILD=build/mut/<id>/<cfg> PARAMS="<CFG_cfg>" GPU_SRC="$G" matrix_run     # = matrix_<cfg> on the mutant
```

The driver ran the same targets one at a time (logs: `build/mut/<id>/<cfg>/<target>.log`).

## H1 — SMEM words ≥ 64 never reach the RTL (High)

SM1, `shared_memory.sv`:

```
-                if (lane_write_enable[i]) mem[lane_write_addr[i][IDX_BITS-1:0]] <= lane_write_data[i];
+                if (lane_write_enable[i]) mem[lane_write_addr[i]] <= lane_write_data[i];
```

SM2, the same file: `assign lane_read_data[g] = mem[lane_read_addr[g][IDX_BITS-1:0]];` → `mem[lane_read_addr[g]]`.

Both pass `matrix_run` at default parameters and every `test_all`-only module. docs/isa.md and `xf.isasim` mask
LDS/STS to the low 8 bits and then to 64 words. AUDIT-9 probed this (scratch, never committed), and no ISA fixture
or cocotb test has an SMEM address ≥ 64.

A probe kernel where thread `t` does `STS 64+t, t+1`, `BAR`, `LDS t`, `LDS 64+t` and stores both values shows the
effect. At HEAD it stores `[1, 2, 3, 4, 1, 2, 3, 4]` twice, which is the model's result. Under SM1 all 16 words are
0. Under SM2 the loads return X, and the harness raises `ValueError` in `memory._to_int`.

`test_audit13::test_isa_fixtures_pin_smem_wrap` reruns every fixture on a model whose STS drops addresses ≥ 64. No
memory changes.

Fix: add an ISA fixture that wraps SMEM addresses, for example the probe above (`xf.isa_fixtures`). It then runs in
`test_isa_diff`/`test_xf_isa_diff` in every matrix configuration.

## H2 — 253–255 threads never launched (High)

B02, `dispatch.sv`:

```
-    assign total_blocks = (thread_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
+    wire [7:0] tc_round = thread_count + THREADS_PER_BLOCK - 1;
+    assign total_blocks = tc_round / THREADS_PER_BLOCK;
```

B02 passes `matrix_run` at default parameters and every `test_all`-only module. Running the stores probe (thread
`t` stores `t` to `64 + t`):

| Threads | HEAD | B02 |
|---|---|---|
| 252 | 2370 cycles, matches the model | identical |
| 253 | 2375 cycles, matches the model | 2 cycles, nothing written |
| 255 | 2375 cycles, matches the model | 2 cycles, nothing written |

The xf binders refuse only above 255. The largest ISA fixture launches 144 threads (`isa_ext_cmp*`), so the
dispatcher's top 3 thread counts are untested.

Fix: add a fixture (or extend `straight_line`) that launches 255 threads.

## M1 — cycle counts are unbounded (Medium)

U02, `lsu.sv`, global write:

```
-                    WAITING: if (mem_write_ready) begin
+                    WAITING: begin
                         mem_write_valid <= 0;
```

Results:
- Survives the full matrix suite at `latency` and default, and the 8-bit and short xf suites at icache, cores1,
  cores4, dcache, pipelined and all.
- At 4 threads per block, 4 data channels and serialised instruction fetch, a core never has more stores in flight
  than free channels. So data is unchanged in every matrix configuration, and only cycles move:

  | Probe | HEAD | U02 |
  |---|---|---|
  | raw, default | 263 | 251 |
  | raw, latency 3 | 368 | 344 |
  | raw, all | 215 | 197 |
  | stores, default | 155 | 149 |

- At `THREADS_PER_BLOCK=8`, 4 of each block's 8 stores are lost (`stores` probe: 8 of 16 words; `divergence_t8`:
  `full_stack: 4 words differ`). That target belongs to `make test_divergence`, not to `test_all` or `matrix`.

BC5, `scheduler.sv`: `~(warp_bar & ~{W{bar_release}})` → `~warp_bar`. It passes warps2 and warps4gto (+1 cycle per
barrier).

No test compares `metrics['cycles']` with a value or a bound (`test_audit13::test_some_test_bounds_cycles`).
`bench/phase-11-all.json` is produced by `bench/run.py`, and nothing re-derives it in a test.

Fix: pin the cycle counts of a few small kernels per matrix configuration (exact at W=1, as SG20 already promises
"bit-identical cycles"). Alternatively, have `bench/run.py --check` compare a fresh run with the frozen JSON.

## L1–L3

- **L1, Q03.** In `xf/elementwise.py::_bind`, `if threads > 255:` → `if threads > 256:`. pytest, `export --check`
  and `isa_fixtures --check` pass. A 256-element `relu` binds 256 threads, and `thread_count` (8 bits) is 0.
  `test_audit13` runs the repo's binder tests in-process against the mutant: none fails. The control: a transposed
  output shape is caught by 3 cases.
- **L2, Q13.** In `xf/kernels.py::matmul`, `if n % TILE:` → `if n % TILE and n > TILE:`. Everything passes. A
  `matmul_tiled` binding with (4×3)@(3×2) or (4×3)@(3×1) then runs, and the model's result differs from
  `golden.matmul_i8`.
- **L3.** In `test/test_isa_diff.py::isa_diff_skipped`, `assert warps > 1 or not skipped`. Under Y06 (in
  `xf/isasim.py`, `if all(w.bar for w in live):` → `any`), `make test_isa_diff PARAMS=-Pgpu.WARPS_PER_CORE=2` from
  the mutated copy passes, with 31 cases instead of 34. `smem_cross8`, `sts_conflict8` and `tiled_matmul8` are
  skipped. HEAD skips nothing at W = 2 or 4, so asserting `not skipped` at every W costs nothing today.

Observations (no row):
- `test/helpers/memory.py::_to_int` replaces only lowercase `x`/`z`. An X on the bus (SM2) therefore raises
  `ValueError` instead of reading as 0. The failure is still a test failure, but it is less readable.
- The `latency` configuration does work as a timing configuration: U06 is equivalent at latency 0 and killed at
  latency 3.
- The new `check_counters` assertions kill CC1 and CC2 on their own (first failure in `regression`).
  `fetches == issues` is checked only at W = 1. At W > 1 only `test_icache_counts`'s `fetches >= steps` remains,
  and the matrix has no W > 1 configuration with an I-cache.
