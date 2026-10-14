# AUDIT-3 — Test coverage (round 1)

Lens: would the cocotb and pytest suites catch real bugs in the RTL, ISA extensions, assembler,
export, golden ops and kernels? Audited at `505887c`, iverilog 12.0, cocotb 2.0.1, torch 2.14.1,
nanoGPT Extended `03dbab8`. AUDIT-1, AUDIT-2 and AUDIT-4 findings are not repeated.

Method: 20 hand-placed mutants (one plausible bug each) run against the targets that cover the
mutated code, `coverage.py` over `xf/` under the full pytest suite, and configurations no test runs.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | L1-D cache correctness is untested: a stale-data cache and a never-hitting cache both pass every target, `test_l1d_indepth` included | below |
| H2 | High | Parameter-gated RTL runs only when someone passes `PARAMS`: an I-cache that hangs every kernel passes every documented command | below |
| M1 | Medium | The int32 overflow guards (`export.check_bounds`, `attention.check_bounds`) can be deleted without any test failing | below |
| M2 | Medium | Without torch and `NANOGPT_DIR`, `pytest xf/tests` is green while skipping the fixture-freshness check and AUDIT-1's tolerance xfail | below |
| L1 | Low | Documented edge semantics with no test: `DIV` by zero, the R13–R15 write guard, `run_launches`' width check | below |
| L2 | Low | Weak pytest discriminators: golden RMSNorm LUT clamp, `fixed_point` rounding term, `check_results.py` skip handling | below |

Checked and covered (mutant killed):
- **Kernels.** Dropping the RMSNorm LUT-index clamp (`test_xf_rmsnorm`), a stray `STR` 32 words
  past `residual_add`'s output (`test_xf_elementwise`) and dropping softmax's `S/2` rounding
  (`test_xf_attention`, 14 cases) are all caught. Every xf cocotb test checks every output
  bit-exact against `xf.golden`, and the chain tests check every intermediate.
- **Golden and assembler.** Requant clamping at −127, softmax row max taken over masked entries, and
  an assembler that accepts writes to R13–R15 each fail `pytest xf/tests`. `decode`/`encode` are
  checked over all 2^16 words.
- **COALESCE.** Coalescing on the low 8 address bits fails SG4's
  `make test_xf_matmul PARAMS="-Pgpu.COALESCE=1 -Pgpu.PIPELINED=1"`.
- **Configurations at HEAD.** Configurations no test exercises all pass today:
  every `test_xf_*` kernel and chain target with 3-cycle memory latency
  (`run_launches(mem_latency=3)`; every test uses 0), the MLP block with it under ICACHE_LINES=16
  PIPELINED=1 and attention under DCACHE_LINES=16 COALESCE=1; `test_xf_attention` with all four of
  COALESCE, PIPELINED, ICACHE_LINES=16, DCACHE_LINES=16; `test_xf_rmsnorm` with DCACHE_LINES=16
  MEDIC=1; and 255-thread relu, 17×15 transpose, 15×8×17 naive and 60×8×16 tiled matmuls at
  NUM_CORES=2 and NUM_CORES=4 COALESCE=1. So H1 and H2 are missing tests, not known RTL bugs. The
  one configuration that fails is a test-harness timeout (H2).
- `coverage.py` (branch) over `xf/` under the full pytest suite with torch: 89 %. Uncovered are
  `export.check` / `main` (run only by `python3 -m xf.export --check`), the `check_bounds` raise
  paths (M1) and a few argument-validation raises.

Mutants that survive by construction: replacing both `BAR`s in `matmul_tiled.asm` with `NOP`
passes `test_xf_matmul` and `test_xf_mlp_block`, because BAR is a decoded no-op under lockstep
(already documented in AUDIT-4).

## H1 — L1-D cache correctness is untested (High)

`test/test_l1d_indepth.py` is the only L1-D test. Its hit-count assertion accepts any result
(`h + m == 0 or h >= m or (h == 0 and m > 0)`), `test_l1d_disable_equivalence` does not check the
request count its docstring promises, and the "survives multiple blocks" and "MeDiC bypass" cases
in the docstring do not exist. Run without `PARAMS`, it does not instantiate the cache at all. No
xf kernel reads an address it wrote in the same launch, and `run_launches` resets the GPU (and the
cache) between launches, so the xf suites cannot see a coherence bug either.

Two mutants of `src/l1d_cache.sv` pass every target run with `DCACHE_LINES=16`:

| Mutant | Effect | Passes |
|--------|--------|--------|
| Write hit does not update the line (`data_arr[...] <= mem_write_data[i];` → `;`) | every read after a write to a cached address returns the old value | `test_l1d_indepth`, `regression`, `test_stress`, `test_launches`, `test_xf_matmul`, `test_xf_elementwise`, `test_xf_rmsnorm`, `test_xf_mlp_block`, `test_xf_attention` |
| A fill never sets `valid` (`<= 1'b1` → `<= 1'b0`) | the cache never hits; Phase 9's speedup and `docs/xf-bench.md`'s hit counts vanish silently | `test_l1d_indepth`, `regression`, `test_xf_matmul` |

A 12-line directed test kills the first one: each thread loads `mem[i]` twice, stores `mem[i] + 1`,
loads it again and stores that to `out[i]`. It passes at HEAD with DCACHE_LINES=16, with MEDIC=1
and with DCACHE_LINES=4 COALESCE=1, and fails with the mutant (`stale read after write`).

Repro:
```bash
sed -i 's/data_arr\[idx_of(mem_write_address\[i\])\] <= mem_write_data\[i\];/;/' src/l1d_cache.sv
make test_l1d_indepth PARAMS=-Pgpu.DCACHE_LINES=16 && make regression PARAMS=-Pgpu.DCACHE_LINES=16 \
  && make test_xf_matmul PARAMS=-Pgpu.DCACHE_LINES=16          # all pass
git checkout src/l1d_cache.sv
sed -i "s/valid_arr\[idx_of(mem_read_address\[i\])\] <= 1'b1;/valid_arr[idx_of(mem_read_address[i])] <= 1'b0;/" src/l1d_cache.sv
make test_l1d_indepth PARAMS=-Pgpu.DCACHE_LINES=16             # passes with zero hits
git checkout src/l1d_cache.sv
```

Fix direction (SG11): a directed read-after-write test (same thread, and a write by one block read
by a later block) at both word widths, exact hit/miss assertions for the hot-address case with and
without MeDiC, and `test_l1d_indepth` run with `DCACHE_LINES=16` by a make target (see H2).

## H2 — Parameter-gated RTL is exercised only by hand (High)

ICACHE_LINES, PIPELINED, COALESCE, DCACHE_LINES, MEDIC and NUM_CORES each instantiate different RTL,
but every make target builds with whatever `PARAMS` the caller passes, defaulting to none.
`test_param_sweep` says it runs "under varying ICACHE_LINES / COALESCE / PIPELINED / DCACHE_LINES"
but is a single cocotb test of a 9-instruction copy kernel at the build's parameters. The only
configured parameters in any PLAN Verify command are SG4's `COALESCE=1 PIPELINED=1` on
`test_xf_matmul`; `bench/run.py --xf` adds COALESCE and DCACHE_LINES for two workloads. The SG4 and
SG6 acceptance "passes under default, COALESCE=1, ICACHE_LINES=16, PIPELINED=1, DCACHE_LINES=16"
is not reproducible by any command, ICACHE_LINES and MEDIC never meet an xf kernel, and no test
runs with memory latency (`mem_latency` is 0 everywhere). There is also no target that runs the
whole suite: `make regression` covers 8 of the 25 test modules, so "the full existing suite" in the
Dispatch protocol is reassembled by hand each time.

Running the matrix by hand finds one failure already: `make test_xf_rmsnorm
PARAMS=-Pgpu.NUM_CORES=1` fails `test_synthetic` with `Kernel did not complete in 20000 cycles`. The
elementwise and RMSNorm tests launch with `Launch`'s default 20,000-cycle timeout (the matmul, MLP
and attention tests pass 200,000), which a single core exceeds; with 400,000 both
`test_xf_rmsnorm` and `test_xf_elementwise` pass at NUM_CORES=1. NUM_CORES is a documented `gpu`
parameter (CLAUDE.md).

Mutant: the I-cache hit ignores the tag (`src/fetcher.sv`, `hit = valid_arr[idx];`). Under
ICACHE_LINES=16 every kernel fetches wrong instructions and `make regression` times out. It passes
`make regression`, `make test_param_sweep`, SG4's Verify and `make test_xf_mlp_block`.

Repro:
```bash
sed -i 's/hit = valid_arr\[idx\] \&\& (tag_arr\[idx\] == tag);/hit = valid_arr[idx];/' src/fetcher.sv
make regression && make test_param_sweep \
  && make test_xf_matmul PARAMS="-Pgpu.COALESCE=1 -Pgpu.PIPELINED=1"   # all pass
make regression PARAMS=-Pgpu.ICACHE_LINES=16                          # TimeoutError
git checkout src/fetcher.sv
make test_xf_rmsnorm PARAMS=-Pgpu.NUM_CORES=1                         # fails at HEAD (timeout)
```

Fix direction (SG12): one make target for the whole suite, and one that runs `regression`,
`regression_xf` and every `test_xf_*` across a fixed configuration list (default; ICACHE_LINES=16;
PIPELINED=1; COALESCE=1; DCACHE_LINES=16; DCACHE_LINES=16 MEDIC=1; all of them; NUM_CORES=1;
NUM_CORES=4) plus one memory-latency run, in separate build directories so configurations can run
in parallel, with the elementwise/RMSNorm launch timeout raised to match the other xf tests.
Apart from the timeout above, everything listed already passes at HEAD, so this is test-only.

## M1 — The int32 overflow guards are unchecked (Medium)

Contract 8 rests on `export.check_bounds` and `attention.check_bounds` proving that no int8 input
can overflow an int32 intermediate, because the hardware wraps silently where the golden model
raises. No test checks that either function detects anything: `test_fixture_bounds` asserts the
returned values are ≤ INT32_MAX, which the function already guarantees by raising. These mutants
pass the full pytest suite (402 passed, 7 xfailed, with torch and `NANOGPT_DIR`):

- `export.check_bounds` drops the `up` projection (`("gate", "up", "down")` → `("gate", "down")`).
- `export.check_bounds` never raises (`if bad:` → `if False:`, `xf/export.py:138`).
- `attention.check_bounds` never raises (`xf/attention.py:103`).

The formulas are sound today (AUDIT-2 re-derived them), and the `test_no_overflow_on_any_int8_input`
sweeps exercise `golden._i32` on ±128/127/random inputs, but those inputs do not reach the
sign-matched worst case the bounds are about. The MLP version also asserts `out` is int8, which
requant's clamp makes always true. Fix direction: per term, an image with that M raised by one
step past the bound must raise, and the bound must equal a brute-force maximum over sign-matched
operands.

## M2 — Torch-gated checks skip silently (Medium)

`python3 -m pytest xf/tests -q`, the Verify command every thread runs since SG1, reports
`400 passed, 5 skipped, 4 xfailed` without torch or without `NANOGPT_DIR`. The five skips are
`test_export_matches_fixtures[mlp_block]` and `[attention]` (the only check that the committed
fixtures match a fresh export), AUDIT-1's strict-xfail MLP tolerance test and AUDIT-2's two
held-out tolerance xfails. Those xfails are what SG9 and SG10 must flip, and SG9/SG10 regenerate the
fixtures, so a builder whose run skips them sees green either way. `python3 -m xf.export --check` (the 200- and 2000-input tolerance gates) is
not part of pytest at all. Fix direction: make the skip loud (fail when `NANOGPT_DIR` is set but
torch is missing, and name the skipped checks in the SG Verify lines), or give the torch tests a
marker that SG Verify commands select explicitly.

## L1 — Documented edge semantics with no test (Low)

- `DIV` returning 0 for `rt = 0` (`docs/isa.md:50`): `(rt == 0) ? 0` → all ones passes
  `regression`, `test_edge_cases`, `test_isa_ext` and `test_xf_isa_ext`.
- The register file ignores writes to R13–R15 (`decoded_rd_address < 13` → `< 16` passes the same
  targets). The assembler rejects such writes, so only hand-encoded programs can reach it.
- `run_launches` rejecting an image whose width differs from the DUT bus (contract 5): removing the
  check passes everything.

## L2 — Weak pytest discriminators (Low)

- `golden.rmsnorm_int` clamping the LUT index to `len - 2` instead of `len - 1` passes pytest: the
  test LUT's last two entries are equal after rounding. `test_xf_rmsnorm` catches it.
- `export.fixed_point` ignoring the `2^(s-1)` rounding term passes
  `test_fixed_point_fits_and_is_tight`: random bounds never land where the term decides `s`.
- `test/helpers/check_results.py` counting skipped tests as run passes `make harness_selftest`,
  which has no all-skipped fixture (the current code does reject that case).
