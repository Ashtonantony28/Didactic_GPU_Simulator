# AUDIT-11 — API/SDK consistency

Lens: do the public surfaces agree with each other and with their docs? In scope:
- the `xf/` binders and libraries (`kernels.py`, `elementwise.py`, `image.py`, `fixture.py`, `export.py`, `golden.py`);
- the ISA tooling (`isa.py`, `asm.py`, `isasim.py`) against `docs/isa.md` and ADR-011 to ADR-015;
- `test/helpers/runner.py` against contracts 5 and 12;
- the Makefile targets and `gpu` parameters against CLAUDE.md, README.md, ROADMAP.md, `docs/architecture.md`
  and `docs/final-report.md`;
- the `bench/run.py` CLI.

I checked them against each other and against PLAN contracts 1–12. Audited at `aa27256` (main) with iverilog 12.0,
sv2v 0.0.13, cocotb 2.0.1 and torch, and without `NANOGPT_DIR` for pytest.

Method:
- **Binder probes.** I called every binder with wrong dtypes, wrong ranks, empty and 0-size shapes, shifts
  outside [0, 31], out-of-range constants, LUTs of the wrong shape or dtype, missing segments, and 255/256 threads.
  I then compared each one's preconditions with the matching `golden` function.
- **Assembler.** I tested every mnemonic and operand form, the immediate syntaxes (`#5`, bare `5`, `#0x80`, `#0b11`,
  `#010`, `#-1`, `#256`), the branch spellings, labels, writes to R13–R15, and programs of 257+ instructions.
  I also ran `assemble(disassemble([w]))` over all 65536 words.
- **ISA model.** I read `isasim.run` against contract 9, `docs/isa.md` (CMP/NZP, SRA, MAX, DIV, LDS/STS masking, BAR,
  `%blockDim`) and the ADR-011/015 stack rules (push order, pop loop, RET compaction, depth). I also probed its
  argument validation.
- **Runner.** I checked `run_kernel`/`run_launches`/`Launch`/`check_counters` signatures, defaults, return values and
  docstrings against contracts 5 and 12.
- **Docs and commands.** I extracted every `make …` and `python3 …` command from README, CLAUDE.md, ROADMAP, the final
  report, the ADRs and the phase reports, and resolved each one against the Makefile and `bench/run.py`. I diffed the
  parameter tables against `src/gpu.sv`'s parameter list.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | `python3 -m pytest xf/tests -q` | 697 passed, 56 skipped before this audit |
  | `python3 bench/run.py --check-citations` | `citations: OK` |
  | `python3 -m xf.asm kernels/relu.asm` | prints binary words |
  | `make test_matadd` | 1/1 PASS |
  | `make test_xf_matmul PARAMS=-Pgpu.THREADS_PER_BLOCK=8` | 6/6 PASS (SG22 verify) |
  | `make test_isa_ext` | 2/2 PASS |
  | `make test_warp_scheduler_gto` | 1/1 PASS |

  I also compiled by hand with `-PICACHE_LINES=16`, `-Pgpu.ICACHE_LINES=16`, `-Pgpu.SMEM_SIZE=32` and
  `-Pgpu.NOT_A_PARAM=3`. All builds went under `build/a11/`.

There are **no critical or high findings**, so this audit adds no SG rows.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| M1 | Medium | The Makefile's own usage line, `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`, builds the default GPU. iverilog ignores `-P` switches without a `gpu.` scope and exits 0, so the run looks like an I-cache/coalescing measurement but is not one | `make gpu_v BUILD=build/x && iverilog -o build/x/a.vvp -s gpu -g2012 -PICACHE_LINES=16 build/x/gpu.v; echo $?` → 0, and `grep '"ICACHE_LINES"' build/x/a.vvp` shows 0. A misspelt scoped name fails: `-Pgpu.NOT_A_PARAM=3` → `error: parameter ... not found`. `test_audit11.py::test_makefile_params_example_is_scoped` |
| L1 | Low | `kernels.matmul(..., mq=0)` (raw int32 accumulators) accepts an **i8** output segment. Later binders then pass `require_i8` on values far outside int8. The model and RTL compute `relu` → 20000 where `golden.relu` raises. This reopens the AUDIT-8 L1 dtype guard from the producer side | `test_audit11.py::test_unrequantized_matmul_rejects_i8_output`; details below |
| L2 | Low | The assembler accepts `BRp`, which is never taken because CMP never sets P (ADR-001). PLAN SG22 (DONE) says the assembler rejects `BRp`. `docs/isa.md` gives `BRzp` as an example spelling, and `BRzp` is just `BRz` | `python3 -c "from xf.asm import assemble; print(assemble('BRp 0'))"` → `[4608]`. `test_audit11.py::test_assembler_rejects_brp` |
| L3 | Low | Program length is not checked against the 256-word program memory (8-bit PC). The assembler and `kernels.program` accept 257+ words, and the runner's `Memory.load` drops the tail without error. Only `isasim.run` refuses, with `ValueError` | `assemble("NOP\n"*256+"RET")` returns 257 words. `test_audit11.py::test_assembler_rejects_programs_past_program_memory` |
| L4 | Low | A binder can fail after it has allocated `<out>.args`: a constant outside int32 raises `OverflowError: o.args: values outside i32` from `Image.write`. The image keeps the zero-filled args segment, so a corrected retry fails with `segment 'o.args' already allocated`. The error also does not name the argument (`M`, `Ma`, `one`) | `test_audit11.py::test_failed_bind_leaves_image_unchanged` |
| L5 | Low | Binder preconditions are looser than golden's, and their errors are inconsistent. `rmsnorm` accepts a 2-D `rsqrt_lut`, where golden says "must be a non-negative 1-D table". `swiglu` checks the LUT's shape but not its i8 dtype. `transpose` checks no dtype. A wrong-rank operand gives `ValueError: not enough values to unpack` (matmul, softmax, transpose). A missing segment gives a bare `KeyError` | `test_audit11.py::test_binders_reject_luts_golden_rejects` (2 cases) |
| L6 | Low | `isasim.run` against contract 9 ("errors raise `IsaSimError`"): its argument checks raise `ValueError`. It also accepts `threads` > 255, where the RTL's `thread_count` is 8 bits and every binder caps at 255. Program words wider than 16 bits run as NOPs | `test_audit11.py::test_isasim_argument_errors_are_isasimerror`, `::test_isasim_rejects_thread_count_past_8_bits` |
| L7 | Low | `Image.alloc` accepts a negative dimension: `alloc("n", [-4], "i8")` returns base 0 and adds no words, so the next segment also gets base 0. Contract 3's bump allocator assumes non-negative sizes | `test_audit11.py::test_image_rejects_negative_shape` |
| L8 | Low | Runner docs. `run_kernel` is annotated `-> dict`, and the module docstring says "Returns a dict of metrics", but it returns `(metrics, data_memory)`; every caller unpacks two values. The metrics dict also omits contract 12's `warp_issues_total` and `bar_stall_cycles_total`, so the tests that need them read the DUT directly | `grep -n "\-> dict\|Returns a dict" test/helpers/runner.py`; `grep -n "metrics, mem = await run_kernel" test/*.py` |
| L9 | Low | Doc and Makefile drift. CLAUDE.md's parameter table repeats `WARPS_PER_CORE`/`WARP_POLICY`/`SIMT_STACK` after the "Shared memory" paragraph, as a second headerless table. No parameter table (CLAUDE.md, ROADMAP) lists `DATA_MEM_ADDR_BITS`/`DATA_MEM_DATA_BITS`/`DATA_MEM_NUM_CHANNELS`, although `WORD_BITS`' row requires one of them; only architecture.md has them. `.PHONY` lists `bench`, which has no rule (`make bench` → "Nothing to be done"). ROADMAP says the matrix runs "the xf tests at the 8- and 32-bit widths", but only the regression runs at both | `sed -n 76,93p CLAUDE.md`; `make -n bench` |
| L10 | Low | Several documented `bench/run.py` commands overwrite files the docs call frozen. The CLAUDE.md phase-3 and phase-5 commands, the README phase-5 command and the phase-3/4/5/9 report commands rewrite `bench/phase-{3,4,5,9}-*.json`, which ROADMAP and the final report call "frozen per-phase history from before SG18". `--params` and positional tests are ignored without warning under `--phase 11` and `--xf` | `python3 bench/run.py --phase 5 --label phase-5-coalesce --params "ICACHE_LINES=16 COALESCE=1"` writes `bench/phase-5-coalesce.json` (code: `out = ROOT / "bench" / f"{label}.json"`). Not run, to keep the tree clean |

## M1 — Makefile usage line passes unscoped parameters (Medium)

`Makefile` line 6 reads ``# Pass parameters via DEFINES, e.g. `make test_matadd PARAMS="-PICACHE_LINES=16 -PCOALESCE=1"`.``
The variable is `PARAMS`, not `DEFINES`. iverilog only applies `-P<module>.<param>`; an unscoped `-PNAME=V` is
dropped silently. That build's `.vvp` has `ICACHE_LINES` = 0 in all three places where `-Pgpu.ICACHE_LINES=16`
gives 16, and iverilog exits 0. A misspelt *scoped* name does fail (`-Pgpu.SMEM_SIZE=32` → `parameter 'SMEM_SIZE'
not found in 'gpu'`), so the only silent failure is the exact form the Makefile recommends.

Everything else uses the scoped form: CLAUDE.md, the ADRs, the matrix `CFG_*` and `bench/run.py`'s `iv()`. A user
who follows the Makefile comment gets default-config cycle counts with no error.

Fix: change the comment to `-Pgpu.ICACHE_LINES=16 -Pgpu.COALESCE=1` and "via PARAMS". Optionally make `compile`/
`compile_xf` reject a `PARAMS` word matching `-P[^.]*=`.

## L1 — matmul without requant into an i8 segment

```python
img.alloc("a", [2,2], "i8", data=100s); img.alloc("b", [2,2], "i8", data=100s)
img.alloc("c", [2,2], "i8"); img.alloc("r", [2,2], "i8")
kernels.matmul(img, "naive", "a", "b", "c")   # mq=0: accepted
ew.relu(img, "c", "r")                         # require_i8("c") passes
# isasim (word_bits=32, addr_bits=12): c = r = [[20000, 20000], [20000, 20000]]; golden.relu(c) -> ValueError
```

SG22 added `require_i8` on binder inputs (AUDIT-8 L1). That guard trusts the segment's dtype, and `matmul` with
`mq = 0` writes int32 values into whatever segment it is given. `test_audit8.py::test_i32_outputs_still_accepted`
covers the allowed direction. Fix: require `dtype == "i32"` for `c` when `mq == 0`. The exported chains always
requant, so they are not affected.

## L2 — BRp

ADR-001 and `docs/isa.md` say CMP sets `N = rs ≠ rt`, `Z = rs = rt`, and never P (`alu.sv` line 33:
`(rs - rt < 0)` on unsigned values). So `BRp` is a branch that is never taken, like the bare `BR` that SG22 now
rejects. PLAN SG22 lists "(assembler rejects `BR` past the program and `BRp`)" and is marked DONE, but
`assemble("BRp 0") == [0x1200]`.

`docs/isa.md` §Assembler gives `BRzp` as an example spelling. It assembles, but it is only `BRz`. `BRnp` is `BRn`.
Also, the spelling must follow n, z, p order: `BRpz` fails with "unknown mnemonic 'BRPZ'" instead of a message
about the condition order.

Fix: reject a condition with only P set, or change PLAN SG22's text. Name only real conditions in the doc.

## L3 — Program length

The assembler checks branch targets against the program length, but never checks the length against
`PROGRAM_MEM_ADDR_BITS = 8`. `test/helpers/memory.py::Memory.write` ignores addresses past its size, so
`run_kernel`/`run_launches` load the first 256 words and drop the rest without error, and the PC wraps.
`isasim.run` raises `ValueError("data or program larger than its memory")`. A target in the dropped part is caught
anyway (`imm` > 255 fails encoding). A tail with no branch into it is not caught. Fix: `assemble` and
`run_kernel`/`run_launches` reject programs longer than 2^`program_addr_bits` words.

## L4 — Failed binds mutate the image

`_bind`/`matmul` call `img.alloc(f"{out}.args", ..., data=args)`. `Image.alloc` registers the segment and extends
`words` before `write` range-checks the data. A constant outside int32 therefore leaves a zero-filled `<out>.args`
behind. Example: `requant(M=2**31)`, `residual_add(Ma=2**33)` or `softmax(one=2**40)`. `golden` itself accepts
`requant(0, 2**31, 0)`; it only checks products, so the two layers fail at different points. Fix: check the
constants against int32 in the binder, before allocating, with an error that names the argument. Alternatively,
make `Image.alloc` validate `data` before it mutates the image.

## L5 — Binder vs golden preconditions

| Binder | Accepts | Golden |
|---|---|---|
| `rmsnorm` | `rsqrt_lut` of shape `[32, 32]` (length taken as `prod`) | `ValueError: rsqrt_lut must be a non-negative 1-D table` |
| `swiglu` | i32 `silu_lut` of 256 entries | `_i8(silu_lut)` → `ValueError: silu_lut is not int8` |
| `transpose` | any dtypes (i32 → i8 copies values unchanged) | n/a |
| `matmul`, `softmax`, `transpose` | 1-D operand → `ValueError: not enough values to unpack (expected 2, got 1)` | named errors |
| all | unknown segment name → `KeyError: 'o'` | n/a |

Every exported image uses the right shapes and dtypes, so this is API hygiene.

## L6 — isasim vs contract 9

Contract 9 says "errors raise `IsaSimError`". Each of these raises `ValueError` instead: `smem_size=48`,
`warps_per_core=3`, `warp_policy="gtlrr"` and a 257-word program. `run([RET], [], 300).steps == 75`: the model runs
75 blocks, while the RTL's `dcr` keeps `300 & 0xFF = 44` threads. A word ≥ 2^16 decodes to an opcode > 15, and the
model runs it as a NOP. `IsaSimError` is a `RuntimeError`, not a `ValueError`, so a caller cannot catch both with
one class. Fix: raise `IsaSimError` everywhere, or have it subclass `ValueError` and amend contract 9. Also reject
`threads > 255` and words ≥ 2^16.

## L8 — Runner documentation

`run_kernel(...) -> dict` and the module docstring ("Returns a dict of metrics") disagree with the code, which ends
`return await _run(...), data_memory`. Contract 5 describes `run_launches` correctly: it returns
`(metrics, data_memory)`, and each launch's metrics dict adds `launch` and `name` to `run_kernel`'s keys.
`warp_issues_total`/`bar_stall_cycles_total` (contract 12) are not in either metrics dict. `test_isa_diff`,
`test_divergence` and `test_bench_sweep` read them from `dut` themselves.

## Checked and consistent

- **Contract 1.** `OPCODES` covers all 16 opcodes with the `docs/isa.md` mnemonics, `FIELDS` matches the encoding
  diagrams, and `test_isa.py` checks the doc table. The `SYNC` encoding (`BR`, nzp `000`, bit 8 = 0) matches
  ADR-011 and contract 10. `disassemble`/`assemble` round-trip `SYNC`.
- **Disassembly.** `decode` is documented as the inverse for canonical words only. The words that do not round-trip
  are the expected ones: NOP/BAR/RET/CMP/STR/STS/LDR/LDS with don't-care bits set, BR with bit 8 set, and
  R13–R15 destinations, which the assembler rejects.
- **Contract 2.** Labels, `%blockIdx/%blockDim/%threadIdx` aliases, `#imm` 0–255 (`#-1` and `#256` rejected),
  writes to R13–R15 rejected for every `rd` form (ALU, LDR, LDS, CONST), bare `BR` rejected, targets past the
  program rejected, `python3 -m xf.asm file.asm` works. The assembler also accepts immediates without `#`, which is
  undocumented but harmless.
- **Contract 9 and ADR-011/015.** `isasim.run` has the contract's keyword names and defaults, and returns
  `Result(memory, steps, divergent, stores_conflict)`. It matches `docs/isa.md` on:
  - CMP as N = ≠ and Z = equal;
  - SRA filling with the sign when `rt ≥ WORD_BITS`;
  - signed MAX;
  - DIV by 0 giving 0;
  - LDR/STR masked to `DATA_MEM_ADDR_BITS`, and LDS/STS masked to the low 8 bits and then to `SMEM_SIZE`.

  It matches the ADRs on: the stack push order (taken side on top), the pop loop, RET removing lanes from every
  entry, an overflow at more than `stack_depth` pushed entries, and BAR at W=1 adding a step and no stall.
- **Contracts 3, 4, 7.** The `Image` signature, the JSON form and `from_dict` contiguity, the golden function list
  and semantics, and the fixture keys all match the code.
- **Contract 5.** `Launch(program, threads, name="", timeout_cycles=20000, max_block=None)` defaults `max_block` to
  `program.max_block`. `run_launches` checks the bus width and `check_block` before any launch runs. The SG22
  verify command passes.
- **Contract 6.** `test_xf_%` uses `XF_PARAMS` followed by `PARAMS`, into `$(BUILD)/sim_xf.vvp`.
- **Contract 12.** `check_counters` implements contract 12's counter rules.
- **`gpu` parameters.** Every parameter in a doc table exists in `src/gpu.sv` with the stated default. `SMEM_SIZE` is
  correctly documented as `core`-only (AUDIT-4 L2 fixed). `gpu.sv` `$fatal`s on `WORD_BITS ≠ DATA_MEM_DATA_BITS`, on
  `WARPS_PER_CORE ∤ THREADS_PER_BLOCK` and on `WARP_POLICY ∉ {0,1,2}`, as documented.
- **Documented targets.** Every documented `make` target exists, and the ones run here pass. The Makefile's
  `MATRIX` has the 13 configurations that README and ROADMAP list, and `CFG_all` matches ROADMAP's "all" row.
- **Bench CLI.** `--check-citations` resolves every citation in `docs/final-report.md`, including the brace
  expansions, and fails on a doc with none. `--xf` without `--label xf-*` errors as documented.
