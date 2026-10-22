# AUDIT-9 — Encoding soundness (spec, model, assembler, RTL)

Lens: do `docs/isa.md`, ADR-011 (SYNC/SIMT semantics), ADR-012/014 (multi-warp), ADR-015 (SIMT RTL), the reference
model (`xf/isasim.py`), the assembler (`xf/asm.py`, `xf/isa.py`) and the RTL (`src/`) agree? And does the checking
have false negatives, meaning bugs that the differential tests do not catch? Audited at `aa27256` (main), with
iverilog 12.0, sv2v, cocotb 2.0.1 and torch.

Method:
- **Seeded mutants.** I wrote 42 one-line RTL mutants: 16 in the SIMT stack (`S*`, `src/simt_stack.sv`), 8 in the
  core/scheduler lane masking (`C*`) and 18 in the decoder or next to it (`D*`). Each mutant was built in its own
  copy under `build/mut/<id>/`, so the worktree `src/` was never touched (`git diff --stat src` is empty). The
  repo's own checks then ran against each mutant: `make test_isa_diff`, `make test_xf_isa_diff`, `make test_multiwarp`
  (w2_lrr, w4_lrr, w4_gto, w4_gtlrr) and `make test_divergence` (all 9 configurations plus both `simt_fatal` runs).
  The runs stopped at the first failure, at most 2 simulations at a time.
- **Targeted RTL-vs-model probes** (scratch, not committed). These cover:
  - hand-encoded writes to R13–R15
  - a branch past the end of the program, so the PC wraps through NOPs to 0
  - SMEM addresses ≥ 64 and 255
  - BR, CMP and STR words with bits set in unused fields
  - taken-first ordering
  - a taken target equal to the reconvergence PC R (double pop)
  - a stale SYNC
  - RET by every taken lane
  - a backward R
  - BAR inside a divergent region
  - RET of a whole warp before a BAR

  Each probe ran at stack off with W=1 and W=2, and with `SIMT_STACK=1` at W=1, 2 and 4 (LRR), W=4 (GTO), and
  `THREADS_PER_BLOCK=8` W=2, with 3 thread counts each. **175/175 RTL runs match `xf.isasim`** (memory and
  `warp_issues_total`). Both `$fatal` paths fire at HEAD on the new programs below.
- **Assembler.** `disassemble → assemble` ran over all 65,536 words, and opcode/field docs were checked against
  `xf/isa.py`. Every canonical word round-trips. The 6,912 words that write R13–R15 disassemble but are rejected on
  reassembly (by design). The 32,093 non-canonical words lose their unused bits, and the RTL and the model ignore
  those bits too.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| M1 | Medium | No `SIMT_STACK=1` RTL test executes `BAR`. Mutant C08 (BAR dropped whenever the stack is on) survives every target, `matrix_simt` included (W=1, where BAR is a no-op) | `xf/tests/test_audit9.py::test_simt_rtl_tests_execute_bar`; C08 below |
| M2 | Medium | The "divergent branch without SYNC" `$fatal` is only tested on a program with no SYNC at all. Mutant S02 (SYNC never consumed) survives: such an RTL silently reconverges at a stale PC where the model raises | `xf/tests/test_audit9.py::test_nosync_fatal_covers_a_consumed_sync`; S02 below |
| L1 | Low | Stack-overflow `$fatal` is tested at 8 (passes) and 10 entries (fatal), never at 9. Mutant S03 (`> DEPTH + 1`) survives | `xf/tests/test_audit9.py::test_overflow_fatal_covers_depth_plus_one`; S03 below |
| L2 | Low | R13–R15 read-only is checked only in the model. Mutant D11 (`rd < 16`) survives every RTL target, because the assembler never emits such writes | D11 below |
| L3 | Low | The pop rule `pc == rpc` cannot be told apart from `pc >= rpc` by any test. Mutant S04 survives. It differs only when R is behind the branch or a lane jumps past R | S04 below |
| L4 | Low | No RTL test executes `NOP`. Mutants D17 (NOP decodes as RET) and D12 (NOP decodes as BAR) survive | D17/D12 below |
| L5 | Low | `xf.isasim.run` accepts `threads > 255`. The RTL DCR is 8 bits, so such a launch cannot run (binders already check this) | `xf/tests/test_audit9.py::test_isasim_rejects_more_than_255_threads` |
| L6 | Low | Docs gaps: `isa.md` does not say that BAR in a divergent region flags the whole warp (ADR-015 only). The opcode table says "the block follows thread 0", but at W>1 it is each warp's lowest lane. Nothing states that a SYNC left pending by a uniform branch is consumed by a later, unrelated divergent branch | below |

**There are no critical or high findings.** The RTL, the model, the assembler and the docs agree on every opcode
and encoding I probed. The findings are false negatives in the checking.

Mutant kill summary: **34 killed, 8 survived**. Of the survivors, 1 is equivalent (S16), and the other 7 are
M1, M2, L1, L2, L3 and L4.

| Group | Killed | Survived |
|---|---|---|
| SIMT stack S01–S16 | 12 (S05–S14 by `divergence_w1`, S15 by `simt_fatal_nosync`, S01 by `simt_fatal_overflow` only) | S02, S03, S04, S16 (equivalent) |
| Core/scheduler masking C01–C08 | 7 (C01–C03 by `divergence_w1`, C04–C07 by `divergence_t8w2`) | C08 |
| Decoder D01–D18 | 15 (13 by `test_isa_diff`, D01 by a 1500 s hang in it, D10 by `test_xf_isa_diff`, D14 by `divergence_w1`) | D11, D12, D17 |

Mutant definitions are one-line edits. Apply one to a copy of `src/`, then run the four targets above:

| ID | File | Edit |
|---|---|---|
| S01 | simt_stack.sv | push the not-taken entry on top (swap `sp+1` / `sp+2`) |
| S02 | simt_stack.sv | delete `n_sync_valid = 1'b0;` in the divergent branch |
| S03 | simt_stack.sv | `sp + 2 > DEPTH)` → `sp + 2 > DEPTH + 1)` |
| S04 | simt_stack.sv | `n_pc[k] == n_rpc[k]` → `n_pc[k] >= n_rpc[k]` |
| S05–S15 | simt_stack.sv | `t = taken` unmasked; mask without `exists`; single pop; `next_done` from top; parent pc not set to R; SYNC not latched; diverge on `t == 0`; RET clears top only; rpc = imm; sync_pc + 1; no-SYNC `$fatal` removed |
| S16 | simt_stack.sv | RET `k <= sp` → `k < sp` (equivalent: the top always empties on RET, ADR-015) |
| C01–C07 | core.sv / scheduler.sv | LSU or regs ungated by `lane_on`; `exists` `<=`; LSU ungated at W>1; all stacks update; `taken` lane index; W>1 PC from lane 0 |
| C08 | scheduler.sv | `if (decoded_bar)` → `if (decoded_bar && !SIMT_STACK)` |
| D01–D18 | decoder.sv (D10/D11 registers.sv, D14 core.sv) | opcode bits `{15:13,11}`; rs/rt swapped; NZP reversed, n/z swapped, shifted; CMP/STR write rd; SRA→MAX; CONST via ALU; CONST sign-extended; `rd < 16`; NOP = BAR; LDS also reads global; SYNC ignored for odd imm; imm bit 7 dropped; rd bit 3 dropped; NOP = RET; DIV→MUL |

## M1 — BAR is never exercised with the SIMT stack on (Medium)

`test_divergence` (the only `SIMT_STACK=1` target) contains no BAR. Neither do its fixed kernels or the
`Structured` generator. `test_multiwarp` runs with the stack off, and the Makefile comment says it "branches
divergently without SYNC". `matrix_simt` is W=1, where BAR is a no-op, so the xf kernels' BARs prove nothing there.
Mutant C08 drops `warp_bar` whenever `SIMT_STACK=1`, which breaks every barrier of a multi-warp SIMT build, and it
passes all of `test_divergence`, `test_multiwarp`, `test_isa_diff` and `test_xf_isa_diff`.

`test_multiwarp` does in fact pass at HEAD with the stack on at 1 lane per warp. There, no warp can diverge:

```
make BUILD=build/mw4s PARAMS="-Pgpu.SIMT_STACK=1 -Pgpu.WARPS_PER_CORE=4" compile
make BUILD=build/mw4s multiwarp_run        # HEAD: 6/6 pass; C08: test_barrier, test_partial_block FAIL
```

Fix: add that configuration to `test_divergence` (or `DV_CFGS`), and add a SYNC-guarded BAR kernel at W=2.

## M2 — consumed-SYNC path of the no-SYNC `$fatal` is untested (Medium)

`simt_fatal nosync` runs M1 with no SYNC at all, so `sync_valid` is never set. The real-bug shape is a valid bit that
is not cleared when a divergent branch consumes it (S02). That shape survives every target. On such an RTL, a second
divergent branch with no SYNC of its own silently reconverges at the first region's PC, corrupting results where
the model raises `IsaSimError`. The differential tests only compare programs that the model accepts, so this class of
bug is invisible to them. The following program fatals at HEAD (`divergent branch at pc 7 without SYNC`) and runs to
completion under S02 (W=1, `SIMT_STACK=1`):

```
    CONST R0, #0
    CONST R1, #5
    SYNC j1
    CMP %threadIdx, R0
    BRn j1
    CONST R1, #7
j1: CMP %threadIdx, R0
    BRz j2              ; divergent, no SYNC of its own
    CONST R1, #9
j2: STR %threadIdx, R1
    RET
```

Fix: add it to `test/simt_fatal.py` PROGRAMS and to `test_divergence`'s fatal runs.

## L1 — overflow `$fatal` boundary untested (Low)

`CAPPED_LOOP` only reaches even depths: cap 4 gives 8 entries (must pass) and cap 5 gives 10 (must fatal).
`sp + 2 > DEPTH + 1` therefore survives, and storage (`DEPTH + 3`) holds the 9th entry silently. The program below
needs exactly 9 entries at 8 lanes (taken lane 0 reconverges at once, then 4 divergent exits in the else-arm). HEAD
fatals (`deeper than 8 at pc 16`) and S03 completes:

```
    CONST R0, #0
    CONST R1, #1
    CONST R3, #0
    SYNC end
    CMP %threadIdx, R0
    BRz end
    CONST R4, #5
    SUB R4, R0, R4
    SUB R2, R0, %threadIdx
    MAX R2, R2, R4
    SUB R2, R0, R2
    ADD R2, R2, R1
loop:
    SYNC end
    ADD R3, R3, R2
    SUB R2, R2, R1
    CMP R2, R0
    BRn loop
end:
    STR %threadIdx, R3
    RET
```

## L2 — R13–R15 read-only not checked in RTL (Low)

`xf/tests/test_isasim.py::test_readonly_registers` covers the model only. The assembler rejects `rd ∈ {13, 14, 15}`,
so no fixture or kernel ever writes them, and D11 (`decoded_rd_address < 16`) survives. A hand-encoded probe of ADD,
CONST, LDR and LDS into R13–R15 matches the model at HEAD (the writes are ignored) and catches D11. Such words can
only reach the RTL through hand-encoding or a bad binder, hence Low.

## L3 — pop rule `==` vs `>=` (Low)

The structured programs always reach R by moving forward, so the S04 mutant (`>=`) is equivalent on them. A loop
whose SYNC points back to its own head shows the difference: with `SYNC top`, a divergent `BRn top` and then
`BRnzp top`, HEAD matches the model and S04 does not, at 3, 4 and 8 threads. ADR-011 specifies `==`. Neither the
kernels nor the generator use a backward R.

## L4 — NOP never executed (Low)

No RTL target executes opcode `0000`. Kernels never use it, and every test program ends in RET before the zeroed
memory. D17 (NOP = RET) and D12 (NOP = BAR, which is benign at lockstep timing) survive. A run that branches past
the end of the program, so the PC wraps from 255 to 0 through NOPs, matches the model at HEAD in all 7
configurations and catches D17.

## L5 — model accepts unrunnable thread counts (Low)

`dcr.sv` holds an 8-bit thread count, but `xf.isasim.run(prog, [], 256)` runs 64 blocks while the RTL would run 0.
`xf/kernels.py` already rejects `threads > 255`, but `run` (the oracle used directly by the cocotb tests) does not.

## L6 — docs (Low)

- `docs/isa.md` (BAR row and "Branches and SYNC") never says what BAR does inside a divergent region. ADR-015 says
  it flags the warp, as the model does. The model shows that a BAR in only one arm does not order that arm against
  the other arm's lanes.
- The opcode table row for BRnzp says "the block follows thread 0". The prose below it is accurate: the warp's
  lowest lane (lane `w·LANES` at W>1).
- A SYNC that a uniform branch leaves pending is consumed by the warp's next divergent branch, wherever that branch
  is (the probe `stale_sync` shows the RTL and the model agree). That is consistent with ADR-011's wording, but it is
  a trap: it also hides the no-SYNC `$fatal`. One sentence in `isa.md` would cover it.

Observation (no row): taken-first order (ADR-011) is pinned only indirectly. S01, which runs the not-taken side first,
leaves memory and step counts unchanged on every test and is killed only by the depth accounting of
`simt_fatal_overflow`. A probe where the taken lanes STS and the not-taken lanes LDS the same SMEM word pins it
directly (it matches at HEAD).

Checked and consistent: the opcode table ↔ `xf/isa.py` ↔ `decoder.sv`; BR layout (nzp in [11:9], bit 8 ignored by
both); CMP NZP `{rs≠rt, rs==rt, 0}`; CONST zero-extension at 32 bits (D10 is killed by `test_xf_isa_diff`); SMEM
masked to 6 bits after the low 8; PC wraps at 8 bits with zeroed program memory; RET with the stack removes the
active lanes from every entry, and the emptied entries are a suffix (ADR-015 argument verified: S16 is equivalent);
lanes without a thread are excluded from masks; W=1/2/4 × SIMT on BAR and RET probes match the model.
