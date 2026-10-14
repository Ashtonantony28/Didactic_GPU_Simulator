# AUDIT-4 — ISA and kernel consistency (round 1)

Lens: do `docs/isa.md`, the ADRs, the ISA table (`xf/isa.py`), the assembler (`xf/asm.py`), the RTL
decoder/ALU/PC/LSU/scheduler at both `WORD_BITS` widths and every `kernels/*.asm` agree on encodings,
semantics, signedness, branch behaviour, shared memory and the args-block convention? Audited at
`c58ff7c`, iverilog 12.0, cocotb 2.0.1.

Method: besides reading every source, an independent instruction-level model written from
`docs/isa.md` alone (lockstep blocks, per-thread NZP, 32-bit wrap, SRA/MAX signed, DIV/CMP unsigned)
ran every xf kernel through the real binders. It agrees bit for bit with the golden chain on the MLP
block and attention head (both matmul kinds, 40 random inputs plus all-zero/−128/127) and with
`golden.matmul_i8`/`requant` on 8 matmul shapes × 4 (mq, s) pairs, including M, K not multiples of 4.
The model also records every branch where the active threads of a block disagree: there are none.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| M1 | Medium | Branches resolve to thread 0's target; no ISA document says so, and block-uniform branching is an unchecked kernel obligation | below |
| M2 | Medium | `matmul_tiled` only works with 4-thread blocks and hangs otherwise; nothing checks `THREADS_PER_BLOCK` | below |
| L1 | Low | `docs/isa.md` draws a 9-bit branch immediate; the RTL, ISA table and tests use 8 bits | below |
| L2 | Low | Stale ISA docs: ADR-006 reserves opcode `1101` (now SRA), ADR-007 and `kernels/tiled_matmul.asm` predate the wiring, CLAUDE.md lists a `gpu.SMEM_SIZE` that does not exist | below |
| L3 | Low | The assembler accepts branches that can never be taken or that leave the program | `xf/tests/test_audit4.py` (2 strict xfails) |
| L4 | Low | Kernel loops are do-while; a zero `K`, `d` or `T` never terminates and the binders accept it | below |

No critical or high findings, so this audit adds no SG rows.

Checked and consistent:
- **Encodings.** `xf/isa.py` covers all 16 opcodes once; `decode(encode(w)) == w` for every canonical
  word (`test_isa.py`); `decoder.sv` field slices (`rd [11:8]`, `rs [7:4]`, `rt [3:0]`, `nzp [11:9]`,
  `imm [7:0]`) and opcode constants match the table, and the opcode table in `docs/isa.md` is checked
  against it. Every `kernels/*.asm` and every legacy program assembles; no BR targets past its program.
- **SRA/MAX/CMP at both widths.** `alu.sv` implements `$signed(rs) >>> rt` with `rt` unsigned over the
  word (≥ W fills the sign) and signed MAX, exactly as `docs/isa.md` and ADR-010 state. CMP sets
  N = "differs", Z = equal, P never at 8 and 32 bits (the relational operands widen to 32 bits, so the
  unsigned compare is the same at both). `make test_isa_ext` and `make test_xf_isa_ext` pass.
- **Signedness in kernels.** SRA and MAX are the only signed ops used where sign matters; every DIV
  has non-negative operands (`i / N`, `N / blockDim`, `blockIdx / tiles`, `i / d`, `i / T`,
  `(e·one + S/2) / S`); every CMP is an equality test (`BRn` = differs, `BRz` = equal; no `BRp`).
  CONST's zero-extension is respected (−128 is built as `0 − 128`). The clamp `−max(−max(x, −128), −127)`
  is safe at INT32_MIN because the first MAX runs before any negation.
- **Args blocks.** Each kernel's header, its loads (word offsets) and its binder's list agree for all
  nine kernels; words 0 and 4 are the ARGS prologue CONSTs and `kernels.program` checks both.
  `rnd = 2^(s−1)` (0 at s = 0) matches `golden.requant`; rmsnorm's `lutmax = len − 1` matches
  `rmsnorm_int`.
- **SMEM and BAR.** `matmul_tiled` uses SMEM[0:24] (≤ the fixed 64 words); every LDS reads a word
  written by the same block after the previous BAR, and the post-loop LDS of SMEM[16+t], SMEM[20+t] and
  SMEM[4i+t] read the thread's own writes. BAR is a decoded no-op (`scheduler.sv` ignores
  `decoded_bar`), as `docs/isa.md` says; under lockstep the kernel does not depend on it.
- **Thread and address limits.** Binders reject > 255 threads (8-bit DCR); `%blockIdx` ≤ 63 blocks at
  4 threads; programs are ≤ 147 words (8-bit PC); images are bounded by `addr_bits = 12`.
- **ISA changes documented.** The only RTL changes since SG1 are SG2's `WORD_BITS` datapath, SRA and
  MAX, all in `docs/isa.md` and ADR-010 (`git diff fb8df68 c58ff7c -- src`).

## M1 — Branches resolve to thread 0's target, undocumented (Medium)

`scheduler.sv` advances the block with `current_pc <= next_pc[0]`: every thread computes its own
branch outcome from its own NZP, and thread 0's wins. A thread whose condition disagrees silently
follows thread 0. `docs/isa.md` defines `BRnzp` per thread ("if `(NZP & nzp) != 0` then `PC ← imm`"),
ADR-006 describes only the planned SIMT stack, and ADR-010 does not mention it. The one place the
rule is written down is a comment in `test/test_isa_ext.py`. Correctness of every xf kernel therefore
rests on an unstated contract that their branches are block-uniform; each kernel asserts it in a
comment and nothing checks it. All nine comply today (model above: zero divergent branches).

Repro (8-bit build): the program below makes threads 1–3 compare "differs" and thread 0 "equal";
all four take thread 0's fall-through path and store 1.

```
    CONST R1, #0
    CMP %threadIdx, R1
    BRn taken
    CONST R2, #1
    STR %threadIdx, R2
    RET
taken:
    CONST R2, #2
    STR %threadIdx, R2
    RET
```

`run_kernel(dut, assemble(PROG), [0]*8, 8)` leaves memory `[1, 1, 1, 1, …]` (expected per
`docs/isa.md`: `[1, 2, 2, 2]`).

Fix direction: state the rule in `docs/isa.md` (branch target and NZP of thread 0 decide for the
block; branch conditions must be block-uniform) with a pointer to ADR-006, and consider a debug check
in the runner or a simulation assertion that flags disagreeing `next_pc` lanes.

## M2 — `matmul_tiled` needs 4-thread blocks and hangs otherwise (Medium)

The kernel reads `%blockDim` for its strides but hard-codes a 4-row tile: four accumulators, SMEM
slots 16+t and 20+t, and the inner-loop step `kk + 3·blockDim − 11`, which returns to `kk + 1` only
when `blockDim = 4`. With any other block size the exit test `kk + 1 == n` is never met and the kernel
spins. `xf/kernels.py` documents `TILE = 4` but cannot see the DUT's `THREADS_PER_BLOCK`, a documented
`gpu` parameter (CLAUDE.md), and `run_launches` does not check it either. The naive matmul and all
elementwise kernels are block-size generic (model: bit-exact at 2 and 8 threads per block).

Repro: `make test_xf_matmul PARAMS="-Pgpu.THREADS_PER_BLOCK=8"` — all 5 tests fail with
`TimeoutError: Kernel did not complete in 200000 cycles`. In the ISA model, `mlp.launches(img, "tiled")`
at 2 or 8 threads per block never terminates; `"naive"` is bit-exact.

Fix direction: either make the tile height a function of `%blockDim` (loop over rows instead of four
named accumulators, SMEM layout from `blockDim`), or have the binder/runner refuse `matmul_tiled` when
the DUT's block size is not 4.

## L1 — Branch immediate drawn 9 bits wide (Low)

`docs/isa.md` shows the BRnzp form as `|0001|nzp [11:9]|immediate [8:0]|`. The decoder takes
`instruction[7:0]`, the PC is 8 bits, `FIELDS["imm"] = (0, 8)`, and `test_br_layout` pins bit 8 to 0.
Repro: compare the diagram with `src/decoder.sv` (`decoded_immediate = instruction[7:0]`). Fix: draw bit
8 as unused.

## L2 — Stale ISA documentation (Low)

- ADR-006 "Consequences" reserves opcode `1101` for a future `RC` (reconverge); ADR-010 assigned
  `1101` to SRA and there are no free opcodes left. `grep -n 1101 docs/decisions/ADR-006-divergence.md`.
- ADR-007 says SMEM is "64 entries × 8 bits … addressable by a 6-bit immediate"; it is `WORD_BITS`
  wide and register-addressed (`rs[7:0]`, masked to 6 bits), and it points to `kernels/tiled_matmul.asm`
  as the kernel.
- `kernels/tiled_matmul.asm` is still the 3-instruction stub saying LDS/STS/BAR are "currently unwired
  in src/"; the real kernels are `matmul_tiled.asm` and the hand-encoded one in `test_tiled_matmul.py`.
- `docs/architecture.md` describes data memory as 8-bit address, 8-bit data without the xf widths.
- CLAUDE.md's parameter table lists `SMEM_SIZE` as a `-Pgpu.` switch; `gpu` has no such parameter
  (`make test_xf_matmul PARAMS="-Pgpu.SMEM_SIZE=16"` → `error: parameter 'SMEM_SIZE' not found in 'gpu'`).
  It exists only on `core`, fixed at 64.

## L3 — Assembler accepts dead and out-of-program branches (Low)

`BR target` with no condition letters assembles to `nzp = 000`, a branch that is never taken, without
a warning (`assemble("top: BR top")` → `0x1000`). A numeric target past the end of the program is also
accepted (`assemble("BRnzp 200\nRET")`), and execution would run through zeroed program memory (NOPs)
and wrap. `BRp` is legal but can never be taken either, since CMP never sets P (ADR-001). No kernel or
legacy program does any of this today.

Repro: `python3 -m pytest xf/tests/test_audit4.py -rxX` (both strict xfail). Fix: reject `BR` with
no flags and targets ≥ program length (and optionally `BRp`/`BRzp` P-only conditions).

## L4 — Zero-size loops never terminate (Low)

`matmul_naive`, `matmul_tiled`, `rmsnorm` and `softmax` count their loop down and test `!= 0` after
the body, so `K = 0`, `d = 0` or `T = 0` wraps the counter and runs ~2^32 iterations. The binders check
shapes but not that these dimensions are positive. Unreachable from the exported images; a one-line
`> 0` check in each binder closes it.
