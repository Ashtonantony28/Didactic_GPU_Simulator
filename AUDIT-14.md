# AUDIT-14 — Encoding soundness, round 4 (differential fuzz)

Lens: do `docs/isa.md`, the reference model (`xf/isasim.py`), the assembler (`xf/asm.py`, `xf/isa.py`) and the RTL
agree, including SG24's STS-conflict rule (highest lane of one `STS` wins; across warps the last `STS` to issue
wins) and its `sts_conflict{8,32}` fixtures? I also looked for false negatives, meaning disagreements that the
existing differential gates (`test_isa_diff`, `test_xf_isa_diff`, `test_divergence`) cannot see. I did not re-report
anything already in AUDIT-9 to AUDIT-12.

Audited at `4444c25` (main) with iverilog 12.0, sv2v, cocotb 2.0.1 and torch. `src/`, the tests, `xf/`, the Makefile
and the docs are unchanged. All builds and scratch tooling went under `build/a14/`, with at most 2 simulations at a
time. One container restart cut campaign 2 short; it was rerun in full.

## Method

- **Random differential fuzz.** A scratch generator (`build/a14/fz/a14gen.py`) emits valid programs of up to 256
  words. They contain:
  - straight-line ALU code over edge and random words;
  - nested if/else with `SYNC` (and, with probability 0.04, an if with no `SYNC`);
  - bounded loops with per-lane trip counts, including backward `SYNC` targets (`SYNC top`, AUDIT-9 L3);
  - `RET` inside divergent regions and `BAR` anywhere, divergent regions included;
  - `LDS`/`STS` with every lane on one word, lane pairs, `threadIdx`-derived, wrapping and data-dependent
    addresses, and addresses ≥ 64 (masked);
  - hand-encoded writes to R13–R15 (ADD/MAX/CONST/LDR/LDS), which the assembler rejects;
  - don't-care bits set in NOP, BAR, CMP, STS, STR and BR (bit 8);
  - `BRp`, and NOP padding so that code reaches PC 255.

  Global loads read a read-only input region or the thread's own slots. Global stores go only to the thread's own
  slots, and each block dumps SMEM to its own 32 words. No result depends on blocks racing, which the model scopes out.
- **Oracle.** At W = 1 the oracle is `xf.isasim.run` itself. At W > 1 it is a trace-driven copy of `xf.isasim.run`
  that differs only in its warp pick: the source is patched with one textual replacement and asserted to match. A
  cocotb monitor records each core's issue order (`scheduler.issue`, `selected_warp`, `block_id`), and the model
  replays exactly that order. Each pick must name a warp that is neither done nor held at a `BAR`. Memory must be
  bit-identical, `warp_issues_total` must equal the model's steps, and `runner.check_counters` must hold. This checks
  "the last STS to issue wins" exactly, on any interleaving the RTL picks, including the interleaving-dependent
  programs that `test_isa_diff` skips. The replay reproduces `xf.isasim`'s own lrr result on every fixture (control
  in `xf/tests/test_audit14.py`).
- **Configurations** (`gpu` parameters per build). The full cross covers:
  - `WORD_BITS` 8 and 32;
  - `WARPS_PER_CORE` 1, 2 and 4 under LRR, GTO and GTLRR;
  - `SIMT_STACK` 0 and 1;
  - `THREADS_PER_BLOCK` 4, 8 and 16. 16 is `Launch.max_block` for `matmul_tiled`.

  On top of that:
  - `THREADS_PER_BLOCK` 1, 2, 6 and 12, the last being a legal `matmul_tiled` blockDim, including W = 3 and W = 6;
  - the W = 1 matrix configurations pipelined, icache, coalesce, dcache, medic, all, cores1, cores4 and latency (3),
    at both widths;
  - multi-warp at `MEM_LATENCY` 3 and 10;
  - launches of 100–255 threads at 32 bits.
- **Model-rejected programs.** These are divergent with no `SYNC`, divergent after the `SYNC` was consumed, need 9 or
  10 stack entries, or diverge in a later block only. Each ran alone on the RTL, which must end in the matching
  `simt_stack` `$fatal`.
- **Directed probes:**
  - `STR` lane conflicts, with a read-back through L1-D;
  - `STS` with no `BAR`, and `STS` after a yielding `LDR`;
  - cross-warp `STR` races with varied skew at latency 0, 3 and 10;
  - `threads = 0`.
- **Sensitivity.** The fuzzer and probes ran against seven mutants that earlier audits left alive or recorded:
  AUDIT-12 S02 (lowest STS lane wins, named SM1 here), and AUDIT-9 S02, S03, S04, D11, D17 and C08.

## Commands actually run

| Command | Result |
|---|---|
| `python3 build/a14/fz/drive.py <cfg> 30 1` × 110 configs (campaign 1) | 3298 match, 2 model rejects (not run), 0 disagreements |
| `python3 build/a14/fz/drive.py <cfg> 12 4` × 77 configs (campaign 2: TPB 1/2/6/12, W 3/6, padding, latency) | 919 match, 5 model rejects, 0 disagreements |
| `python3 build/a14/fz/drive.py <cfg> 6 3` × 4 configs (campaign 3: 100–255 threads, 32-bit) | 24 match |
| reject probes × 5 SIMT configs | 60/60 agree (same `$fatal` kind, or both accept) |
| directed probes × 10 configs; STR read-back × 7 (incl. dcache, medic, all) | only `STR` conflicts disagree (H1) |
| 7 mutants (fuzz plus probes) | 7/7 killed |
| `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` | 769 passed, 24 xfailed (3 new xfails, 5 new passing tests) |
| `python3 bench/run.py --check-citations` | `citations: OK` |

## Findings

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | `STR` lane conflict: `xf.isasim` stores the highest lane. Its docstring says "the last lane wins", and `test_isasim::test_store_conflict` pins it. The RTL stores lane 0 at 4 lanes; in general it stores the lowest lane of the last group of `DATA_MEM_NUM_CHANNELS` lanes (lane 4 of 8, lane 12 of 16). The conflict is silent in the RTL, in the default (matrix) build and every other configuration probed. `docs/isa.md` states no rule. The fixtures skip `stores_conflict` programs, so no gate compares them | `test_audit14.py::test_store_conflict_model_matches_rtl`; RTL command below |
| L1 | Low | The interleaving-independence gate (`model lrr == model gto`, in `test_isa_diff` and `test_divergence.check`) is not sound. It admits `sts_conflict{8,32}`, whose W > 1 result depends on which warp issues its `STS` last, although `xf/isa_fixtures.py` says the `BAR` makes it independent and `isa.md` says the fixtures pin "highest lane wins". Fuzz found 12 valid programs where lrr == gto but the RTL's legal order, which the trace replay confirms, gives other memory. One of them is in the `warps2` 32-bit matrix configuration. As fixtures, these would fail a correct RTL | `test_audit14.py::test_sts_conflict_fixture_is_interleaving_independent` |
| L2 | Low | `docs/isa.md`'s RET row says "finish current block". At `WARPS_PER_CORE > 1` without the stack, RET ends only the issuing warp (model and RTL agree) | `test_audit14.py::test_isa_md_ret_row_names_the_warp` |

**There are no critical findings.** H1 is the only model/RTL disagreement found. On every program the model accepts
without a store conflict (4241 random programs over 191 configurations, plus the directed probes), the RTL matched
the model bit for bit, issue order included. That covers R13–R15 writes, don't-care bits, `BRp`, NOPs, PC 255,
backward `SYNC`, `RET` and `BAR` in divergent regions, and SMEM conflicts within and across warps. The new
STS-conflict rule holds as stated.

### H1 — STR lane conflict: model and RTL pick different lanes (High)

The program, at `THREADS_PER_BLOCK = 4`, one block:

```
    CONST R0, #100
    CONST R1, #1
    ADD R2, %threadIdx, R1
    STR R0, R2              ; every lane: MEM[100] = t + 1
    RET
```

Observed: RTL `MEM[100] = 1`. Expected per `xf.isasim`: `4`, with `stores_conflict = [(0, 3, 100)]`. The same split
held in each configuration probed:
- `THREADS_PER_BLOCK` 3, 4, 8 and 16, where the RTL gives lane 0, 0, 4 and 12 and the model the last lane;
- W = 1, 2 (LRR, GTO) and 4 (GTO, stack on);
- `MEM_LATENCY` 0, 3 and 10;
- `COALESCE=1`, `DCACHE_LINES=16`, `MEDIC=1` and the `all` set, at 8 and 32 bits.

A read-back `LDR` after the conflicting store returns the memory's value, so the L1-D stays coherent with memory.

Cause: the lanes' writes reach the data channels in the same cycle, and the same-cycle winner is whatever the
external memory does. `test/helpers/memory.py` unpacks the packed bus from channel N−1 down to 0, so channel 0
(lane 0 of the group) is applied last. The model hard-codes the opposite.

The model flags the conflict, but `run()` still returns a memory image that claims a winner, and the docstring says it
is "bit for bit with the RTL". `test_divergence`, `test_isasim` and the xf binders' callers never look at
`stores_conflict`. Fix: state in `docs/isa.md` (with an ADR, as SG24 did for STS) that an `STR` conflict is undefined,
or that the lowest lane of the last channel group wins. Then make the model raise or follow that rule, and update
`test_store_conflict`.

RTL repro (default build, nothing added under `test/`):

```
mkdir -p build/a14/repro && cat > build/a14/repro/a14_h1.py <<'EOF'
import cocotb
from xf.asm import assemble
from xf.isasim import run
from test.helpers.runner import run_kernel
SRC = "CONST R0, #100\nCONST R1, #1\nADD R2, %threadIdx, R1\nSTR R0, R2\nRET"
@cocotb.test()
async def h1(dut):
    tpb = int(dut.THREADS_PER_BLOCK.value)
    _, mem = await run_kernel(dut, assemble(SRC), [], tpb)
    m = run(assemble(SRC), [], tpb, threads_per_block=tpb)
    print(f"H1: THREADS_PER_BLOCK={tpb} RTL MEM[100]={mem.memory[100]} model={m.memory[100]} {m.stores_conflict}")
    assert mem.memory[100] == m.memory[100]
EOF
make BUILD=build/a14/repro compile
PYGPI_PYTHON_BIN=$(which python3) LIBPYTHON_LOC=$(cocotb-config --libpython) COCOTB_TEST_MODULES=a14_h1 \
  PYTHONPATH=.:build/a14/repro COCOTB_RESULTS_FILE=build/a14/repro/results.xml \
  vvp -M $(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/a14/repro/sim.vvp
# → H1: THREADS_PER_BLOCK=4 RTL MEM[100]=1 model=4 [(0, 3, 100)]   (FAIL)
```

Add `PARAMS="-Pgpu.THREADS_PER_BLOCK=16"` to `make` to see RTL 13 against model 16.

Severity: the brief's rule says a silent wrong result in a matrix-covered configuration is High. It is not Critical
for three reasons: the model does flag the conflict, no kernel or fixture has one, and `isa.md` never promised a
winner. This is the `STR` half of AUDIT-12 H3. That finding was closed for `STS` only.

### L1 — "interleaving-independent" is checked by two schedules (Low)

`test_isa_diff.expected` and `test_divergence.check` treat a program as interleaving-independent when xf.isasim's
lrr and gto runs agree. The RTL's GTO picks the oldest warp and yields on global memory, so it reaches orders that
neither model policy produces:

- `sts_conflict8` at W = 2 can legally end with `SMEM[0]` = 1 or 3 in block 0, and with 0–3 at W = 4. That is 4 and
  16 distinct images under random legal schedules (`test_sts_conflict_fixture_is_interleaving_independent`). The
  RTL's three policies happen to issue warp W−1's `STS` last, so the fixture passes. At W = 4 (1 lane per warp) it
  checks no lane order at all.
- Of the 4241 fuzz programs, 12 passed the lrr == gto gate, yet the RTL's legal order gave different memory, and the
  trace replay matched the RTL every time. The IDs are listed below; `build/a14/cfg/<cfg>/batch1.json` holds the
  program text. One of them is in the `warps2` 32-bit matrix configuration (`w2lrr_t4_s0_b32/1/2`). Others are
  `w2gto_t4_s1_b8/1/12` (108 words, 4 threads), `w4gtlrr_t8_s1_b32/1/20` (68 words, 3 threads) and
  `w4gto_t4_s1_b8_c4/1/24`.

Fix: compare W > 1 runs against a model replay of the RTL's recorded issue order (as here), or check independence
over many random legal schedules. Correct the comment in `xf/isa_fixtures.py` and the "pin this" sentence in
`isa.md`.

### L2 — RET row (Low)

`docs/isa.md` opcode table: `RET | finish current block (with the SIMT stack: the active lanes)`. ADR-012/014 and both
implementations end only the issuing warp at W > 1 (control `test_ret_ends_only_the_issuing_warp`). This is not
AUDIT-9 L6, which covered the BR row and BAR.

## Fuzz statistics

| Campaign | Configs | Programs compared | Issues (model steps) | RTL cycles | Disagreements |
|---|---|---|---|---|---|
| 1: W {1, 2, 4} × policy × SIMT × TPB {4, 8, 16} × {8, 32}-bit, plus W=1 matrix configs | 110 | 3298 | 1,470,933 | 12,030,362 | 0 |
| 2: TPB {1, 2, 6, 12}, W {1, 2, 3, 4, 6}, padded to PC 255, multi-warp at latency 3/10 | 77 | 919 | 473,133 | 3,926,344 | 0 |
| 3: 100–255 threads, 32-bit | 4 | 24 | 95,948 | 635,696 | 0 |
| Total | 191 | 4241 | 2,040,014 | 16,592,402 | 0 |

- **Programs and features.** 7 random programs are model rejects (not run). 479 compared programs are
  interleaving-dependent and could only be checked through the trace replay. Feature counts over campaign 1's
  matches:

  | Feature | Programs |
  |---|---|
  | SMEM | 2679 |
  | if/else | 1755 |
  | R13–R15 writes | 1464 |
  | don't-care bits | 1485 |
  | BAR | 1450 |
  | loops | 1187 |
  | `BRp` | 633 |
  | RET in a region | 177 |
  | backward SYNC | 176 |
- **Rejects:** 60/60 probe runs agree (`without SYNC`, `deeper than 8`, or both run).
- **Mutants:** 7/7 killed.

  | Mutant | Killed by |
  |---|---|
  | SM1 = AUDIT-12 S02 | 5/30 programs, `w1_t4_s0_b8` |
  | D11 | 4/30 |
  | D17 | 4/30 |
  | C08 | 6/30 trace errors at `w2lrr_t4_s1_b8` |
  | S04 | 1/30 (hang) at `w1_t8_s1_b8`; 0/30 at TPB 4 |
  | S02 | `rej_consumed_{4,8}` run to completion instead of `$fatal` |
  | S03 | `rej_overflow9_8` runs to completion |

  S04 survives the default-width fuzz, so a dedicated backward-SYNC test (AUDIT-9 L3) is still worthwhile.
- **Discarded:** a first pass of campaign 2 let blocks' SMEM dumps overlap (TPB 1, 2, 6, 12 at 8 bits). That is a
  harness bug causing cross-block races, which the model scopes out. Every disagreement it produced sat in the overlap
  words. The pass was discarded, the layout fixed (32-word stride, thread cap) and the campaign rerun.

## Checked and consistent

- The opcode table, `xf/isa.py`, the assembler and `decoder.sv`.
- Within one `STS` the highest active lane wins, at every TPB and W.
- Across warps, the last `STS` issued wins: trace-exact, including after yielding `LDR`s and with no `BAR`.
- Cross-warp `STR` races follow issue order even at latency 10.
- `BAR` eligibility: no trace ever issued a warp at a `BAR` or after `RET`.
- `warp_issues_total` equals the model's steps.
- The contract-12 counters hold in every run.
- SIMT stack behaviour at 1–16 lanes per warp, including non-power-of-two warps (3 and 6 lanes), PC wrap at 255, and
  L1-D read-back after a conflict.
