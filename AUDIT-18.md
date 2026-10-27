# AUDIT-18 — Encoding soundness, round 5 (no false negatives)

Lens: do `docs/isa.md`, `xf/isasim.py` and the RTL agree after SG25 (STR conflicts: highest lane wins, ADR-016; the
harness memory applies a cycle's channel writes from channel 0 up; MeDiC priority on reads only; W > 1 compared against
`xf.isasim.run_trace` replaying the RTL's recorded issue order) and SG26 (SMEM wrap, 253/255-thread launches)? And is
the replay harness itself sound: can a wrong RTL pass because the model replays whatever order the RTL chose? Findings
already recorded in AUDIT-9 to AUDIT-16 are not repeated unless their severity was wrong.

Audited at `e9c9eef` (main) with iverilog 12.0, sv2v, cocotb 2.0.1. `src/`, the existing tests, the `xf` library, the
Makefile and the docs are unchanged. Scratch tooling and every build lived under `build/a18/` (git-ignored), at most two
simulations at a time. AUDIT-14's tooling was copied from `/home/claude/audit14/build/a14/fz` and retargeted, not
modified in place.

## Findings

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| H1 | High | Cross-warp `STR` order is not issue order. At `WARPS_PER_CORE > 1`, when a warp's store lanes wait for a data channel and a lower-numbered warp's later store arrives before a channel frees, the controller claims the lower consumer indices first. The earlier-issued store lands last. `xf.isasim.run_trace`, replaying the RTL's own legal order, stores 4 at word 100. The RTL stores 8. The `str_conflict` fixtures rely on exactly this rule at W > 1 (all warps of a block store to word `b`), and `docs/isa.md` states no cross-warp `STR` rule (for `STS` it says "the last warp to issue wins") | `test_audit18.py::test_h1_model_matches_rtl_on_recorded_order` (strict xfail), `::test_h1_isa_md_states_cross_warp_str_rule` (strict xfail); RTL command below |
| H2 | High | Re-rates **AUDIT-10 L2** (Low): no matrix configuration runs several warps with memory latency or an I-cache (`warps2`/`warps4gto` are L0 without the I-cache; `latency`, `icache` and `all` are W = 1). H1 needs both, so the H1 RTL passes `make -j4 matrix`. The harness itself catches H1 (it is a mismatch, not a silent pass), but no gate runs it where it occurs | `test_audit18.py::test_h2_matrix_has_multiwarp_with_latency_or_icache` (strict xfail) |
| M1 | Medium | Unstated memory-ordering assumptions. The STR rule and every cross-warp result rest on three properties of `test/helpers/memory.py`, and ADR-016 names only the first: (a) same-cycle writes apply from channel 0 up; (b) a read and a write to one word in the same cycle return the old word (reads are serviced before writes); (c) every channel has the same fixed latency, so claim order is completion order. An external memory that meets (a) but not (b) or (c) gives other results with no RTL change. For one `STR`, the RTL has no conflict resolution of its own: the result is whatever the memory does | `test_audit18.py::test_memory_model_same_cycle_read_sees_old_value` (control), `::test_m1_adr016_states_memory_ordering_assumptions` (strict xfail) |

There are no critical findings. Outside H1, every program the model accepts matched the RTL bit for bit, with `warp_issues_total` equal to the model's steps and `check_counters` holding (statistics below).

### H1 — cross-warp STR order (High)

Program (`H1_SRC` in `xf/tests/test_audit18.py`, 41 words). Block 1, on core 1, keeps the four data channels busy
with 4-lane loads. Block 0, on core 0, has two 4-lane warps. Warp `w` spins `d[w]` loop trips (data words 0 and 1), then
every lane stores `threadIdx + 1` to word 100 (kind 0 in words 4/5; kind 1 loads it instead). After `BAR` every lane
reads word 100 back to `140 + threadIdx`. Data: `d = (2, 1)`, kinds `(0, 0)`, 16 threads.

Build: `THREADS_PER_BLOCK=8 WARPS_PER_CORE=2 ICACHE_LINES=16 PIPELINED=1`, `MEM_LATENCY=10`.

- Observed (RTL, through `run_kernel(trace=...)`): `MEM[100] = 8` and `MEM[140..147] = 8`, so warp 1's lane 3 wins.
- Expected (`xf.isasim.run_trace` on the recorded order, which it accepts as legal): `MEM[100] = 4`. Warp 1 issues its
  `STR` first, so warp 0's later `STR` should win.

Mechanism: warp 1's store lanes (consumers 4–7) are waiting because block 1 holds all four channels. While they wait,
warp 0 issues its store (consumers 0–3). With the I-cache and `PIPELINED`, the issue gap is about 5 cycles; at L10 a
channel is held about 13 cycles. When the channels free, every `IDLE` channel scans consumers from 0, so warp 0's lanes
claim first and warp 1's lanes are written one latency later. Lane order inside one `STR` still holds: its lanes rise
together and are claimed lowest first. Only the order between warps breaks.

Command (no file under `test/`):

```
mkdir -p build/a18/repro
# a18_repro.py: run_kernel(trace=...) then xf.isasim.run_trace, as test_isa_diff does (build/a18/repro/a18_repro.py)
python3 -c "import json,sys; sys.path.insert(0,'.'); from xf.tests.test_audit18 import *; \
  json.dump(dict(src=H1_SRC, data=H1_DATA, threads=H1_THREADS), open('build/a18/repro/h1.json','w'))"
make BUILD=build/a18/repro/h1 PARAMS="-Pgpu.THREADS_PER_BLOCK=8 -Pgpu.WARPS_PER_CORE=2 -Pgpu.ICACHE_LINES=16 -Pgpu.PIPELINED=1" compile
MEM_LATENCY=10 A18_CASE=build/a18/repro/h1.json COCOTB_TEST_MODULES=a18_repro PYTHONPATH=.:build/a18/repro \
  COCOTB_RESULTS_FILE=build/a18/repro/h1/results.xml PYGPI_PYTHON_BIN=$(which python3) \
  LIBPYTHON_LOC=$(cocotb-config --libpython) vvp -M $(cocotb-config --lib-dir) -m libcocotbvpi_icarus build/a18/repro/h1/sim.vvp
# -> A18: 9 words differ [100] RTL 8 model 4 [140] RTL 8 model 4 ...   (FAIL)
```

Reach: the full sweep (`d0, d1 ∈ 1..5`, hog phase 1..6, kinds SS/SL/LS = 450 programs per config) gives:

| Configuration | Mismatches |
|---|---|
| `W2 TPB8 ICACHE PIPELINED L10`, 8-bit | 2 / 450 (`d,kh` = `21,1` and `43,3`, both store/store) |
| `W2 TPB8 ICACHE PIPELINED L10`, 32-bit, `SIMT_STACK=1` | 2 / 450 |
| `W2 TPB8 L3`; `W2 TPB4 L0` (= `warps2`); `W2 TPB8 ICACHE PIPELINED L3`; `W2 TPB4 ICACHE PIPELINED L10`; `W2 TPB8 +DCACHE ICACHE PIPELINED L10`; `W2 GTO TPB8 +MEDIC ICACHE PIPELINED L10` | 0 / 450 each |
| `W4 GTO TPB4 ICACHE PIPELINED L10` (300 random tables) | 0 / 300 |

No load/store mix diverged in the sweep. A load can be overtaken in the same way, but I did not land one.

The fix is either in the RTL, by making the order between warps hold under contention, or in the contract. An RTL fix
could stop a warp issuing a global store while another warp of the core has unclaimed store lanes, or claim in age order
rather than consumer index. A contract fix would make conflicting cross-warp `STR`s without a `BAR` undefined in
`docs/isa.md` and ADR-016, with `str_conflict` restructured so its warps' stores are `BAR`-separated. Either way,
`docs/isa.md` needs a cross-warp `STR` sentence, like the one for `STS`.

### H2 — matrix gap (High; AUDIT-10 L2 re-rated)

AUDIT-10 L2 recorded the gap as a coverage note. With H1 it becomes a gate hole: an RTL that disagrees with the model
on the `str_conflict` rule at W > 1 passes every matrix configuration. A configuration such as
`-Pgpu.WARPS_PER_CORE=2 -Pgpu.THREADS_PER_BLOCK=8 -Pgpu.ICACHE_LINES=16 -Pgpu.PIPELINED=1` with `MEM_LATENCY=10`,
running `test_isa_diff` plus a contention fixture shaped like `H1_SRC`, would gate it. A contention fixture is needed
because `str_conflict` alone never lands in the window (0 failures in every swept configuration at that fixture's
shape).

### M1 — memory-ordering assumptions (Medium)

`Memory.run` services every channel's read before any write in the same call, and counts a fixed `latency` per channel.
The controller claims lowest consumer first, and all lanes of one instruction raise valid in the same cycle. The
documented rule (highest lane wins) follows only if the memory also completes in claim order and applies writes from
channel 0 up. ADR-016 states the second requirement ("An external memory must do the same") but not in-order
completion, and not read-before-write. The read-before-write rule decides the result whenever a racing read and write
are claimed in the same cycle, which is AUDIT-10 M2's window. So the answer to the brief's question is:
- For one `STR`, ADR-016's channel order (plus fixed latency) is the only thing deciding the winner; the RTL does not
  resolve conflicts itself.
- Between warps, the controller's consumer-index claim order also decides it, which is H1.

## Replay-harness soundness

- **Illegal orders.** `run_trace` raises on a pick of a done warp, a warp held at `BAR` (including after `RET`), an empty
  warp, a short trace (a warp never issued, or a block missing) and a long trace. `test_isa_diff` additionally checks
  `warp_issues_total == steps`. Control: `test_audit18.py::test_run_trace_rejects_skipped_warp_and_post_ret_issue`.
  An independent re-implementation of the replay (the source-patched copy from AUDIT-14) agreed with the library
  `run_trace` on every multi-warp run of the campaign (memory and steps), and neither raised where the other did not.
- **Mutants at the `warps4gto` matrix parameters** (`test_isa_diff`, `test_multiwarp`): see the table below. A warp that
  is never issued, or a `BAR` that holds nothing, is rejected by the replay. A scheduler that ignores `WARP_POLICY`
  passes `test_isa_diff` (by design: every policy is legal) and is caught only by `test_multiwarp::test_policy_order`.
- **What the replay cannot see.** The model applies each instruction's memory effect atomically at issue; the RTL
  overlaps several warps' global operations, and their effect order comes from channel claiming. The harness records
  issue order, not effect order. When the two differ it reports a mismatch (H1), so this is not a false negative in
  the comparison itself. The false negative is in coverage: no matrix configuration produces the contention (H2).

| Mutant (`build/a18/mut/<m>/src`) | `test_isa_diff` (8-bit fixtures) | `test_multiwarp` |
|---|---|---|
| `bar`: `scheduler.sv` never sets `warp_bar` | 6 / 37 fail, e.g. `str_conflict8`, `smem_wrap8`: `IsaSimError: block 0: issue 6 picks warp 0, ready are [1, 2, 3]` | 2 / 6 fail (`test_barrier`, `test_partial_block`) |
| `skip`: the last warp is never live | 33 / 37 fail: `IsaSimError: block 0: issue 60 is past the end of the trace` | 6 / 6 fail |
| `pol`: `warp_scheduler` always runs LRR (`case (0)`) | **37 / 37 pass** (any ready order is legal) | 3 / 6 fail (`test_policy_order`, `test_policy_ages`, `test_age_saturation`) |

Command: `build/a18/mut.sh <m>` (compiles with `GPU_SRC=` the mutated sources and
`PARAMS="-Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1"`, then runs both modules).

## Method

- **Random differential fuzz.** AUDIT-14's generator (nested if/else with `SYNC`, per-lane loops, backward `SYNC`, `RET`
  and `BAR` in divergent regions, R13–R15 writes, don't-care bits, `BRp`, SMEM conflicts) plus one new statement class.
  Each block has 16 shared global words, and `STR`/`LDR` hit them with every lane on one word, lane pairs, groups of four
  and lane mod 3. That gives `STR` conflicts inside one store and between warps, and cross-warp load/store races, in
  every configuration. `STS` conflicts are always on. Oracle: `xf.isasim.run` at W = 1, and at W > 1 both
  `xf.isasim.run_trace` and AUDIT-14's independent copy, on the order recorded from `scheduler.issue`,
  `selected_warp` and `block_id`. Memory must be bit-identical, `warp_issues_total` must equal steps, and
  `check_counters` must hold.
- **Directed race probes.** `race.py`, `race2.py` and `race3.py` under `build/a18/fz/`: per-warp delay loops, channel
  hogging by the other core, and every warp storing to or loading from one word. `race3` sweeps the warp delays, the
  hog phase and the store/load mix.
- **Configurations.** Fuzz: `WORD_BITS` 8 and 32; W = 1, and W = 2/4 under LRR, GTO and GTLRR; `SIMT_STACK` 0 and 1;
  TPB 4, 8 and 16; `MEM_LATENCY` 0, 3 and 10 rotated across them; W = 1 and W > 1 with `MEDIC`, `COALESCE`, `DCACHE` and
  `all`; 100–255-thread launches at 32 bits. Probes: the configurations in the tables.
- **docs/isa.md rules.** Checked against `xf/isasim.py`, `simt_stack.sv`, `scheduler.sv`, `shared_memory.sv` and
  `controller.sv`:
  - `STS`: highest lane, and across warps the last issuer. Holds: SMEM accesses do not yield, so warps' `STS` are
    serialised.
  - `STR`: highest lane within one `STR` holds. Across warps there is no rule (H1).
  - `RET`: ends the issuing warp, or its active lanes with the stack. The model's "drop every emptied entry" equals the
    RTL's "pop to the highest non-empty entry", because emptied entries are a suffix.
  - `BAR` in divergence: flags the whole warp in both implementations (doc gap already AUDIT-9 L6).
  - SMEM mask to 6 bits and the 64–255 wrap: agree.

## Fuzz statistics

| Campaign | Configs | Programs compared | Issues (model steps) | RTL cycles | Disagreements |
|---|---|---|---|---|---|
| Broad fuzz, seed 1 (`cfgs_broad.txt`, 10 each; big launches 6 each, seed 3) | 48 | 477 (+1 model reject, not run) | 284,709 | 3,248,823 | 0 |
| Pilot fuzz, seed 2 (W 2/4 at L3/L10, `medic`/`dcache`/`coal`) | 8 | 160 | 61,763 | 1,043,303 | 0 |
| `race` probe (per-warp delays, store or load of one word) | 18 | 216 | 47,526 | 633,355 | 0 |
| `race2` probe (32-bit, looping stores/loads under contention) | 27 | 432 | 338,760 | 3,482,478 | 0 |
| `race3` sweep (delays × hog phase × store/load mix) | 9 | 3,900 | 1,928,884 | 16,713,798 | **4 (H1)** |
| Total | — | 5,185 | 2,661,642 | 25,121,757 | 4 |

- Programs with an `STR` conflict (model `stores_conflict` non-empty): 127 broad, 33 pilot, 114 + 288 + 3,600 probes.
  Every broad program had `STS` conflicts available; feature counts over the broad campaign's 477: shared global words
  367, SMEM 333, shared `LDR` 278, shared `STR` 276, if/else 209, don't-care bits 155, R13–R15 writes 148, `BAR` 143,
  loops 111, `BRp` 56, backward `SYNC` 18, `RET` in a region 12.
- Multi-warp runs replayed through both `xf.isasim.run_trace` and the independent copy: 5,059, all equal in memory and
  steps.
- Counters (`check_counters`, `warp_issues_total == steps`): no failure in any run.
- Discarded: the first big-launch pass hit an assembler range error in my generator (`CONST` of an output base
  > 255 once the shared region pushed it up). The generator was fixed and the three configurations were rerun.
- Mutants: 3 / 3 caught by the `warps4gto` matrix modules; the policy mutant only by `test_multiwarp`.

## Checked and consistent

- Within one `STR`, the highest active lane wins at every TPB (4, 8, 16), W, latency, and under `COALESCE`,
  `DCACHE`, `MEDIC` and `all`.
- `STS` conflicts within and across warps: no disagreement.
- The SG26 shapes: SMEM wrap and masking at 8 and 32 bits, and 100–255-thread launches.
- `warp_issues_total` equals the model's steps on every compared run, and contract-12 counters held on every run.
- No issued order was ever rejected by `run_trace` on the unmutated RTL (no warp issued at a `BAR` or after `RET`).
