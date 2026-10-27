# ADR-014 — Multi-warp core implementation

## Status
Accepted (PLAN SG20). Implements ADR-012 (contract 12); the warp scheduler is ADR-008's module, ports
unchanged.

## Context
ADR-012 fixes the interface: one block per core, split into `WARPS_PER_CORE` warps of `LANES =
THREADS_PER_BLOCK / WARPS_PER_CORE` consecutive threads that time-share the core's lanes, per-warp PC,
NZP, registers and LSU state, block-level SMEM, a real `BAR`, and `WARPS_PER_CORE = 1` cycle-identical.
The core has one fetcher, decoder and control FSM (`scheduler.sv`), which steps one instruction at a
time through FETCH, DECODE, REQUEST, WAIT, EXECUTE, UPDATE.

## Decision
- **Issue slot.** At `WARPS_PER_CORE > 1`, `IDLE` is the issue slot: `warp_scheduler` picks a warp,
  the FSM loads that warp's PC and runs one instruction, and `UPDATE` stores the warp's next PC and
  returns to `IDLE`. One instruction is in flight in the front end at a time.
- **Scheduler inputs.** `warp_ready` = the warp has threads, has not executed `RET`, is not held at a
  `BAR`, and is not waiting on a yielded load or store; it is driven only in the issue slot (zero
  otherwise), so the LRR pointer and GTO's sticky warp move once per issue. `warp_progressing` = ready
  and the last warp issued. `warp_age` = instructions the warp has issued, saturating at 254 (the
  module's GTO search starts from 255 with a strict `<`, so a warp of age 255 could never be picked and
  the core would hang once every ready warp got there); GTO's "oldest" is the smallest age, ties to the
  lowest warp.
- **Lanes and banking.** `LANES` ALUs are shared: lane `l` takes thread `current_warp * LANES + l`'s
  operands. Registers, NZP/next-PC and the LSU stay per thread (`THREADS_PER_BLOCK` of each), which
  is per warp × lane; only the issuing warp's registers and PC units are enabled. The data memory
  interface keeps `THREADS_PER_BLOCK` consumers per core, so `gpu.sv`'s controller wiring is
  unchanged.
- **Load/store yield.** A warp whose global `LDR`/`STR` is still in flight in `WAIT` yields: it is
  marked pending, the core returns to `IDLE` and may issue another warp. The LSUs of a non-issuing
  warp see `core_state = IDLE` and keep the memory operation latched at their `REQUEST`, so the access
  completes in the background. Once its LSUs are done the warp is ready again and re-issues the same
  instruction: the LSUs, already `DONE`, ignore `REQUEST`, and `EXECUTE`/`UPDATE` write the result back.
  SMEM accesses take two cycles and do not yield.
- **Branches.** A warp follows its lowest thread's next PC (`next_pc[warp * LANES]`), as the block
  followed thread 0 before; `xf.isasim` does the same per warp.
- **BAR.** `UPDATE` of a `BAR` flags the warp. The barrier releases in the first cycle every live warp
  (has threads, no `RET`) is flagged; a warp's `RET` can release it. The block finishes in the first
  issue slot with no live warp.
- **`WARPS_PER_CORE = 1`.** The FSM keeps today's `UPDATE → FETCH` path, `BAR` is ignored, and no
  warp scheduler is instantiated, so every cycle count is unchanged.
- **Differential harness.** `test_isa_diff` / `test_xf_isa_diff` run `xf.isasim` with the DUT's
  `WARPS_PER_CORE`. At more than one warp the model replays the RTL's recorded issue order (SG25, ADR-016).
  `warp_issues_total` must equal the model's steps.

## Alternatives rejected
- Whole-instruction time-sharing without the yield: simpler, but a warp waiting on memory holds the
  core, so nothing overlaps and GTO never sees a stall other than `BAR`.
- `LANES` LSUs with saved per-warp state: the state is the same storage as per-thread LSUs, plus a
  save/restore path and a narrower memory interface.
- Keeping the decoded instruction per warp instead of re-issuing after a yield: one re-fetch (an
  I-cache hit when enabled) is cheaper than per-warp copies of every decoded field.

## Consequences
- With `MEM_LATENCY = 0` a block takes roughly `WARPS_PER_CORE` times the cycles (each instruction
  runs once per warp on `LANES` lanes, plus a re-issue per load or store); overlap only pays off
  when memory latency exceeds the switch cost.
- At `WARPS_PER_CORE = 2` the two greedy policies coincide; GTO and GTLRR differ only with more warps
  (`test_multiwarp::test_policy_ages`).
- A warp issues only its `LANES` loads together, so `test_controller`'s channel-saturation probe
  expects at least `min(channels, LANES)` distinct addresses in flight at `WARPS_PER_CORE > 1`.
- `make test_multiwarp` runs `test/test_multiwarp.py` at `WARPS_PER_CORE = 4` under each policy and
  at 2; its barrier tests fail if `BAR` is made a no-op. The matrix gains `warps2` and `warps4gto`.
