# ADR-016 — STR lane conflicts: highest lane wins

## Status
Accepted (PLAN SG25, AUDIT-14 H1 and L1). Amended by SG28 (AUDIT-18 H1 and M1).

## Context
When several lanes of one `STR` wrote the same word, `xf.isasim` kept the highest lane but the RTL kept the lowest lane
of the last group of `DATA_MEM_NUM_CHANNELS` lanes (lane 0 of 4, 4 of 8, 12 of 16). `docs/isa.md` stated no rule, and
the ISA fixtures skipped store-conflicting programs, so no gate compared them.

Two causes:
- **The memory model's order.** All lanes of an `STR` raise `write_valid` in the same cycle. Each idle channel, scanned
  upward, claims the lowest free consumer, so within a cycle a higher lane gets a higher channel, and a later group never
  starts before an earlier one. `test/helpers/memory.py` unpacks the packed bus most significant channel first, so it
  applied a cycle's writes from channel N−1 down, and channel 0 won.
- **MeDiC's priority pass** (`MEDIC=1`) also covered writes. Lanes of one `STR` with different utility claimed channels
  high-utility first, which broke the lane order (lanes 1, 3, 0, 2 on channels 0–3, so lane 2 won).

## Decision
- When lanes of one `STR` hit the same word, the highest lane wins, as for `STS` (ADR-007) and in `xf.isasim`.
- The memory model applies one cycle's channel writes from channel 0 up, so the highest channel wins. An external
  memory must do the same. The port lists do not change.
- `controller.sv` gives writes no priority pass: they are claimed in the second pass with every consumer, in consumer
  order. Reads keep MeDiC's priority (ADR-009).
- The fixture generator writes store-conflicting programs; `str_conflict{8,32}` check the rule.
- **AUDIT-14 L1.** `test_isa_diff` and `test_divergence` no longer treat a program as interleaving-independent when
  `"lrr"` and `"gto"` agree. At `WARPS_PER_CORE > 1` the runner records each block's warp issue order from the RTL
  (`run_kernel(trace=...)`), and the model replays it (`xf.isasim.run_trace`, built on a callable `warp_policy`).
  The replay fails if the RTL issues a warp that is done or held at a `BAR`. Every fixture is now compared at every
  warp count, the interleaving-dependent `sts_conflict` and `str_conflict` included.
- **Between warps (SG28, AUDIT-18 H1).** The rule covers the lanes of one `STR` only. `STR`s from two warps of a block
  to one word with no `BAR` between them race, and the word's value is unspecified (`docs/isa.md`), as on real GPUs:
  under channel contention a warp's store lanes can wait while a lower-numbered warp's later store claims first, so
  effect order is not issue order. `xf.isasim` flags such stores (`store_races`) and the words whose final value
  depends on them (`race_words`, carried through registers, loads and SMEM but not branches).
  `test.helpers.runner.memory_diff` leaves exactly those words out of the RTL comparison and requires the caller to
  name them, at most 16, so the set left out cannot grow unseen (AUDIT-13 L3).
- **Memory ordering (AUDIT-18 M1).** The rule and every cross-warp result also rest on two properties of
  `test/helpers/memory.py`, which an external memory must share: a read and a write of one word in the same cycle
  return the old word (reads are serviced before writes), and every channel has the same fixed latency, so
  transactions complete in the order they were claimed.

## Alternatives rejected
- Order cross-warp `STR`s in the RTL (claim in age order, or hold a warp's global store while another warp of the core
  has unclaimed store lanes). It costs claim logic or issue slots for an order real GPUs do not give without a
  barrier.
- Make `STR` conflicts undefined, or have the model raise. That discards a rule both implementations can meet
  cheaply, and it differs from `STS`.
- Lowest lane of the last channel group (the old RTL behaviour). That ties the ISA to `DATA_MEM_NUM_CHANNELS`.
- Check interleaving independence over random schedules. That is sound only with high probability, and it still
  skips the dependent fixtures.

## Consequences
- `test_medic`'s mixed workload keeps every number (hot lanes 18 / 28, 94 requests, 781 / 1071 / 1813 cycles). Its
  monitor checks priority on reads only. `test_medic_str_conflict` kills a write-priority mutant: odd lanes high
  utility, even lanes low, then one conflicting `STR`.
- `test_divergence::test_str_conflict` checks the rule over a whole warp at every warp count, 8 lanes (two channel
  groups) at one warp.
- `str_conflict{8,32}` store from lane pairs only; at one lane per warp a pair is two warps, a race that
  `test_isa_diff` names (`RACED`). `make matrix` config `warps2lat` (the AUDIT-18 H1 build) runs `test_cross_warp`:
  the H1 race is flagged and left out, and the same stores in `BAR`-separated rounds match word for word.
