# ADR-016 — STR lane conflicts: highest lane wins

## Status
Accepted (PLAN SG25, AUDIT-14 H1 and L1).

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

## Alternatives rejected
- Make `STR` conflicts undefined, or have the model raise. That discards a rule both implementations can meet
  cheaply, and it differs from `STS`.
- Lowest lane of the last channel group (the old RTL behaviour). That ties the ISA to `DATA_MEM_NUM_CHANNELS`.
- Check interleaving independence over random schedules. That is sound only with high probability, and it still
  skips the dependent fixtures.

## Consequences
- `test_medic`'s mixed workload keeps every number (hot lanes 18 / 28, 94 requests, 781 / 1071 / 1813 cycles). Its
  monitor checks priority on reads only. `test_medic_str_conflict` kills a write-priority mutant: odd lanes high
  utility, even lanes low, then one conflicting `STR`.
- `test_divergence::test_str_conflict` checks the rule at 8 lanes (two channel groups).
