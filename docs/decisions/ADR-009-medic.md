# ADR-009 — MeDiC memory-divergence correction (Phase 9)

## Status
Implemented; classification revised in SG19 (below). `src/l1d_cache.sv` is a direct-mapped,
write-through L1-D shared by every LSU of the GPU (`DCACHE_LINES`); with `MEDIC=1` it classifies each
LSU lane, bypasses low-utility lanes, and the data controller serves high-utility lanes first.

## Context

[Ausavarungnirun et al., "Exploiting Inter-Warp Heterogeneity to Improve GPGPU Performance" (MeDiC)](https://arxiv.org/abs/1804.11038)
reports about 21.8% speedup from:

1. **Online warp characterization**: track each warp's hit ratio.
2. **Cache bypassing for low-utility warps**: warps that mostly miss skip the L1-D, leaving its
   capacity to the others.
3. **Memory controller priority**: high-utility warps' requests go first.

## Decision
The three mechanisms are applied per L1-D consumer, i.e. per LSU lane (`NUM_CORES ×
THREADS_PER_BLOCK` of them), not per warp: the L1-D sits between the LSUs and the data controller and
sees lanes, and at `WARPS_PER_CORE > 1` a lane's LSU belongs to one thread.

### Characterization
Each lane keeps a 4-lookup hit/miss window; it is high-utility while at least `MEDIC_THRESHOLD = 2` of
its last 4 lookups hit (`warp_high_utility`, `warp_low_utility` outputs).

### Bypass
On a miss by a low-utility lane the fill goes to the lane without installing the line (the install is
gated). Writes are write-through for every lane.

### Priority
The controller (`PRIORITY_ENABLE = MEDIC`) runs two claim passes over its consumers each cycle:
high-utility lanes first, then the rest.

### Revision (SG19): lanes start high-utility

As first built, each lane's 4-lookup hit window reset to 0 and `MEDIC_THRESHOLD` was 4, so every
lane began low-utility and was bypassed; its own misses never installed a line, so it could never
leave bypass. `MEDIC=1` was a cache-off switch (hot-address test 0 hits / 12 misses; conv1d 0 / 18)
and the priority pass never fired.

Decision: the window resets to all hits (`4'b1111`) on reset and `telemetry_reset`, and a lane is
high-utility while at least `MEDIC_THRESHOLD = 2` of its last 4 lookups hit. The bypass decision at
fill time sees the window including the current miss, so a streaming lane installs its first 2
misses (1110, 1100) and bypasses from the third (1000). A hot lane with one miss in three stays
high and installs; a low-utility lane leaves bypass once hits (from lines another lane installed)
bring it back to 2 of 4. Priority applies to high-utility lanes, which is every lane until it
demotes.

Alternatives rejected: a separate warm-up counter (same effect as the all-hits reset, more state);
threshold 3 (one miss in a lane's first two lookups would bypass it); threshold 1 (streaming lanes
pollute the cache for 3 fills, and a lane is demoted only by 4 straight misses).

## Tests
- `make test_l1d_indepth PARAMS="-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1"`: the hot-address case hits with
  exact counts.
- `make test_medic` (`test/test_medic.py`) runs a mixed workload (hot and streaming lanes in one
  launch) at `MEDIC=0` then `MEDIC=1`: streaming lanes install 2 lines each and bypass the rest, hot
  lanes hit at least as often as at `MEDIC=0`, and a monitor fails the run if a bypassed fill leaves a
  line behind or a low-utility lane gets a channel while a high-utility lane waits. Data memory runs
  4 cycles slower than program memory so the two cores' requests overlap and priority has to decide.

## Results
From `bench/phase-11-all.json` (`docs/final-report.md` cites the rows):
- On the eight regression kernels `MEDIC=1` equals plain `DCACHE_LINES=16` in cycles and counts: no
  lane demotes.
- On the mixed workload, hot lanes hit 18 / 28 against 14 / 28, data requests drop from 98 to 94, and
  cycles drop by 3–4 at every latency. The cycle gain is small because a one-warp block waits for its
  slowest lane, a streaming lane that misses either way.

## Consequences
- At `MEDIC=0` the window feeds only `MEDIC`-gated logic, so the L1-D behaves as a plain cache.
- Phase 9's results for `MEDIC=1` (conv1d 0 hits / 18 misses, 335 cycles, in `bench/phase-9-medic.json`)
  are the pre-SG19 cache-off behaviour, kept as history.
