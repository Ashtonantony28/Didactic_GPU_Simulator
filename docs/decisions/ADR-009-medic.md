# ADR-009 — MeDiC memory-divergence correction (Phase 9)

## Status
Implemented; classification revised in SG19 (below). `src/l1d_cache.sv` provides a real
direct-mapped L1-D with per-lane hit/miss tracking; MeDiC bypass and controller priority
arbitration are wired and parameterized. Functionally working — the
opportunity to see MeDiC's *value* depends on inter-warp heterogeneity
which our single-warp datapath can't yet produce; see "What MeDiC
needs" below.

## Context

[Ausavarungnirun et al., "Exploiting Inter-Warp Heterogeneity to Improve
GPGPU Performance" (MeDiC)](https://arxiv.org/abs/1804.11038) reports
~21.8% speedup and ~20.1% energy improvement by:

1. **Online warp characterization** — track hit / miss ratio per warp.
2. **Cache bypassing for low-utility warps** — warps that miss too often
   are forced to bypass the L1-D, freeing capacity for the high-utility
   warps.
3. **Memory controller priority** — high-utility warp requests jump the
   memory queue.

## Decision

MeDiC's three components depend on:

- An L1-D cache (which we don't have yet).
- Multiple warps per core (Phase 8).

We document the design and leave the implementation as a follow-on once
Phase 8 lands.

## Component 1 — Warp characterization

Per warp, maintain a 4-bit shift register of recent hit/miss outcomes.
`hit_count = popcount(window)`; classify the warp as `high` (≥ threshold)
or `low` (< threshold) utility.

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

Consequences: at `MEDIC=0` nothing changes (the window feeds only `MEDIC`-gated logic; regression
and `test_l1d_indepth` cycle-identical at default and `DCACHE_LINES=16`). At `MEDIC=1` every kernel
test now matches plain `DCACHE_LINES=16` (conv1d 10 / 8, hot address 8 / 4). `make test_medic`
runs a mixed workload at `MEDIC=0` then `MEDIC=1`: the streaming lanes install 2 lines each and
bypass the rest, the hot lanes hit 18 / 28 against 14 / 28, and a monitor fails the run if a
bypassed fill leaves a line behind or a low-utility lane gets a channel while a high-utility lane
waits. With one warp per core, priority only decides when the two cores' requests overlap; the test
slows data memory (`MEM_LATENCY + 4`) relative to program memory to make them overlap.

## Component 2 — Cache bypass

L1-D lookup logic adds an extra input: `warp_class`. If `class == low`,
the cache passes the request through to main memory without updating LRU
or installing the line in the cache. Implementation: gate the
"install_on_miss" signal.

## Component 3 — Memory controller priority

The controller's arbitration loop scans consumers from index 0
upward. Add a priority queue: high-utility requests jump ahead of
low-utility ones. Implementation: two parallel scan passes over the
consumer list.

## Test plan
- All Phase 5 kernels must still pass.
- New kernel: deliberately memory-divergent (some warps thrashing, others
  cache-friendly).
- Bench report: L1-D hit-rate per warp; cycle count vs. no-MeDiC baseline.

## What MeDiC needs to *help* (rather than just *run*)

MeDiC's three mechanisms — characterization, bypass, priority — all
target *inter-warp heterogeneity*. With a single warp per core, every
lane sees the same instruction stream and similar memory behavior. The
mechanisms still run correctly (you can observe the bypass kick in
when a lane gets classified low-utility), but the *benefit* is zero
because there's no competing high-utility traffic to favor.

Wiring multi-warp-per-core (ADR-008's open work) unlocks MeDiC's win.
The L1-D + MeDiC machinery is in place; the workload to exploit it
isn't yet.
