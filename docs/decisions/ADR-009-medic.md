# ADR-009 — MeDiC memory-divergence correction (Phase 9)

## Status
Implemented. `src/l1d_cache.sv` provides a real direct-mapped L1-D
with per-lane hit/miss tracking; MeDiC bypass and controller priority
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
