# ADR-012 — Multi-warp core and telemetry contract

## Status
Accepted (PLAN SG17, contract 12). Interface only; the RTL is SG20, the counter fixes SG18.
Implemented in SG20; implementation choices in ADR-014.

## Context
A core runs one block as one warp, so `BAR` is a no-op, the warp scheduler (ADR-008) has nothing
to schedule and MeDiC (ADR-009) sees no inter-warp heterogeneity.

## Decision
- `gpu.WARPS_PER_CORE` (default 1). A core still runs one block of `THREADS_PER_BLOCK` threads, now
  as `WARPS_PER_CORE` warps of `THREADS_PER_BLOCK / WARPS_PER_CORE` consecutive threads that
  time-share the core's lanes. `%blockDim`, `%threadIdx` and SMEM keep their block-level meaning.
- Each warp has its own PC, NZP, registers, LSU state and (SG21) SIMT stack. Branch rules apply per
  warp.
- `gpu.WARP_POLICY` (0 LRR, 1 GTO, 2 GTLRR) selects the `warp_scheduler` policy; its ports are
  unchanged.
- `BAR` stalls a warp until every unfinished warp of its block has reached a `BAR`.
- `WARPS_PER_CORE = 1` must reproduce today's cycle counts exactly.
- New outputs: `warp_issues_total`, `bar_stall_cycles_total`.
- Counter semantics, for every configuration:
  - `data_mem_requests` = downstream data transactions.
  - `data_mem_requests_coalesced` = consumer requests served by another consumer's transaction.
  - `icache_hits_total` / `icache_misses_total` = sums over all `NUM_CORES`.
  - `warp_issues_total` = instructions issued, summed over all warps and cores: each instruction a
    warp starts counts once, `RET` and `BAR` included, and an instruction that yields and re-issues
    (ADR-014) still counts once. It equals `xf.isasim`'s `steps` for the same launch.
  - `bar_stall_cycles_total` = summed over all warps and cores, the cycles a warp is held at a `BAR`:
    from the cycle after its `BAR` completes up to, not including, the cycle the barrier releases
    (the first cycle in which every unfinished warp of the block is at a `BAR`). The last warp to
    arrive adds 0; it is always 0 at `WARPS_PER_CORE = 1`.
  - All counters reset with the GPU, not with the per-block core reset.

## Alternatives rejected
- Several resident blocks per core: hides latency, but `BAR` stays meaningless and SMEM would
  need partitioning.
- Physical lanes = `THREADS_PER_BLOCK` with warps as extra blocks: same problem.

## Consequences
Programs that relied on lockstep for SMEM ordering (no `BAR`) become racy at `WARPS_PER_CORE > 1`.
`xf/isasim.py` (`warps_per_core`, `warp_policy`) models this, but round-robin (`"lrr"`) keeps warps
one instruction apart and hides most missing-`BAR` races (`matmul_tiled` without `BAR` still passes).
Greedy (`"gto"`) runs a warp until it stalls and exposes them, so race checks use `"gto"`.
