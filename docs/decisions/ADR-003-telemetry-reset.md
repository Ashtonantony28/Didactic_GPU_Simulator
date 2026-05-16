# ADR-003 — Telemetry counters on a separate reset domain

## Status
Accepted

## Context
Phase 3 added `icache_hits` / `icache_misses` counters to the fetcher.
Phase 5 added `mem_requests` / `mem_requests_coalesced` to the controller.

These counters initially shared the per-module `reset` signal. That
worked for a single block, but per-block dispatch asserts `core_reset` on
every finished block — so the counters were zeroed between blocks, and
ended up reading back as 0 at the end of every kernel.

## Decision
The fetcher takes a separate `telemetry_reset` input. It's wired from the
GPU-level external `reset` (never asserted by the dispatcher between
blocks). The counters reset only on `telemetry_reset`.

For the controller, `reset` and `telemetry_reset` are the same signal —
the data memory controller isn't reset between blocks (only the cores
are), so a separate domain isn't needed.

## Consequences
- `fetcher.sv` has an extra port.
- `core.sv` forwards it to the fetcher.
- `gpu.sv` ties `telemetry_reset` to the external `reset` for every core.
- I-cache contents are also flushed on `telemetry_reset` so that a fresh
  kernel launch starts cold.
