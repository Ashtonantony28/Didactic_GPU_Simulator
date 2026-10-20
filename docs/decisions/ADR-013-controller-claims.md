# ADR-013 — Controller claims and exact telemetry

## Status
Accepted (PLAN SG18). Implements the counter semantics of contract 12 (ADR-012).

## Context
`controller.sv` loops over its channels in one clocked block. It claimed a consumer with
`channel_serving_consumer[j] <= 1`, so a later channel in the same cycle still saw the consumer as
free: every idle channel took the same consumer. An 8-thread load probe never had more than one
distinct address on the 4 data channels and issued duplicate transactions (145 cycles; 207 at
`MEM_LATENCY=3`). The counters were `x <= x + 1` in the same loop, so they counted once per cycle
rather than once per transaction or merged consumer, and `icache_*_total` summed cores 0 and 1 only.

## Decision
- The three claims (read, coalesced read, write) are blocking `=`, as in upstream tiny-gpu. Releases
  stay non-blocking: a release needs the consumer's valid low and a claim needs it high, so they
  never meet on the same bit in one cycle.
- `mem_requests` / `mem_requests_coalesced` are accumulated in blocking locals over all channels and
  written once per cycle: one per downstream transaction, one per consumer merged into another's.
- `icache_hits_total` / `icache_misses_total` sum all `NUM_CORES` per-core counters.
- `test/helpers/runner.py::check_counters` checks every kernel run: `data_mem_requests` equals the
  reads plus writes the memory model serviced, `data_mem_requests_coalesced` equals the controller's
  consumer read handshakes minus serviced reads, and the I-cache totals equal the per-core sum. The
  memory model services a transaction once per valid-high period (it samples the bus before the
  edge's updates, so it used to see valid high one cycle after ready and service it twice at latency
  0; the controller ignored that second ready, so no cycle count depends on it).

## Alternatives rejected
- Blocking releases as well (upstream): no cycle gain, since a consumer released this cycle has
  already dropped its valid, and it widens the change.
- Counting at the memory side only: `mem_requests_coalesced` is invisible there.

## Consequences
The probe keeps 4 distinct addresses in flight with no duplicates (97 cycles; 135 at
`MEM_LATENCY=3`); kernels with parallel loads or stores get faster and every committed bench number
changes. `make test_controller` fails if the read or write claim reverts to `<=`; the coalesced
claim's mutant is caught under `COALESCE=1` (it is dead code otherwise).
