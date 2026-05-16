# ADR-004 — Instruction cache (Phase 3)

## Status
Accepted

## Context
Phase 3 adds a per-core L1-I cache. The fetch path is on the critical
loop for every cycle (each instruction goes FETCH → DECODE → REQUEST →
…), so even a small cache should produce visible cycle wins on
loop-heavy kernels.

## Options considered

| Option                          | Size    | Complexity | Expected win              |
|---------------------------------|---------|------------|---------------------------|
| Direct-mapped, 1 instr/line     | small   | low        | loop-heavy kernels        |
| Direct-mapped, N instr/line     | medium  | medium     | + spatial locality        |
| Set-associative                 | medium  | high       | + thrash protection       |
| Skip the cache entirely         | n/a     | none       | nothing                   |

## Decision
Direct-mapped, 1 instruction per line, parameterized line count
(`ICACHE_LINES`). Default `ICACHE_LINES=0` keeps the cache bypassed so
Phase 1's behavior is preserved.

The implementation lives behind a `generate if (ICACHE_LINES == 0)` block
in `src/fetcher.sv` — the no-cache path is identical to upstream.

## Consequences
- `matmul` (a kernel with a tight 14-instruction loop) gets a measurable
  speedup: 556 → 504 cycles (-9.4%), 13 hits / 28 misses with 16 lines.
- Straight-line kernels (matadd, vector_dot, relu, conv1d, transpose)
  see 0 hits — every instruction is fetched once per block. Two blocks
  on two cores each have their own cold cache.
- An ADR-noted future enhancement (Phase 3.1) would share the cache
  across blocks on the same core, or extend lines to >1 instruction for
  spatial locality.
