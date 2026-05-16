# Phase 6 — Branch divergence

## Status
Partial — kernel-level data divergence working; SIMT-stack control
divergence designed in [ADR-006](../decisions/ADR-006-divergence.md).

## What landed
- `kernels/`/`test/test_divergent_relu.py`: a kernel where threads
  produce different outputs from the same instruction stream by
  combining via mask-blend. Validates that the no-divergence SIMT
  correctly handles data-divergent computations.

## What's documented but not wired
- Per-thread PC + active mask + SIMT stack with immediate-post-dominator
  reconvergence. Design fully written in ADR-006.
- A new opcode `RC` (reconverge) reserved in [docs/isa.md](../isa.md)
  for explicit reconvergence markers.

## Results

```
[PASS] divergent_relu  cycles=290  mem_req=16  coal=0
```

Result matches `out[i] = in[i] ± 1` based on parity of i.

## Reproducibility
```
make test_divergent_relu
```
