# Phase 9 — MeDiC memory-divergence correction

## Status
Designed in [ADR-009](../decisions/ADR-009-medic.md).

## Why it stops at design
MeDiC's three components — warp characterization, cache bypassing,
memory controller priority — all assume:
1. An L1-D cache. (We don't have one yet; only the memory controller.)
2. Multiple warps per core. (See Phase 8 — designed but not wired.)

Without those prerequisites, "MeDiC" would just be a model of MeDiC,
not the thing itself.

## Citation
Ausavarungnirun et al. "Exploiting Inter-Warp Heterogeneity to Improve
GPGPU Performance" (MeDiC). [arXiv:1804.11038](https://arxiv.org/abs/1804.11038)

## Test plan (for future)
- New kernel: deliberately memory-divergent — half the threads
  thrash an L1-D, half stay cache-friendly.
- Bench: L1-D hit-rate per warp, cycle count vs. a "no-MeDiC" baseline.
- Ablation: each of MeDiC's three components on/off independently.

## Reproducibility
No code changes to run in this revision.
