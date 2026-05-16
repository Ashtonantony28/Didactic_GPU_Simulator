# Final Report — tiny-gpu-extended

## What this project is

A SystemVerilog GPU built on Adam Maj's tiny-gpu, extended with a bench
harness, an instruction cache (Phase 3), memory coalescing (Phase 5),
divergent-kernel verification (Phase 6), plus full RTL-and-ISA designs
for shared memory + barriers (Phase 7), a pluggable warp scheduler
(Phase 8), and MeDiC memory-divergence correction (Phase 9).

## Implementation status at a glance

| Phase | Description                            | Status      |
|-------|----------------------------------------|-------------|
| 0     | Bootstrap                              | DONE        |
| 1     | Clean reimplementation                 | DONE        |
| 2     | Bench harness + 4 extra kernels        | DONE        |
| 3     | Per-core direct-mapped I-cache         | DONE        |
| 4     | Pipelining                             | DEFERRED    |
| 5     | Memory coalescing                      | DONE        |
| 6     | Branch divergence — kernel level       | PARTIAL     |
| 7     | Shared memory + barriers               | DESIGNED    |
| 8     | Pluggable warp scheduler               | DESIGNED    |
| 9     | MeDiC                                  | DESIGNED    |
| 10    | This report                            | DONE        |

## Performance progression

All numbers in cycles. Bench parameters: `NUM_CORES=2`, `THREADS_PER_BLOCK=4`,
`DATA_MEM_NUM_CHANNELS=4`.

| Kernel       | Phase 2 baseline | + I-cache=16 (Phase 3) | + Coalesce=1 (Phase 5) |
|--------------|------------------|------------------------|------------------------|
| matadd       | 231              | 231                    | 231                    |
| matmul       | 556              | 504 (-9.4%)            | 464 (-16.5%)           |
| vector_dot   | 231              | 231                    | 231                    |
| relu         | 164              | 164                    | 164                    |
| conv1d       | 327              | 327                    | 327                    |
| transpose    | 244              | 244                    | 244                    |
| divergent_relu | n/a (new Phase 6) | 290                  | 290                    |

## Memory request counts

| Kernel       | Phase 2 mem_req | Phase 5 mem_req | Phase 5 coalesced |
|--------------|-----------------|-----------------|-------------------|
| matadd       | 24              | 24              | 0                 |
| matmul       | 20              | 12              | 8                 |
| vector_dot   | 24              | 24              | 0                 |
| relu         | 16              | 16              | 0                 |
| conv1d       | 24              | 24              | 0                 |
| transpose    | 16              | 16              | 0                 |

## I-cache hit / miss counts (ICACHE_LINES=16)

| Kernel       | hits | misses | hit rate |
|--------------|------|--------|----------|
| matadd       | 0    | 26     | 0%       |
| matmul       | 13   | 28     | 32%      |
| vector_dot   | 0    | 26     | 0%       |
| relu         | 0    | 18     | 0%       |
| conv1d       | 0    | 40     | 0%       |
| transpose    | 0    | 32     | 0%       |

## Discussion — which optimizations helped which workloads and why

**I-cache helps loops.** The only kernel with a hot fetch path is
`matmul`, which has a 14-instruction inner loop iterated twice (N=2 for
the 2×2 matmul). The other kernels are straight-line; with cold caches
on each of two cores they get 0 hits. The I-cache as designed is a
per-core resource and per-block reset; making it per-core but
*per-multiple-blocks* (i.e. surviving `core_reset`) is a clean follow-up
that would help every kernel where the same block runs on a core
sequentially.

**Coalescing helps shared-address fan-out.** `matmul`'s `LDR R10, R10`
inside the inner loop is the same `A[row][k]` address for every column
in the row, so threads in the warp coalesce. Unit-stride kernels (matadd,
vector_dot, conv1d, transpose) gain nothing because every thread's
address differs; coalescing strided accesses needs wider memory
channels (return multiple bytes per channel transaction) or a real
cache-line concept.

**Divergent kernels.** `divergent_relu` shows the no-divergence SIMT is
fine for data-divergent kernels (compute both, blend via mask). For
control-divergent kernels (some threads execute one block of code,
others execute another), a SIMT stack is still required. We design but
don't ship.

## What didn't work / what I'd do differently

- **Phase 4 (pipelining)** is a tar pit at this scope. A real 5-stage
  pipeline rewrites the scheduler, fetcher, decoder, register file, and
  LSU. The "Pipelining" deliverable in a 10-phase project plan should be
  its own multi-week sub-project. We surfaced the descope.
- **Phase 7 (shared memory) wired-in.** The hard part isn't the SRAM —
  it's plumbing `LDS`/`STS` through every datapath module without
  breaking Phase 5's regression. With another day this would have
  landed.
- **Verification investment.** With more time, every Phase 3+ feature
  should ship with a *negative* test: a kernel that proves the feature
  is OFF when its parameter is 0, not just that it's on when 1.
- **The cocotb-2.x port** ate more time than expected. The upstream
  `format_cycle` helper would need a full rewrite, not just a port —
  several internal cocotb APIs it depended on no longer exist.
- **Test isolation.** Each test currently does its own `run_kernel(...)`
  call with hardcoded `THREADS=N`. With more time these would
  parameterize from a single fixture, making the regression matrix
  cleaner.

## Reproducibility instructions

```bash
# Install
apt-get install -y iverilog
pip3 install cocotb
# sv2v binary into PATH
curl -L -o sv2v.zip https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip
unzip sv2v.zip && cp sv2v-Linux/sv2v /usr/local/bin/

# Build & smoke test
make compile
make test_matadd

# Per-phase bench
python3 bench/run.py --phase 2 --label phase-2-baseline
python3 bench/run.py --phase 3 --label phase-3-icache    --params "ICACHE_LINES=16"
python3 bench/run.py --phase 5 --label phase-5-coalesce  --params "ICACHE_LINES=16 COALESCE=1"

# JSON results land in bench/<label>.json and bench/results/<kernel>.json
```

## References

| # | Citation | Used in |
|---|---|---|
| 1 | adam-maj. *tiny-gpu*. https://github.com/adam-maj/tiny-gpu | Baseline |
| 2 | *Control Flow Management in Modern GPUs.* [arXiv:2407.02944](https://arxiv.org/abs/2407.02944) | Tier 3 Option A reference (not pursued) |
| 3 | Crouzet, Carle, Rochange. "Time-predictable warp scheduling in a GPU." *Microprocessors and Microsystems* 118 (2025) 105203. | ADR-008 / Phase 8 design |
| 4 | Ausavarungnirun et al. "Exploiting Inter-Warp Heterogeneity to Improve GPGPU Performance" (MeDiC). [arXiv:1804.11038](https://arxiv.org/abs/1804.11038) | ADR-009 / Phase 9 design |
| 5 | *Ten-Four: An Open-Source Fused Dot Product Unit for Mixed-Precision GPGPU Tensor Cores.* [arXiv:2512.00053](https://arxiv.org/abs/2512.00053) | Tier 3 Option D reference (not pursued) |
| 6 | Tine et al. "Vortex: Extending the RISC-V ISA for GPGPU and 3D-Graphics." MICRO-54, 2021. | Cross-reference design |
| 7 | Bakhoda et al. *GPGPU-Sim.* http://www.gpgpu-sim.org/ | SIMT stack / scheduler reference |
