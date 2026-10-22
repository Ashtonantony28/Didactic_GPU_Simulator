# Final Report — tiny-gpu-extended

## What this project is

A SystemVerilog GPU built on Adam Maj's tiny-gpu. Every extension is a `gpu` parameter whose default
reproduces the Phase 1 baseline, and every one is wired into the top level and tested:

| Feature | Parameter | Built in | Design |
|---|---|---|---|
| Per-core direct-mapped I-cache | `ICACHE_LINES` | Phase 3 | ADR-004 |
| Combinational decode, DECODE state skipped | `PIPELINED` | Phase 4 | ADR-005 |
| Same-address read coalescing | `COALESCE` | Phase 5 | — |
| Shared memory, `LDS`/`STS`/`BAR` | always on (`core.SMEM_SIZE` = 64) | Phase 7 | ADR-007 |
| L1-D (per GPU, write-through) | `DCACHE_LINES` | Phase 9 | ADR-009 |
| MeDiC bypass and controller priority | `MEDIC` | Phase 9, fixed in SG19 | ADR-009 |
| 32-bit datapath for the transformer kernels | `WORD_BITS` | SG2 | ADR-010 |
| Parallel memory channels, exact counters | — | SG18 | ADR-013 |
| Multi-warp core, warp scheduler, real `BAR` | `WARPS_PER_CORE`, `WARP_POLICY` | SG20 | ADR-008, ADR-012, ADR-014 |
| SIMT-stack divergence (`SYNC`) | `SIMT_STACK` | SG21 | ADR-006, ADR-011, ADR-015 |

`xf/isasim.py` is an instruction-level reference model; `test_isa_diff` compares the RTL against it on
every kernel and on random programs, and every phase-11 kernel row below was checked against it in the same
run (the MeDiC rows against `test_medic`'s own expected results).

## How the numbers are produced

`python3 bench/run.py --phase 11 --label phase-11-all` writes `bench/phase-11-all.json`; two runs give
identical files. Each row is a kernel run on one build at one memory latency, keyed
`<section>/<build>/<kernel>/L<latency>`; the file's `builds` map gives each build's parameters.
Latency `L` is the extra cycles a memory channel holds each transaction (`MEM_LATENCY` in the tests).
Unless a section says otherwise it applies to both program and data memory.

Citations below are `bench/<file>#<key>`; `{a,b}` cites one row per alternative.
`python3 bench/run.py --check-citations` (also `xf/tests/test_bench_citations.py`) fails if any cited row
does not exist. Files other than `phase-11-all.json` and `xf-*.json` are frozen per-phase history from
before SG18.

## Features on the regression kernels

matmul (2x2, 4 threads; the most loop-heavy of the eight) in cycles:

| Build | L0 | L3 | L10 |
|---|---:|---:|---:|
| default | 481 | 619 | 941 |
| `ICACHE_LINES=16` | 429 (−10.8%) | 528 (−14.7%) | 759 (−19.3%) |
| `COALESCE=1` | 481 (0) | 619 (0) | 941 (0) |
| `PIPELINED=1` | 440 (−8.5%) | 578 (−6.6%) | 900 (−4.4%) |
| `DCACHE_LINES=16` | 491 (+2.1%) | 629 (+1.6%) | 951 (+1.1%) |
| `DCACHE_LINES=16 MEDIC=1` | 491 (+2.1%) | 629 (+1.6%) | 951 (+1.1%) |
| all five | 398 (−17.3%) | 497 (−19.7%) | 728 (−22.6%) |

Source: `bench/phase-11-all.json#kernels/{default,icache,coalesce,pipelined,dcache,medic,all}/matmul/L{0,3,10}`.

Every kernel, default against all five features:

| Kernel | default L0 | all L0 | default L10 | all L10 |
|---|---:|---:|---:|---:|
| matadd | 168 | 161 (−4.2%) | 419 | 421 (+0.5%) |
| matmul | 481 | 398 (−17.3%) | 941 | 728 (−22.6%) |
| vector_dot | 168 | 161 (−4.2%) | 419 | 421 (+0.5%) |
| relu | 119 | 114 (−4.2%) | 293 | 294 (+0.3%) |
| conv1d | 250 | 238 (−4.8%) | 635 | 654 (+3.0%) |
| transpose | 196 | 184 (−6.1%) | 503 | 504 (+0.2%) |
| divergent_relu | 240 | 224 (−6.7%) | 623 | 624 (+0.2%) |
| tiled_matmul | 373 | 348 (−6.7%) | 713 | 688 (−3.5%) |

Source: `bench/phase-11-all.json#kernels/{default,all}/{matadd,matmul,vector_dot,relu,conv1d,transpose,divergent_relu,tiled_matmul}/L{0,10}`.

Counters at L0 (requests are downstream data transactions, contract 12):

| Kernel, build | data requests | coalesced | I-cache hit/miss | L1-D hit/miss |
|---|---:|---:|---:|---:|
| matmul, default | 20 | 0 | — | — |
| matmul, `COALESCE=1` | 12 | 8 | — | — |
| matmul, `ICACHE_LINES=16` | 20 | 0 | 13/28 | — |
| matadd, `ICACHE_LINES=16` | 24 | 0 | 0/26 | — |
| conv1d, default | 24 | 0 | — | — |
| conv1d, `DCACHE_LINES=16` | 14 | 0 | — | 10/8 |
| conv1d, `DCACHE_LINES=16 MEDIC=1` | 14 | 0 | — | 10/8 |

Source: `bench/phase-11-all.json#kernels/{default,coalesce,icache}/matmul/L0`,
`bench/phase-11-all.json#kernels/icache/matadd/L0`, `bench/phase-11-all.json#kernels/{default,dcache,medic}/conv1d/L0`.

- **I-cache** is the largest single win and grows with latency, because it removes program-memory
  transactions; only loops hit (matmul 13 hits, the straight-line kernels none).
- **Coalescing** cuts matmul's requests from 20 to 12 but saves no cycles; since SG18 the four data
  channels serve distinct consumers in the same cycle (ADR-013).
- **`PIPELINED`** saves one cycle per instruction (41 for matmul); its share shrinks as latency grows.
- **L1-D** lowers conv1d's requests (24 to 14) but adds a lookup cycle to every access and, being
  write-through, never removes a store: it costs 4–10 cycles per kernel at L0
  (`bench/phase-11-all.json#kernels/{default,dcache}/{matadd,matmul,vector_dot,relu,conv1d,transpose,divergent_relu,tiled_matmul}/L0`)
  and conv1d is slower even at L10 (`bench/phase-11-all.json#kernels/{default,dcache}/conv1d/L10`: 635 vs
  660). With all five features on, six of the eight kernels are slightly slower than default at L10.
- **MeDiC** at `MEDIC=1` is cycle- and count-identical to plain `DCACHE_LINES=16` on all eight kernels
  (`bench/phase-11-all.json#kernels/{dcache,medic}/{matadd,matmul,vector_dot,relu,conv1d,transpose,divergent_relu,tiled_matmul}/L{0,3,10}`):
  their lanes never demote, so nothing is bypassed. Its effect needs the mixed workload below.

## Multi-warp latency hiding

Kernels `stream` (each of 32 threads sums 4 loads, one ALU instruction per load) and `poly` (8 ALU
instructions per load), on 2 cores. Each warp has 4 lanes, so `THREADS_PER_BLOCK = 4 · WARPS_PER_CORE`
and every build has the same lanes and the same 288 (`stream`) / 480 (`poly`) issued instructions; W=1 runs
8 blocks, W=4 runs 2.

Data-memory latency only (program memory at L0, no I-cache), cycles:

| Build | stream L0 | L3 | L10 | poly L0 | L3 | L10 |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 1703 | 1804 | 1921 | 2759 | 2860 | 2977 |
| W2 LRR | 1941 | 1941 | 1990 | 3093 | 3093 | 3142 |
| W2 GTO | 1941 | 1941 | 1941 | 3093 | 3093 | 3093 |
| W2 GTLRR | 1941 | 1941 | 1941 | 3093 | 3093 | 3093 |
| W4 LRR | 1938 | 1938 | 1992 | 3090 | 3090 | 3144 |
| W4 GTO | 1938 | 1938 | 1938 | 3090 | 3090 | 3090 |
| W4 GTLRR | 1938 | 1938 | 1938 | 3090 | 3090 | 3090 |

Source: `bench/phase-11-all.json#warps/{w1,w2lrr,w2gto,w2gtlrr,w4lrr,w4gto,w4gtlrr}/{stream,poly}/L{0,3,10}`.

Load latency is hidden: L10 costs W1 218 cycles on both kernels, LRR 49–54, and GTO/GTLRR nothing. But the
multi-warp issue slot (ADR-014: an `IDLE` cycle per instruction plus a re-issue after each yielded load or
store) costs 12–14% at L0, more than the 218 cycles saved, so W1 is still fastest here.

Both memories at latency L, with `ICACHE_LINES=16`:

| Build | stream L0 | L3 | L10 | poly L0 | L3 | L10 |
|---|---:|---:|---:|---:|---:|---:|
| W1 | 1247 | 1434 | 1928 | 2087 | 2478 | 3539 |
| W2 LRR | 1373 | 1463 | 1662 (−13.8%) | 2213 | 2359 | 2693 (−23.9%) |
| W2 GTO | 1373 | 1457 | 1723 (−10.6%) | 2213 | 2368 | 2879 (−18.6%) |
| W2 GTLRR | 1373 | 1457 | 1723 | 2213 | 2368 | 2879 |
| W4 LRR | 1354 | 1424 | 1590 (−17.5%) | 2146 | 2234 | 2442 (−31.0%) |
| W4 GTO | 1354 | 1425 | 1656 (−14.1%) | 2146 | 2241 | 2556 (−27.8%) |
| W4 GTLRR | 1354 | 1425 | 1656 | 2146 | 2241 | 2556 |

Source: `bench/phase-11-all.json#warps_icache/{w1,w2lrr,w2gto,w2gtlrr,w4lrr,w4gto,w4gtlrr}/{stream,poly}/L{0,3,10}`.

Here multi-warp wins at L10 on both kernels and already at L3 on `poly`. Part of the gain is the
I-cache, which the warps of a block share: W4 takes 36 / 48 I-cache misses where W1, running 8 blocks,
takes 60 / 144 (`bench/phase-11-all.json#warps_icache/{w1,w4lrr}/{stream,poly}/L10`). With fetch latency in
play LRR beats GTO by 3.5–6.5% at L10 at the same miss counts; with data latency alone GTO is the better
policy. GTLRR equals GTO in every row: as implemented (ADR-008) it differs from GTO only in the fallback
order after a stall. No row has barrier stalls (these kernels have no `BAR`).

## SIMT stack

8 threads, L0:

| Kernel | `SIMT_STACK=0` | `SIMT_STACK=1` | issued |
|---|---:|---:|---:|
| divergent_relu (mask blend, no branch) | 240 | 240 | 40 |
| branch_relu (same result, if/else with `SYNC`) | — | 229 | 38 |
| uniform_loop (every lane 4 trips) | 356 | 356 | 62 |
| lane_loop (lane trips 1–4, `SYNC` each trip) | — | 389 | 68 |

Source: `bench/phase-11-all.json#simt/{simt0,simt1}/{divergent_relu,uniform_loop}/L0`,
`bench/phase-11-all.json#simt/simt1/{branch_relu,lane_loop}/L0`.

The stack costs no cycles on block-uniform code (ADR-015 commits it at `UPDATE` without an extra state).
The if/else version issues 38 instructions against the mask blend's 40 and is 4.6% faster; the warp runs
both sides, each under its lane mask. A divergent loop runs until its longest lane finishes: `lane_loop`
issues the 4-trip body like `uniform_loop` and costs 9.3% more only through its 3 extra instructions per
block that compute the trip counts (68 vs 62 issued over 2 blocks). The branchy kernels give wrong results
without the stack, so they have no `SIMT_STACK=0` row.

## MeDiC on a mixed workload

`test/test_medic.py`: per block, 2 hot lanes re-read one address and 2 streaming lanes read a new line
every iteration, 7 iterations, `DCACHE_LINES=16`; data memory is L + 4, program memory L.

| | `MEDIC=0` | `MEDIC=1` |
|---|---:|---:|
| hot-lane hits / reads | 14 / 28 | 18 / 28 |
| all L1-D hits / misses | 14 / 42 | 18 / 38 |
| streaming lane installs / bypasses (each) | 7 / 0 | 2 / 5 |
| data requests | 98 | 94 |
| cycles L0 / L3 / L10 | 784 / 1075 / 1817 | 781 / 1071 / 1813 |

Source: `bench/phase-11-all.json#medic/{medic0,medic1}/L{0,3,10}`.

Bypassing the streaming lanes after their first two installs stops them evicting the hot lines (hot-lane
hit rate 50% to 64%) and saves 4 transactions. The cycle gain is 3–4 cycles: each load waits for all
lanes of its block, the streaming lanes miss either way, and controller priority only matters when the
two cores' requests overlap.

## Transformer kernels (xf track)

The int8/int32 kernels of `xf/` (attention head and SwiGLU MLP of a nanoGPT model) run at `WORD_BITS=32`
and are bit-exact against `xf.golden`; `docs/xf-bench.md` has the full tables. Example: a 4x8x16 matmul
takes 10543 cycles naive and 8325 tiled through shared memory, and with `COALESCE=1 DCACHE_LINES=16` the
tiled kernel issues 275 data requests against 678
(`bench/xf-matmul.json#4x8x16/{naive,tiled}/{default,COALESCE=1+DCACHE_LINES=16}`; spaces in a build
are written `+`).

## Earlier claims, corrected

These numbers came from frozen bench files of the phase named and no longer describe the design:

- **Coalescing −16.5% on matmul** (Phase 5): 556 → 464 cycles
  (`bench/phase-2-baseline.json#matmul`, `bench/phase-5-coalesce.json#matmul`) measured I-cache and
  coalescing together on a controller that let every idle channel take the same consumer (ADR-013). After
  SG18 coalescing alone saves no cycles on matmul (above).
- **"MeDiC correctly bypasses conv1d"** (Phase 9): 0 hits / 18 misses and 335 cycles
  (`bench/phase-9-medic.json#conv1d`) was MeDiC never installing a line: every lane started low-utility,
  so `MEDIC=1` acted as a cache-off switch. Fixed in SG19 (ADR-009); `MEDIC=1` now matches plain L1-D
  on conv1d and helps the mixed workload above.
- **Warp scheduler "not wired"** (Phase 8): `warp_scheduler.sv` was a standalone module with its own
  testbench. Since SG20 it picks the issuing warp in every core at `WARPS_PER_CORE > 1`.
- **SIMT stack "design only"** (Phase 6, ADR-006): implemented in SG21 behind `SIMT_STACK`, with `SYNC`
  (`BR` with `nzp = 000`) as the reconvergence marker instead of the reserved opcode `1101`, which became
  `SRA`.
- **`BAR` "a free no-op"** (Phase 7): still a no-op at `WARPS_PER_CORE = 1`; a real barrier since SG20.
- Every Phase 2–10 cycle count predates SG18's controller fix; every regression kernel's default count
  changed (matmul 556 → 481, matadd 231 → 168:
  `bench/phase-2-baseline.json#{matmul,matadd}`, `bench/phase-11-all.json#kernels/default/{matmul,matadd}/L0`).

## Limitations

- One front end per core: one instruction in flight, so multi-warp hides load latency but not fetch
  latency, and pays an issue-slot cost at low latency (above).
- GTLRR is not the coordinated fetch/issue scheme of Crouzet et al.; it is greedy with a round-robin
  fallback (ADR-008).
- The L1-D adds a cycle per access and is write-through; MeDiC classifies per lane (LSU), not per warp.
- Data memory is 256 words at 8 bits (4096 at 32 bits); the 8-bit PC limits programs to 256 words.

## Reproducibility

```bash
apt-get install -y iverilog                       # 12.0
pip install cocotb==2.0.1 numpy pytest torch      # cocotb 2.1.x breaks the Makefile
# sv2v 0.0.13 into /usr/local/bin; nanoGPT Extended @ 03dbab8 in $NANOGPT_DIR for test_all

make regression
NANOGPT_DIR=... make test_all
make -j4 matrix
python3 bench/run.py --phase 11 --label phase-11-all   # about 100 s on 4 cores
python3 bench/run.py --check-citations
python3 bench/run.py --xf --label xf-matmul            # and xf-mlp; regenerates docs/xf-bench.md
```

## References

| # | Citation | Used in |
|---|---|---|
| 1 | adam-maj. *tiny-gpu*. https://github.com/adam-maj/tiny-gpu | Baseline |
| 2 | *Control Flow Management in Modern GPUs.* [arXiv:2407.02944](https://arxiv.org/abs/2407.02944) | ADR-006 |
| 3 | Crouzet, Carle, Rochange. "Time-predictable warp scheduling in a GPU." *Microprocessors and Microsystems* 118 (2025) 105203. | ADR-008 |
| 4 | Ausavarungnirun et al. "Exploiting Inter-Warp Heterogeneity to Improve GPGPU Performance" (MeDiC). [arXiv:1804.11038](https://arxiv.org/abs/1804.11038) | ADR-009 |
| 5 | Bakhoda et al. *GPGPU-Sim.* http://www.gpgpu-sim.org/ | SIMT-stack reference |
