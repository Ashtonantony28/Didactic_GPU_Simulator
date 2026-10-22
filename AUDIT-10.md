# AUDIT-10 — Performance under load

Lens: where the design or the simulator falls over, deadlocks, starves or blows the time budget when pushed. In
scope:
- memory-channel contention and utilisation at `NUM_CORES=4`, `WARPS_PER_CORE=4` under each `WARP_POLICY`, with
  and without `COALESCE`, `DCACHE_LINES`, `MEDIC`, `ICACHE_LINES` and `SIMT_STACK`, at memory latency 0, 3 and 10;
- large launches (16 blocks per launch, `THREADS_PER_BLOCK` 16 = `TILED_MAX_BLOCK` and 64), long loops (more than
  255 issues per warp), and divergent kernels under the SIMT stack at latency 10;
- simulator throughput against the PLAN SG17 baseline and budget, and `xf/isasim.py` time per instruction;
- `make -j4 matrix` wall time, per configuration and per test module. I did not run the full matrix.

I checked against PLAN contracts 5, 9 and 12, the latency budget under SG17, ADR-012 to ADR-015,
`docs/architecture.md` and `docs/final-report.md`. Audited at `1313fb6` (main) with iverilog 12.0, sv2v 0.0.13
and cocotb 2.0.1, on 4 cores with another audit running, and with at most two jobs of my own at a time. All builds
went under `build/a10/` and `build/a10-pytest/`.

**Normalisation.** This container runs slower than the one the SG17 baseline was recorded on. The brief gives
`isa_diff` at 2880 cycles/s here against 4280 then, a factor of 0.673. Throughput figures marked "normalised" are
divided by 0.673. Wall times are as measured.

Method:
- **Probes.** `test/test_audit10.py` is a cocotb module of load probes:
  - `dispatch_spread`: 16 blocks of a compute loop; counts the blocks each core runs.
  - `data_fairness`: one block per core of a load loop; records each core's finish cycle.
  - `scaling`: 16 blocks of a compute loop and of a load loop; reports cycles and channel occupancy.
  - `gto_long_run_fairness`: one block of 4 warps, 409 issues per warp; records each warp's finish cycle.
  - `counters_under_load`: two blocks per core of loads, SMEM and a `BAR` each trip; checks memory and
    `warp_issues_total` against `xf.isasim`, plus the runner's `check_counters`.
  - `l1d_stale_fill`: a unit probe on an `l1d_cache` top.

  I ran them on 20 builds: `NUM_CORES` 1, 2 and 4; `WARPS_PER_CORE=4` under LRR, GTO and GTLRR; the
  `COALESCE`/`DCACHE`/`MEDIC` combinations; everything on at 4 cores and 4 warps; `THREADS_PER_BLOCK` 16 and 64. Each
  build ran at latency 0 and 10, and 3 where it mattered.
- **Existing suites under load.** `test_divergence` (`DV_w4c4` and `DV_t8w2`) at `MEM_LATENCY=10`;
  `test_multiwarp` at `NUM_CORES=4`, `MEM_LATENCY=10` under each policy.
- **Throughput.** `make test_isa_diff` / `make test_xf_isa_diff` for each of the 13 matrix configurations. Cycles/s
  is the total simulated time in `results.xml` over the summed per-test wall time. I also timed every module of
  `matrix_run` on the default build, and `test_xf_rmsnorm` at `WARPS_PER_CORE=2`.
- **History.** The `real` lines of `/home/claude/matrix-*.log` and `/home/claude/testall-*.log`: the session logs
  of the SG runs, outside the repository.
- **Model.** I timed `xf.isasim.run` on every fixture under `lrr`, `gto` (W=4) and `simt`, on 49k-step loops at 4,
  16 and 255 threads per block, and up to the 1M-step cap.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | `make test_isa_diff` / `make test_xf_isa_diff`, 13 configurations (`BUILD=build/a10/tp_<cfg>`) | all pass; cycles/s below |
  | each `matrix_run` module on the default build | all pass; times in M3 |
  | `MEM_LATENCY=10 make divergence_w4c4` / `divergence_t8w2` (`BUILD=build/a10/dv`) | 6/6 and 6/6 pass (4 m 41 s, 2 m 12 s) |
  | `MEM_LATENCY=10 make multiwarp_w4_{lrr,gto,gtlrr} PARAMS=-Pgpu.NUM_CORES=4` (`BUILD=build/a10/mw`) | 5/6 each: `test_age_saturation` times out (L2) |
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` | 756 passed, 14 xfailed (4 of them new) |
  | `python3 bench/run.py --check-citations` | `citations: OK` |

There are **no critical or high findings**, so this audit adds no SG rows. Nothing hung. Every probe run passed
`check_counters`, and `warp_issues_total` equalled the model's steps in every configuration (contract 12).

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| M1 | Medium | `NUM_CORES=4` does not scale past 2 without the I-cache. The one program channel saturates, and both controllers claim consumers in fixed index order, so the low cores starve the high ones. 16 blocks run as `[5, 5, 5, 1]` per core at L0 and `[7, 7, 1, 1]` at L3/L10. 4 cores are 6% faster than 2 at L0 and no faster at L3/L10. At 4 warps × 4 cores, `COALESCE`+`DCACHE`+`MEDIC` cut requests from 224 to 47 for 0 cycles | `test_audit10.py::test_four_cores_share_the_blocks`; control `::test_four_cores_share_the_blocks_with_icache` |
| M2 | Medium | Shared L1-D coherence under contention. When a miss fill and another lane's write-through to the same address complete in the same cycle, the write-through's hit check misses the line being installed. The fill then installs the pre-write value, and later reads hit it while memory holds the write | `test_audit10.py::test_l1d_fill_racing_a_write_is_not_stale` (unit probe on an `l1d_cache` top) |
| M3 | Medium | `make -j4 matrix` is over the PLAN's 60-minute budget. Every recorded run since SG17 took 54–82 min; my per-module timings put it at about 84 min here (56 normalised). The long pole is `matrix_warps2`: it is the only multi-warp configuration that runs the attention and MLP chains, at 2.4× the default cycles (about 44 min alone), and it starts 11th of 13 | `test_audit10.py::test_matrix_skips_long_chains_on_slow_configs` (structural); timings below |
| L1 | Low | GTO under long-running loops. Once warp ages saturate at 254 (ADR-014), every "oldest" tie goes to the lowest warp, so warps 0 and 1 ping-pong while 2 and 3 wait. At 80 trips (409 issues per warp) warps 2–3 finish 23% after warps 0–1; at 200 trips, 59%. Below saturation (40 trips) all four finish within 0.6%. The makespan is unchanged | `test_audit10.py::test_gto_stays_fair_past_age_saturation`; control `::test_lrr_long_run_is_fair` |
| L2 | Low | No matrix configuration runs multi-warp at memory latency: `latency` is W=1, and `warps2`/`warps4gto` run at L0. Under load `test_multiwarp::test_age_saturation` fails at `NUM_CORES=4`, `MEM_LATENCY=10`, under every policy. Its hard-coded 50 000-cycle timeout is shorter than the 75 134 cycles the kernel needs. It is not a hang: with a 500 000 timeout it passes | `MEM_LATENCY=10 make multiwarp_w4_gto BUILD=build/a10/mw PARAMS=-Pgpu.NUM_CORES=4` → `TimeoutError: Kernel did not complete in 50000 cycles` (about 40 s) |
| L3 | Low | Simulator throughput falls roughly linearly with `NUM_CORES × THREADS_PER_BLOCK` LSUs. At 8 LSUs it is 2180 cycles/s, at 64 (TPB 16 × 4 cores) 734, and 484 with `DCACHE`+`MEDIC` (2.07 s per 1k cycles, over the 2 s budget here, 1.39 s normalised). At 256 (TPB 64 × 4) it is 91 cycles/s (11 s per 1k, 7.4 s normalised). TPB 16 is the largest block the tiled kernels allow, and at 484 cycles/s `test_xf_attention` alone would take about 32 min | `scaling` probe per build (commands below) |

## M1 — Program channel and fixed-priority claims at 4 cores

`gpu.sv` gives the program controller `PROGRAM_MEM_NUM_CHANNELS = 1` channel for `NUM_CORES` fetchers. The
`controller.sv` `IDLE` scan claims the lowest-index valid consumer (`for (j = 0; ...) if (!found && ...)`), with
no rotation. Without the I-cache, every instruction of every core goes through that channel:

| `scaling` probe, 16 blocks | spin L0 | spin L3 | spin L10 | loads L0 | loads L10 | program valid, L0 / L10 |
|---|---:|---:|---:|---:|---:|---:|
| `NUM_CORES=1` | 4530 | 5778 | 8690 | 10978 | 21698 | 17% / 52% |
| `NUM_CORES=2` | 2271 | 3266 | 6059 | 5495 | 14363 | 34% / 78% |
| `NUM_CORES=4` | 2130 | 3303 | 6074 | 4733 | 14378 | 39% / 78% |
| `NUM_CORES=4 ICACHE_LINES=16` | 894 | — | 1203 | 2079 | 3126 | 7% / 29% |
| `NUM_CORES=4 W=4` LRR | 8945 | — | 24972 | 22381 | 64332 | 38% / 80% |
| `NUM_CORES=4 W=4` LRR, `COALESCE DCACHE MEDIC` | 8945 | — | 24972 | 22381 | 64332 | 38% / 80% |

"Program valid" counts the cycles the channel's valid is high. With the claim and relay cycles around each
transaction, the channel is saturated from 2 cores at L0. The data channels stay at 1–9% occupancy in every
I-cache-less row.

The fixed claim order decides who gets the saturated channel. `dispatch_spread` on `NUM_CORES=4`:

```
L0:  16 blocks, 2130 cycles, blocks per core [5, 5, 5, 1], issues per core [125, 125, 125, 25]
L3:  16 blocks, 3303 cycles, blocks per core [7, 7, 1, 1], issues per core [175, 175, 25, 25]
L10: 16 blocks, 6074 cycles, blocks per core [7, 7, 1, 1]
ICACHE_LINES=16, L0: [4, 4, 4, 4]; L10: [5, 5, 3, 3]
```

A core that finishes a block is reset and given the next block within a few cycles, and the next block's fetches
again win over the higher-index cores. The data controller has the same fixed order, over `NUM_CORES ×
THREADS_PER_BLOCK` LSUs. In `data_fairness` (one block per core), core 3 finishes last by a wide margin:
`[915, 920, 925, 1498]` cycles at L0, and `[3841, 3846, 4275, 6994]` at 4 warps.

Nothing hangs, and the makespan of a saturated channel would be the same under fair arbitration. The effects
are:
- the extra cores buy nothing, and individual blocks see starvation latencies of whole other blocks;
- data-side features show no cycle effect at 4 warps × 4 cores, because the core is waiting for fetches.

`docs/final-report.md` §Limitations names "one front end per core" but not the single program channel or the
fixed order. No document claims that 4 cores scale, so this is a performance finding rather than a broken claim.

Fix options:
- rotate the claim start (round-robin `j`) in `controller.sv` for both controllers;
- set `PROGRAM_MEM_NUM_CHANNELS` to `NUM_CORES`. `run_kernel` already takes `program_channels`;
  `run_launches` hard-codes 1;
- or state the limitation and recommend `ICACHE_LINES` at 4 cores.

## M2 — L1-D fill racing a write-through

`l1d_cache.sv` installs a miss in `L_MISS` when `mem_read_ready` arrives, with non-blocking writes to
`valid_arr`/`tag_arr`/`data_arr`. A write-through updates the line in `L_WT` only `if (valid_arr[idx] && tag_arr[idx]
== tag)` at the moment `mem_write_ready` arrives. Suppose the downstream read of X and another lane's write of X
are serviced in the same cycle: on two channels, at equal latency, with the memory model doing reads before
writes. Then the read returns the old value, and both readies reach the cache on the same edge. The write's hit
check sees the line still invalid and skips the update, and the fill installs the old value. Every later read of
X hits on it, while memory holds the new value. Any other order is safe:
- read serviced first: the line is installed first and the write-through then updates it;
- write serviced first: the read returns the new value.

The unit probe drives exactly that cycle:

```
l1d_stale_fill: second read of X returned 0 (hits 1, memory holds 7)
```

At the GPU level the window is one cycle. My system-level sweep of writer skews and latencies did not land on
it: an instruction costs about 11 cycles there, so no skew lines the two up. But the ordering is ordinary for
the controller: two consumers with valid high in the same cycle, on two idle channels. The racing first read is
itself a data race, between blocks or between unsynchronised warps. Even so, a value that stays stale after the
write has completed breaks what `BAR` promises at `WARPS_PER_CORE > 1`. Take a warp whose load raced another
warp's store: after a `BAR` it can still read the old value. Fix: in `L_MISS`, do not install, or install the
written data, when a write-through of the same index and tag completes in the same cycle. Alternatively, have
`L_WT` invalidate the line instead of conditionally updating it.

## M3 — Matrix wall time

Recorded `make -j4 matrix` runs:

| Log | `real` |
|---|---|
| `matrix-base.log` (before SG17) | 48 m 24 s |
| `matrix-sg17.log` | 46 m 15 s |
| `matrix-sg18.log` | 67 m 03 s |
| `matrix-sg19.log` | 81 m 28 s |
| `matrix-sg20.log` | 54 m 04 s |
| `matrix-sg21.log` | 72 m 32 s |
| `matrix-sg22.log` | 81 m 58 s |

PLAN says "`make -j4 matrix` ≤ 60 min" and "a sub-goal that slows any matrix configuration by > 20 % stops and
reports". PLAN line 440 still says "about 53 min".

Per-module wall time on the default build (this container):
- the 32-bit side, about 1096 s in all, dominated by `test_xf_attention` (362 s, 922k cycles),
  `test_xf_mlp_block` (353 s), `test_xf_matmul` (173 s), `test_xf_rmsnorm` (72 s), `test_xf_bench` (57 s),
  `test_xf_elementwise` (45 s) and `test_xf_isa_diff` (31 s);
- every 8-bit module under 9 s.

Measured ratios to default (wall time of `test_xf_isa_diff`; `test_xf_rmsnorm` for `warps2`):
- `warps2` 2.36 (169 s against 72 s);
- `cores1` 1.41, `latency` 1.25, `cores4` 1.19;
- `warps4gto` 3.9, on the reduced xf list;
- the rest 0.87–1.11.

A greedy 4-slot schedule in `MATRIX` order then gives about 276 CPU-minutes and an 84 min makespan (56 normalised).
`warps2` (about 44 min) starts at about 40 min. `warps4gto` and `simt` already drop the long chains (`XF_warps4gto`,
`XF_simt`); `warps2` does not. Fix:
- give `warps2` the same `XF_` filter, which saves about 30 CPU-min;
- or move `warps2` to the front of `MATRIX`, so the long pole starts first: the makespan becomes about
  max(44, 276/4) ≈ 69 min;
- and re-record the budget for the slower container.

## L1 — GTO fairness past age saturation

ADR-014 caps `warp_age` at 254, so that GTO's `<` search can always pick a warp. Ages count issues, and GTO's
"oldest" is the smallest age: until saturation, GTO favours the warp that has issued least. Once every warp
reaches 254, all ages tie, and `warp_scheduler.sv` keeps the first, the lowest index. `gto_long_run_fairness` on
`NUM_CORES=1 WARPS_PER_CORE=4` (one lane per warp, a load every 5 instructions), finish cycle per warp:

| Policy | 40 trips (209 issues) | 80 trips (409) | 200 trips (1009), 4 cores |
|---|---|---|---|
| LRR | — | 22887 … 22923 | 104898 … 104964 |
| GTO | 11651 … 11723 | **18603, 18627, 22899, 22923** | **35404, 35428, 56500, 56524** |
| GTLRR | — | 22851 … 22923 | 104832 … 104964 |

Long-running kernels (the xf chains run thousands of issues per warp) therefore get "lowest warp first" rather
than GTO. `xf.isasim`'s `"gto"` is "stay, then the lowest ready warp" from the start, so the model does not
show this either. Fix: saturate by halving all ages when one reaches 254 (this keeps their order), or document
that GTO degrades to lowest-index after 254 issues.

## L2 — Multi-warp is not run at latency in the matrix

`CFG_latency` is `MEM_LATENCY=3` on the default build (W=1). `warps2` and `warps4gto` run at L0, and nothing
runs `test_multiwarp` at `NUM_CORES=4`. The only latency coverage of the yield path is the phase-11 bench (2
cores, 3 kernels). Under `NUM_CORES=4 MEM_LATENCY=10`:
- `test_age_saturation`'s fixed `_run(..., 50_000)` times out under every policy;
- the kernel needs 75 134 cycles; it passes with a 500 000-cycle timeout (patched `_run`, scratch module);
- every other `test_multiwarp` test passes.

`test_divergence` at `MEM_LATENCY=10` passes on `w4c4` and `t8w2`, and every probe here completed. Fix:
derive the timeout from the model's steps (for example `steps × (12 + 2·latency) × NUM_CORES`), and add a W>1
latency configuration to the matrix, or a `MEM_LATENCY` axis to `test_multiwarp`.

## L3 — Throughput against LSU count

`scaling` probe, cycles/s (measured; the normalised figure divides by 0.673):

| Build | LSUs | cycles/s | s per 1k cycles (normalised) |
|---|---:|---:|---:|
| default (2 cores × 4) | 8 | 2180 | 0.46 (0.31) |
| `NUM_CORES=4` | 16 | 1701 | 0.59 (0.40) |
| `THREADS_PER_BLOCK=16` | 32 | 1276 | 0.78 (0.53) |
| `NUM_CORES=4 THREADS_PER_BLOCK=16` | 64 | 734 | 1.36 (0.92) |
| … `COALESCE=1` | 64 | 731 | 1.37 (0.92) |
| … `DCACHE_LINES=16 MEDIC=1` | 64 | 484 | **2.07** (1.39) |
| `NUM_CORES=4 THREADS_PER_BLOCK=64` | 256 | 91 | **11.0** (7.4) |

`controller.sv` and `l1d_cache.sv` are behavioural loops over every consumer in every cycle. With `MEDIC` the
controller scans twice per idle channel; with `COALESCE` there is an inner loop over consumers per claim. Each
thread also adds a register file, PC and LSU. No matrix configuration goes past 16 LSUs, so the matrix is not
affected. A TPB-16 xf run at 4 cores with the L1-D would be about 3.4× slower than default. Fix (if wanted):
`break` out of the claim scan once found, or keep a ready-consumer mask. Otherwise, record a throughput floor
per LSU count in the budget.

## Running the probes

Without the Makefile (any build directory):

```bash
B=build/a10/c4; mkdir -p $B
sv2v -w $B/raw.v $(ls src/*.sv | grep -v warp_scheduler_tb) && (printf '`timescale 1ns/1ns\n'; cat $B/raw.v) > $B/gpu.v
iverilog -o $B/sim.vvp -s gpu -g2012 -Pgpu.NUM_CORES=4 $B/gpu.v
PYGPI_PYTHON_BIN=$(which python3) LIBPYTHON_LOC=$(cocotb-config --libpython) AUDIT10_LATENCY=3 \
  COCOTB_TEST_MODULES=test.test_audit10 COCOTB_TEST_FILTER='dispatch_spread$' \
  vvp -M $(cocotb-config --lib-dir) -m libcocotbvpi_icarus $B/sim.vvp
# L1-D unit probe: sv2v src/l1d_cache.sv, then
#   iverilog -s l1d_cache -Pl1d_cache.NUM_CONSUMERS=2 -Pl1d_cache.LINES=4 ..., COCOTB_TEST_FILTER='l1d_stale_fill$'
```

`xf/tests/test_audit10.py` does the same for M1, M2 and L1 (about 25 s; skipped without the toolchain). Its
build output goes to `build/a10-pytest/`. `make test_audit10` also works: at the default parameters every
probe passes or returns early, so `make test_all` stays green.

## Checked and consistent

- **No hangs or deadlocks.** None on:
  - 4 warps × 4 cores under LRR, GTO and GTLRR at L0/L10;
  - everything on at once (`ICACHE`, `PIPELINED`, `COALESCE`, `DCACHE`, `MEDIC`, `SIMT_STACK`, 4 warps, 4 cores);
  - TPB 16 at 1 and 4 warps, and TPB 64;
  - 16-block launches and loops of 1000+ issues per warp;
  - `test_divergence` at L10;
  - `BAR` every trip under load.

  The warp-age saturation of ADR-014 holds (L1 is fairness, not a hang).
- **Contract 12 under load.** `check_counters` passed on every probe run. `data_mem_requests` equalled the
  serviced reads and writes, `…_coalesced` equalled the handshakes minus reads, and the I-cache totals equalled
  the per-core sums. `warp_issues_total` equalled `xf.isasim` steps in all 22 `counters_under_load` runs
  (464 at W=1, 1856 at W=4, any policy, any feature set). `bar_stall_cycles_total` was 0 at W=1. At W=4,
  GTO = GTLRR exactly, as the final report says.
- **Throughput against the SG17 baseline.** No configuration is more than 20% below its normalised baseline. The
  differential harness stays far inside 2 s per 1k cycles (0.27–0.54 s here):

  | Config | measured 8 / 32-bit | normalised | SG17 baseline | change |
  |---|---|---|---|---|
  | default | 2698 / 2506 | 4010 / 3724 | 4280 / 4060 | −6% / −8% |
  | icache | 2986 / 2580 | 4438 / 3834 | 4290 / 4110 | +3% / −7% |
  | pipelined | 2853 / 2445 | 4240 / 3634 | 4330 / 3790 | −2% / −4% |
  | coalesce | 2905 / 2608 | 4317 / 3876 | 4170 / 3930 | +4% / −1% |
  | dcache | 2603 / 2484 | 3868 / 3692 | 3340 / 3920 | +16% / −6% |
  | medic | 1968 / 2300 | 2925 / 3418 | 3220 / 3350 | −9% / +2% |
  | all | 2256 / 2245 | 3353 / 3336 | 3570 / 3060 | −6% / +9% |
  | cores1 | 3310 / 3235 | 4919 / 4808 | 4700 / 4370 | +5% / +10% |
  | cores4 | 1866 / 1992 | 2773 / 2960 | 3340 / 3140 | **−17%** / −6% |
  | latency | 3040 / 2780 | 4518 / 4131 | 4090 / 4060 | +10% / +2% |

  These are single runs alongside another audit, so ±10% is noise. `cores4` at −17% is the one to watch (see L3).
  The configurations added after SG17, which have no baseline: `warps2` 2610 / 2594, `warps4gto` 2961 / 2764,
  `simt` 2830 / 2628.
- **`xf.isasim`.**
  - 3.2 µs per instruction on average over every fixture, under `lrr`, W=4 `gto` and `simt` (83 474 steps in
    0.27 s);
  - the worst per-step cost on a non-trivial run is 17 µs, from fixed per-run setup on 6-step programs;
  - 2.6–3.8 µs per step on 49k-step loops at 4–16 threads per block, and 85 µs at 255 threads per block (the
    lane list is rebuilt each step);
  - the 1M-step cap trips in 3.0 s;
  - the model is about 100× cheaper than the RTL per instruction, and never on the critical path.
- **Data channels.** With the I-cache on, 4 cores keep the data channels 11% (L0) to 44% (L10) occupied. No
  channel served duplicate transactions: `check_counters` passed throughout (ADR-013).
