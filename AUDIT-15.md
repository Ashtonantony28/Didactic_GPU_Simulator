# AUDIT-15 — Performance under load, round 4

Lens: where the design or the simulator falls over, deadlocks, starves or blows the time budget when pushed. This
round:
1. re-measures AUDIT-10's findings on current main, after:
   - SG25: the MeDiC priority pass now covers reads only; the memory model applies a cycle's writes in channel
     order; runner traces replay at more than one warp;
   - SG24 and SG26: new tests and fixtures, including the 253/255-thread launches;
   - SG27: the PARAMS checks;
2. checks that none of these regressed cycles, simulator throughput or `make -j4 matrix` wall time;
3. pushes the design further than AUDIT-10 did, looking for hangs, starvation, deadlock and counter
   inconsistencies (contract 12):
   - `NUM_CORES=4` × `WARPS_PER_CORE=4` under each policy, with `SIMT_STACK`, `MEDIC`+`COALESCE`+`DCACHE_LINES`
     and `MEM_LATENCY=10`, with and without the I-cache;
   - divergent kernels with `BAR` under contention;
   - the largest launches (253–255 threads);
   - loops past the warp-age saturation point (254 issues per warp).

I checked against PLAN contracts 5, 9 and 12, the SG17 latency budget, ADR-009 and ADR-012 to ADR-016,
`docs/architecture.md`, `docs/final-report.md` and AUDIT-9 to AUDIT-14 and AUDIT-16. Audited at `290f041` (main),
with iverilog 12.0, sv2v 0.0.13 and cocotb 2.0.1, on 4 cores. The machine was otherwise idle, and I ran at most four
simulations at a time. All builds went under `build/a15/`, `build/matrix/` and `build/a15-pytest/`.
`git status` is clean apart from the two deliverables.

**Normalisation.** This container is faster than AUDIT-10's, which shared its machine with another audit. To compare
with SG17, I ran AUDIT-10's commit `1313fb6` as a control: `git archive 1313fb6` into `build/a15/old`, measured
alongside main. Its default 8-bit `isa_diff` runs at 3114 cycles/s here, against SG17's 4280, a factor of 0.728.
Throughput figures marked "normalised" are divided by 0.728. Wall times are as measured.

Method:
- **Throughput.** I ran `make test_isa_diff` and `make test_xf_isa_diff` for each of the 13 matrix configurations,
  on main and on `1313fb6`, four at a time, with old and new builds of the same configuration side by side.
  Cycles/s is the total simulated time over the summed per-test wall time in `results.xml`. I also diffed the
  per-fixture cycles.
- **Matrix.** One run of `time make -j4 matrix`, with per-module times taken from `build/matrix/<cfg>/log.txt`. I
  timed the modules that SG24–SG26 touched on main and on `1313fb6`.
- **Bench rows.** I re-ran the phase-11 sweep through `bench/run.py`'s own `_sweep`/`_medic`, without writing
  `bench/<label>.json`, and compared all 276 rows with `bench/phase-11-all.json`. I also ran `test_xf_bench` with
  `BENCH_RESULTS=1`, with its working directory under `build/`, and compared it with the default rows of
  `bench/xf-{matmul,mlp}.json`.
- **Load probes.** A scratch cocotb module (`build/a15/probe15.py`, not committed) runs a kernel at
  `MEM_LATENCY` and checks three things: memory against `xf.isasim` replaying the RTL's issue order
  (`run_trace`), `warp_issues_total` against the model's steps, and per-core issues summing to the total.
  `run_kernel`'s `check_counters` also runs. A monitor records each controller consumer's wait from valid to
  ready. The kernels are:
  - `divbar`: a per-lane loop of `data[gid]` ∈ 1..4 trips under `SYNC`, then an SMEM neighbour exchange between two
    `BAR`s, repeated `OUTER` times;
  - `uniform`: a load, an STS and a `BAR` per trip;
  - `mixed`: odd blocks write their own word every trip, while even blocks alternate an L1-D hit with a streaming
    miss, so that under `MEDIC` their misses are priority reads competing with the writes;
  - `nobar`: `divbar` without the `BAR`s.

  `xf/tests/test_audit15.py` embeds the same probe (it writes it to `build/a15-pytest/probe_a15.py`), so every
  number below can be reproduced from the repository (see "Running the probes").
- **AUDIT-10's probes** (`test/test_audit10.py`, on main) were re-run with the same parameters.
- **Existing suites under load.** I ran `test_isa_diff` and `test_xf_isa_diff`, which include the 253/255-thread and
  STR/STS-conflict fixtures, and `test_divergence`, at the brief's configuration. I also ran `test_multiwarp` at
  `NUM_CORES=4`, `MEM_LATENCY=10`.
- **Commands actually run.**

  | Command | Result |
  |---|---|
  | `make test_isa_diff` / `test_xf_isa_diff` × 13 configs, main and `1313fb6` (`build/a15/tp_{new,old}/<cfg>`) | all pass; throughput below; every shared fixture has identical cycles on both commits |
  | `time make -j4 matrix` (once) | `matrix: OK`, **64 m 05 s** |
  | phase-11 sweep (`build/a15/p11.py`, which calls `bench/run.py`'s `_sweep`/`_medic`) | 276/276 rows identical to `bench/phase-11-all.json`, 67 s |
  | `test_xf_bench`, default build, `BENCH_RESULTS=1` (cwd `build/a15/xfb`) | 18/18 default rows identical to `bench/xf-{matmul,mlp}.json` |
  | probes `divbar`/`uniform`, C4 W4, policies 0/1/2, `SIMT COALESCE DCACHE MEDIC`, L10, 253–255 threads, ± `ICACHE_LINES=16`, TPB 4 and 8 | 18 runs, all pass; no hang; counters exact |
  | probe `mixed`, C4 W4 GTO, TPB 4 and 16, `+ICACHE`, L10, on main and on `1313fb6`'s RTL | identical cycles (97 702; 13 366) |
  | `MEM_LATENCY=10 make test_isa_diff`, brief config, policies 0/1/2 (`build/a15/s/st_p*`) | 37/37 each |
  | `MEM_LATENCY=10 make test_xf_isa_diff`, brief config, GTO | 54/54 |
  | `MEM_LATENCY=10 make compile divergence_run`, brief config + `THREADS_PER_BLOCK=8`, policies 0/1/2 | 7/7 each |
  | `MEM_LATENCY=10 make test_xf_isa_diff PARAMS="-Pgpu.NUM_CORES=1 -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1"` | 54/54; the largest fixture needs 163 549 of the 200 000-cycle timeout (L2) |
  | same, `MEM_LATENCY=16`, `COCOTB_TEST_FILTER=xf_mlp_block_matmul_naive` | **TimeoutError** (L2) |
  | `MEM_LATENCY=10 make multiwarp_w4_gto BUILD=build/a15/mw PARAMS=-Pgpu.NUM_CORES=4` | 5/6: `test_age_saturation` times out (AUDIT-10 L2, unchanged) |
  | `NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests -q` | 825 passed, 29 xfailed (2 of them new); the AUDIT-10 strict xfails still xfail |
  | `python3 bench/run.py --check-citations` | `citations: OK` |

"The brief config" is `-Pgpu.NUM_CORES=4 -Pgpu.WARPS_PER_CORE=4 -Pgpu.SIMT_STACK=1 -Pgpu.DCACHE_LINES=16
-Pgpu.MEDIC=1 -Pgpu.COALESCE=1 -Pgpu.WARP_POLICY=<p>`.

There are **no critical or high findings**, so this audit adds no SG rows. Nothing hung or deadlocked in any
configuration I ran. Every probe and suite passed `check_counters`, and `warp_issues_total` equalled the model's
steps everywhere (contract 12). No cycle count regressed against a documented bench row. SG24–SG27 cost the
matrix about 1.4 CPU-minutes.

| ID | Severity | Finding | Repro |
|----|----------|---------|-------|
| L1 | Low | On kernels with `BAR`, GTO is much slower than LRR, and the final report's policy comparison does not cover it. The report says "with fetch latency in play LRR beats GTO by 3.5–6.5% at L10" and "GTLRR equals GTO in every row", but none of its rows has a `BAR`. In the same setup (both memories at L10, `ICACHE_LINES=16`), a divergent loop with two `BAR`s per trip runs: on one block, GTO **+12.2%** and GTLRR +6.6% over LRR; at 4 cores × 4 warps with 253–255 threads, GTO **+19.5% to +31.6%** and GTLRR +7.5% to +12.6%. Without the `BAR`s, GTO is +1.1%. GTO also multiplies `bar_stall_cycles_total` by 3–4 | `xf/tests/test_audit15.py::test_gto_cost_on_a_bar_kernel_is_within_the_reported_range`; control `::test_gto_cost_without_bar_is_within_the_reported_range` |
| L2 | Low | AUDIT-10 L2's fixed-timeout problem also affects the differential gates. `test_isa_diff`, `test_xf_isa_diff` and `test_divergence` give every kernel `timeout_cycles=200_000`. At `NUM_CORES=1 WARPS_PER_CORE=4` GTO, `xf_mlp_block_matmul_naive` takes 163 549 cycles at L10 (82% of the limit) and 186 204 at L13. At L16 it fails with `TimeoutError: Kernel did not complete in 200000 cycles`. It is not a hang: with a 3M limit it completes in 208 858 cycles, bit-exact with the model | `MEM_LATENCY=16 COCOTB_TEST_FILTER=xf_mlp_block_matmul_naive make test_xf_isa_diff BUILD=build/a15/l16 PARAMS="-Pgpu.NUM_CORES=1 -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1"` (about 50 s); structural: `test_audit15.py::test_differential_timeouts_scale_with_the_run` |

## Re-measured: AUDIT-10's findings on `290f041`

None was fixed: `controller.sv`'s claim order, `l1d_cache.sv`, the `MATRIX` order and `test_multiwarp`'s timeout are
as AUDIT-10 found them. The `test_audit10.py` xfails are still strict and still fail. Severities are unchanged.

| AUDIT-10 | Then | Now (same probe, same parameters) | Status |
|---|---|---|---|
| M1 fixed claim order, 4 cores | 16 blocks `[5,5,5,1]` L0, `[7,7,1,1]` L3/L10; 2130 / 3303 / 6074 cycles | identical: `[5,5,5,1]` / `[7,7,1,1]` / `[7,7,1,1]`, 2130 / 3303 / 6074 cycles; `scaling` 2130/4733 L0, 6074/14378 L10, program channel 39% / 78% busy | open, Medium. At the brief's configuration it is more extreme (below) |
| M2 L1-D fill racing a write-through | stale value 0, memory 7 | `l1d_cache.sv` unchanged; `test_audit10.py::test_l1d_fill_racing_a_write_is_not_stale` still xfails | open, Medium |
| M3 matrix over 60 min | about 84 min estimated (56 normalised); `warps2` about 44 min, started 11th | **64 m 05 s measured**; `warps2` 32.3 min, started 11th at 31.5 min; 216 CPU-min in all | open, Medium (below) |
| L1 GTO past age saturation | 80 trips: 18603, 18627, 22899, 22923; 200 trips: 35404 … 56524 | 80 trips: 18603, 18627, 22899, 22923; 200 trips (1 core): 35403, 35427, 56499, 56523; LRR 22887 … 22923; 40 trips 11651 … 11723 | open, Low; on these kernels the makespan is unchanged, but see L1 above for `BAR` kernels |
| L2 `test_age_saturation` 50 000-cycle timeout | fails at C4 L10 | still fails (`TimeoutError: Kernel did not complete in 50000 cycles`; 5/6) | open, Low; L2 above is a second instance |
| L3 throughput against LSU count | 484 cycles/s at 64 LSUs with `DCACHE`+`MEDIC` | not re-swept; `mixed` at TPB 16 × 4 cores (64 LSUs, `+MEDIC+DCACHE+ICACHE`, with my monitor attached) runs 792 cycles/s on main and 782 on `1313fb6` | open, Low; no regression |

**M1 at the brief's configuration.** 255 threads (64 blocks) of `divbar` (12 outer trips) at C4 W4, `SIMT`,
`COALESCE`, `DCACHE`, `MEDIC`, L10, no I-cache. Per-core warp issues were `[40424, 40008, 1304, 1304]`: cores 2 and 3
each ran **one block out of 64**. The makespan was 1 298 365 / 1 298 587 / 1 298 363 cycles under LRR / GTO /
GTLRR, so the program channel decides it and the policy does not matter. With `ICACHE_LINES=16` the same launch takes
208 401 / 253 181 / 223 935 cycles (6.2× faster) and per-core issues even out to `[20864, 20864, 20864, 20448]`
under LRR. The finding and its fix options are AUDIT-10's. The new information is the ratio: at W=4 and L10 the
fixed order hands 97% of a long launch to two cores.

**M3: what SG24–SG26 added.** They added fixtures (`launch{8,32}_{253,255}`, `smem_wrap`, `str_conflict`,
`sts_conflict`) and tests (`test_controller` 3 → 5, `test_medic` 1 → 2, `test_divergence` 6 → 7):
- `test_isa_diff` + `test_xf_isa_diff`: +3.9 s per configuration at default, +8 s at `warps2`, +19.4 s at
  `warps4gto`; 71 s over all 13 configurations;
- `test_controller` +0.07 s and `test_medic` +0.15 s per build; `test_divergence` +2 s at `simt`;
- the runner's per-cycle trace at W > 1 (SG25) costs no measurable throughput (`warps2` −0% / +0%, `warps4gto`
  −7% / −2%, inside the noise of the other configurations).

That is about 1.4 of 216 CPU-minutes, and no configuration slowed by more than the PLAN's 20%. The overrun is still
AUDIT-10's: `warps2` (32.3 min, 2.27× default) is the only multi-warp configuration that runs the attention and MLP
chains, and it starts 11th. Per-configuration CPU time, in minutes:

| default | icache | pipelined | coalesce | dcache | medic | all | cores1 | cores4 | latency | warps2 | warps4gto | simt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14.2 | 12.7 | 13.2 | 14.3 | 15.0 | 17.5 | 14.5 | 23.0 | 17.0 | 19.6 | **32.3** | 18.3 | 4.8 |

Finish times after the 09:36:28 start: icache +13.1, pipelined +13.6, default +14.7, coalesce +14.8, dcache +28.5,
all +29.5, medic +31.5 (then `warps2` starts), cores1 +38.1, cores4 +45.8, latency +49.4, simt +50.9, warps4gto
+56.7, warps2 +64.1 min. With `warps2` first the makespan would be about max(32, 216/4) ≈ 55 min; with AUDIT-10's
`XF_warps2` filter (dropping 23 CPU-min of attention, MLP and bench), about 50 min. During the last 12 minutes, while only `warps2` and `warps4gto` were left, one
probe and one `multiwarp` run of mine shared the machine (never more than four processes).

## L1 — GTO and GTLRR on kernels with `BAR`

`docs/final-report.md` §Multi-warp latency hiding compares the policies on `stream` and `poly`, which have no
`BAR` ("No row has barrier stalls"). From those rows it concludes that "with fetch latency in play LRR beats GTO by
3.5–6.5% at L10 at the same miss counts; with data latency alone GTO is the better policy. GTLRR equals GTO in every
row". The probe kernel `divbar` adds two `BAR`s per outer trip. The warps reach them at different times, because
`data[gid]` sets each lane's inner trip count (1–4). Both memories are at L10 with `ICACHE_LINES=16`, the report's
second table setup:

| Run | LRR | GTO | GTLRR | `bar_stall_cycles_total` (LRR / GTO / GTLRR) |
|---|---:|---:|---:|---|
| 1 core, 1 block (4 threads), W4 TPB4, 6 trips | 6411 | 7195 (**+12.2%**) | 6831 (+6.6%) | 3048 / 10032 / 7152 |
| same, `BAR`s removed (`nobar`) | 6349 | 6419 (+1.1%) | 6307 (−0.7%) | 0 |
| C4 W4 TPB4, 255 threads, 6 trips, `SIMT COALESCE DCACHE MEDIC` | 109 923 | 131 353 (+19.5%) | 119 296 (+8.5%) | 193 308 / 687 158 / 453 672 |
| same, 12 trips (326 issues/warp, past saturation) | 208 401 | 253 181 (+21.5%) | 223 935 (+7.5%) | 386 616 / 1 374 074 / 907 344 |
| C4 W4 TPB8 (2 lanes/warp, divergent), 253 threads, 6 trips | 56 629 | 73 479 (**+29.8%**) | 62 240 (+9.9%) | 79 124 / 326 168 / 254 808 |
| same, 12 trips | 107 685 | 141 670 (**+31.6%**) | 121 219 (+12.6%) | 158 264 / 691 468 / 509 616 |
| C4 W4 TPB4, `uniform` (one `BAR` per trip, no divergence), 254 threads, 40 trips | 168 756 | 168 993 (+0.1%) | 168 929 (+0.1%) | 121 280 / 535 096 / 515 440 |

The cost is the same below and past age saturation (+19.5% at 6 trips, +21.5% at 12), so it is not AUDIT-10 L1.
It comes from greed at a barrier. GTO keeps issuing one warp until that warp stalls, so the warps arrive at a `BAR`
one after another instead of overlapping their loads. The block then waits for the slowest, and GTLRR's round-robin
fallback recovers about half of the loss. Under GTO the per-core issue counts at 4 cores also spread
(`[22168, 21752, 19560, 19560]` against LRR's `[20864, 20864, 20864, 20448]`): blocks take uneven times, and the
fixed claim order (M1) favours the low cores.

This is a performance characteristic, not a defect, and the report states its kernels have no `BAR`. Its policy
guidance, though, reads as general, and `BAR` kernels are what the project's tiled kernels are (`matmul_tiled`, used by the tiled xf chains, and
`tiled_matmul`). Fix: add a `BAR` kernel (for
example `divbar`) to the phase-11 `warps`/`warps_icache` sweeps, and qualify the report's sentence ("on kernels
without `BAR`").

## L2 — Fixed 200 000-cycle timeouts in the differential tests

`test_isa_diff.diff_test` and `test_divergence.check` call `run_kernel(..., timeout_cycles=200_000)` for every
kernel, whatever the fixture, `NUM_CORES`, `WARPS_PER_CORE` or `MEM_LATENCY`. The largest 32-bit fixtures, at
`NUM_CORES=1 WARPS_PER_CORE=4` GTO, need:

| `xf_mlp_block_matmul_naive` | matrix `warps4gto` (2 cores, L0) | 1 core, L10 | L13 | L16 |
|---|---:|---:|---:|---:|
| cycles | 44 022 | 163 549 | 186 204 | 208 858 → `TimeoutError` at 200 000 |

`xf_mlp_block_rmsnorm` and `xf_attention_rmsnorm` take 134 493 cycles at L10; at the 7.5k cycles per latency step measured above they would cross the limit near L21 (extrapolated). No
matrix configuration combines one core, several warps and latency (AUDIT-10 L2), so the gate passes today. Anyone
sweeping `MEM_LATENCY` on the differential harness (as the phase-11 bench does for its own kernels) gets a failure
that reads like a hang. The 253/255-thread launches added by SG26 are far inside the limit (44 057 cycles at
C1 W4 GTO L10). Fix, as for AUDIT-10 L2: derive the timeout from the model's steps, for example
`steps × (12 + 2·latency) × NUM_CORES` with a floor.

## Checked and consistent

- **No hangs, deadlocks or starvation to completion.** None of the following hung:
  - every probe at C4 W4 under LRR, GTO and GTLRR with `SIMT_STACK`, `COALESCE`, `DCACHE`, `MEDIC` and L10, with and
    without the I-cache;
  - TPB 4 (1 lane per warp) and TPB 8 (2 lanes, divergent within the warp);
  - 253, 254 and 255 threads; the last block is partial, so warps 3 (TPB 4, 255 threads), 2–3 (TPB 8, 253) or 3
    (TPB 8, 255) are partly or wholly empty while the live warps meet at `BAR`;
  - loops to 326–356 issues per warp, past the 254 saturation;
  - the existing differential suites at the brief's configuration and L10.

  The longest single run was 1.3 M cycles (`divbar`, no I-cache). Cores 2 and 3 waited most of it for the program
  channel, but they completed (M1).
- **Contract 12 under load.**
  - `warp_issues_total` equalled the model's steps in every run (83 040, 45 092, 73 152, 38 784 and 3 616 issues on
    the large probes), and per-core issues summed to it.
  - `check_counters` passed in every run: requests equal the serviced reads plus writes, coalesced equals the
    handshakes minus the read transactions, and the I-cache totals equal the per-core sums.
  - The L1-D counters add up as well: hits + misses equal the LDRs issued (3060 = 255 × 12 in `divbar`; 1280 in
    `mixed`); `data_mem_requests` = misses − coalesced + writes (510 = 255 + 255; 1423 = 1019 − 300 + 704).
  - `bar_stall_cycles_total` is 0 at W = 1 and grows with policy greed as expected (table in L1).
  - `counters_under_load` (AUDIT-10's probe) at the brief's configuration with the I-cache, L10: 1856 issues
    (model 1856) under all three policies.
- **SG25's controller change (writes leave the MeDiC priority pass).** I compared the `mixed` workload with the
  current RTL and with `1313fb6`'s RTL, under the current harness:
  - cycles are identical: 97 702 at TPB 4, and 13 366 at TPB 16, where the channels are contended;
  - at TPB 16 the longest write wait went from 43 to 58 cycles and the longest read wait from 60 to 49; the mean
    waits are 15.9 → 16.7 cycles (write) and 16.4 → 15.4 (read).

  Writes lose some priority to MeDiC's high-utility reads, as ADR-016 intends. No write starved: every run
  finished, and pass 1 still scans writes in consumer order.
- **Cycles against documented rows.**
  - Every fixture common to `290f041` and `1313fb6` has identical cycles in all 13 configurations at 8 and 32 bits.
  - The phase-11 sweep reproduces all 276 rows of `bench/phase-11-all.json` exactly (the final report says
    "about 100 s on 4 cores"; 67 s here).
  - `test_xf_bench` reproduces the 18 default rows of `bench/xf-{matmul,mlp}.json`.
  - `test_medic` passes on main (2/2 at `MEDIC=0` and `MEDIC=1`).
- **Throughput against the SG17 baseline.** No configuration is below its normalised baseline by more than noise.
  Main is within ±8% of `1313fb6` in every configuration, so SG24–SG27 cost no throughput. The differential harness
  runs at 0.22–0.39 s per 1k cycles, far inside the 2 s budget:

  | Config | main 8 / 32-bit | `1313fb6` 8 / 32 | change | AUDIT-10 measured | main normalised | SG17 baseline | vs SG17 | s per 1k cycles |
  |---|---|---|---|---|---|---|---|---|
  | default | 3050 / 3238 | 3114 / 3252 | −2% / −0% | 2698 / 2506 | 4192 / 4451 | 4280 / 4060 | −2% / +10% | 0.33 / 0.31 |
  | icache | 3106 / 3323 | 3363 / 3203 | −8% / +4% | 2986 / 2580 | 4269 / 4567 | 4290 / 4110 | −0% / +11% | 0.32 / 0.30 |
  | pipelined | 3290 / 3161 | 3226 / 3216 | +2% / −2% | 2853 / 2445 | 4521 / 4344 | 4330 / 3790 | +4% / +15% | 0.30 / 0.32 |
  | coalesce | 3071 / 3262 | 2918 / 3441 | +5% / −5% | 2905 / 2608 | 4221 / 4484 | 4170 / 3930 | +1% / +14% | 0.33 / 0.31 |
  | dcache | 3361 / 2996 | 3326 / 2930 | +1% / +2% | 2603 / 2484 | 4619 / 4118 | 3340 / 3920 | +38% / +5% | 0.30 / 0.33 |
  | medic | 2904 / 2558 | 2693 / 2724 | +8% / −6% | 1968 / 2300 | 3991 / 3516 | 3220 / 3350 | +24% / +5% | 0.34 / 0.39 |
  | all | 2861 / 2748 | 3056 / 2779 | −6% / −1% | 2256 / 2245 | 3933 / 3777 | 3570 / 3060 | +10% / +23% | 0.35 / 0.36 |
  | cores1 | 4510 / 4157 | 4526 / 4177 | −0% / −0% | 3310 / 3235 | 6199 / 5713 | 4700 / 4370 | +32% / +31% | 0.22 / 0.24 |
  | cores4 | 2833 / 2800 | 2827 / 2800 | +0% / +0% | 1866 / 1992 | 3894 / 3848 | 3340 / 3140 | +17% / +23% | 0.35 / 0.36 |
  | latency | 3852 / 3384 | 3702 / 3556 | +4% / −5% | 3040 / 2780 | 5294 / 4651 | 4090 / 4060 | +29% / +15% | 0.26 / 0.30 |
  | warps2 | 3467 / 3442 | 3482 / 3439 | −0% / +0% | 2610 / 2594 | 4766 / 4731 | — | — | 0.29 / 0.29 |
  | warps4gto | 3473 / 3436 | 3720 / 3517 | −7% / −2% | 2961 / 2764 | 4774 / 4723 | — | — | 0.29 / 0.29 |
  | simt | 3714 / 3344 | 3710 / 3240 | +0% / +3% | 2830 / 2628 | 5105 / 4596 | — | — | 0.27 / 0.30 |

  "Change" compares main with `1313fb6`, run side by side; four jobs at a time give ±8% noise. AUDIT-10's `cores4`
  −17% does not reproduce on an idle machine: +17% / +23% normalised.

  Matrix per-module times on the default build, against AUDIT-10's:
  - `test_xf_attention` 294 s (AUDIT-10: 362), `test_xf_mlp_block` 267 (353), `test_xf_matmul` 129 (173);
  - `test_xf_rmsnorm` 51 (72), `test_xf_bench` 42 (57), `test_xf_elementwise` 35 (45);
  - `test_xf_isa_diff` 23.6 (31), `test_isa_diff` 8.2.

  The ratios between modules are unchanged; the container is about 1.25× faster.
- **`xf.isasim`.** 2.5 µs per step over every fixture at W = 1 and W = 4 (78 514 steps in 0.20 s). The 253/255-thread
  launch fixtures run at 2.3–3.4 µs per step. `run_trace` adds nothing measurable. The model is never on the critical
  path.
- **SG27.** `check_params` is a Make-time filter with no effect on simulation time. Every matrix configuration
  passes it.

## Running the probes

`NANOGPT_DIR=/home/claude/nanogpt python3 -m pytest xf/tests/test_audit15.py` writes the probe module to
`build/a15-pytest/probe_a15.py` and runs L1, its control and the 255-thread counters control (about 30 s). It skips
without the toolchain. The load runs above use the same module:

```bash
B=build/a15/c4w4; mkdir -p $B
sv2v -w $B/raw.v $(ls src/*.sv | grep -v warp_scheduler_tb) && (printf '`timescale 1ns/1ns\n'; cat $B/raw.v) > $B/gpu.v
iverilog -o $B/sim.vvp -s gpu -g2012 -Pgpu.NUM_CORES=4 -Pgpu.WARPS_PER_CORE=4 -Pgpu.WARP_POLICY=1 \
  -Pgpu.SIMT_STACK=1 -Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1 -Pgpu.COALESCE=1 $B/gpu.v   # add -Pgpu.ICACHE_LINES=16
PYTHONPATH=$PWD:$PWD/build/a15-pytest PYGPI_PYTHON_BIN=$(which python3) LIBPYTHON_LOC=$(cocotb-config --libpython) \
  COCOTB_TEST_MODULES=probe_a15 MEM_LATENCY=10 A15_THREADS=255 A15_OUT=$B/out.json \
  A15_KERNEL="$(python3 -c 'from xf.tests.test_audit15 import DIVBAR; print(DIVBAR.format(outer=12))')" \
  vvp -M $(cocotb-config --lib-dir) -m libcocotbvpi_icarus $B/sim.vvp; cat $B/out.json
# NOBAR instead of DIVBAR for the no-BAR control; -Pgpu.THREADS_PER_BLOCK=8 and A15_THREADS=253 for the TPB-8 rows.
# AUDIT-10's probes: COCOTB_TEST_MODULES=test.test_audit10 COCOTB_TEST_FILTER='dispatch_spread$' AUDIT10_LATENCY=10 ...
```

The `mixed` and `uniform` kernels and the wait monitor are in `build/a15/probe15.py` (scratch, not committed). The
numbers above do not depend on them for any finding.
