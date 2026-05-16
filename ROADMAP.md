# ROADMAP

Live status tracker. Update as phases land.

| Phase | Description                                  | Status            | Report                                          |
|-------|----------------------------------------------|-------------------|-------------------------------------------------|
| 0     | Bootstrap (clone baseline, verify toolchain) | DONE              | [phase-0](docs/phase-reports/phase-0-report.md) |
| 1     | Clean reimplementation in `src/`             | DONE              | [phase-1](docs/phase-reports/phase-1-report.md) |
| 2     | Bench harness + extra kernels                | DONE              | [phase-2](docs/phase-reports/phase-2-report.md) |
| 3     | Instruction cache (parameterised)            | DONE              | [phase-3](docs/phase-reports/phase-3-report.md) |
| 4     | Pipelining                                   | DEFERRED (note 1) | [phase-4](docs/phase-reports/phase-4-report.md) |
| 5     | Memory coalescing                            | DONE              | [phase-5](docs/phase-reports/phase-5-report.md) |
| 6     | Branch divergence — kernel level             | PARTIAL (note 2)  | [phase-6](docs/phase-reports/phase-6-report.md) |
| 7     | Shared memory + barriers                     | DESIGNED (note 3) | [phase-7](docs/phase-reports/phase-7-report.md) |
| 8     | Pluggable warp scheduler                     | DESIGNED (note 4) | [phase-8](docs/phase-reports/phase-8-report.md) |
| 9     | MeDiC memory divergence correction           | DESIGNED (note 5) | [phase-9](docs/phase-reports/phase-9-report.md) |
| 10    | Final report                                 | DONE              | [final](docs/final-report.md)                   |

## Notes on partial/deferred phases

1. **Phase 4 — Pipelining.** Honest descope. The FETCH→DECODE→REQUEST→WAIT
   →EXECUTE→UPDATE 6-stage *sequential* FSM in `scheduler.sv` is shared across
   all threads in a block. Converting it to a classic 5-stage *overlapping*
   pipeline with hazard detection touches every module: the fetcher would
   need to stay producing while the executor consumes, the register file
   needs forwarding paths, the LSU needs a per-stage queue, and the
   scheduler needs PC-redirect on branch. We surface this rather than
   silently descope. See `docs/decisions/ADR-005-phase-4-deferred.md`.

2. **Phase 6 — Branch divergence.** We added a `divergent_relu` kernel
   exercising *data divergence* under the existing single-PC SIMT (threads
   produce different outputs via mask-and-blend). True SIMT-stack
   reconvergence requires per-thread PC and an active mask in the
   scheduler. See `docs/decisions/ADR-006-divergence.md` for the design.

3. **Phase 7 — Shared memory + barriers.** Architecture designed in
   `docs/decisions/ADR-007-shared-memory.md` with new opcodes `LDS`,
   `STS`, `BAR` documented in `docs/isa.md`. RTL stub in
   `src/shared_memory.sv` (not wired into the core in this revision —
   wiring touches the entire datapath; we document the design and a
   tile-blocked kernel that would use it).

4. **Phase 8 — Pluggable warp scheduler.** Designed in
   `docs/decisions/ADR-008-warp-scheduler.md`. The current architecture
   runs one warp per core (THREADS_PER_BLOCK threads in lockstep), so
   warp scheduling is degenerate. Demonstrating LRR vs GTO vs GTLRR
   requires multi-warp-per-core, a deeper change.

5. **Phase 9 — MeDiC.** Designed in
   `docs/decisions/ADR-009-medic.md`. Requires the multi-warp scheduler
   from Phase 8 plus a real L1-D before the three MeDiC components
   become meaningful.

The phases marked DESIGNED include enough detail (ADRs, RTL skeletons,
ISA updates, test plans) that future iterations can implement them
without re-doing the design work.
