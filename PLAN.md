# PLAN — Transformer inference on tiny-gpu-extended

Run an int8-quantized transformer MLP block (and, as a stretch, one attention
head) exported from nanoGPT Extended on this simulator, verified bit-exact
against an integer golden model and within quantization tolerance against
PyTorch.

Every thread reads this file first and updates its own row (status, PR,
open issues) when it finishes. Interface changes after SG1 merges stop the
thread and go back to the user.

## Status

| ID  | Sub-goal                               | Depends on      | Owner (thread) | Status  | PR |
|-----|----------------------------------------|-----------------|----------------|---------|----|
| SG0 | Harness hygiene                        | —               | Harness hygiene (SG0) | DONE | [#2](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/2) |
| SG1 | Shared interfaces                      | SG0             | Shared interfaces (SG1) | DONE | [#3](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/3) |
| SG2 | Wide datapath + signed ISA ops (RTL)   | SG1             | —              | TODO | — |
| SG3 | Quantization + export (M1)             | SG1             | —              | TODO | — |
| SG4 | Int8 matmul kernels, naive + tiled (M2)| SG2             | —              | TODO | — |
| SG5 | Elementwise + RMSNorm kernels (M3)     | SG2             | —              | TODO | — |
| SG6 | MLP block end-to-end (M4)              | SG3, SG4, SG5   | —              | TODO | — |
| SG7 | Benchmarks (M5)                        | SG4, SG6        | —              | TODO | — |
| SG8 | Stretch: single attention head (M6)    | SG6             | —              | TODO | — |
| AUD | Audit rounds (AUDIT-<n>.md)            | SG7 (+SG8)      | —              | TODO | — |

SG1 open issue: `softmax_lut` rows sum to `one` within T/2 LSB (2 at T=4); SG8's "1 LSB" criterion needs revisiting.

Concurrency: at most two build threads at once. SG0 → SG1 → SG2 run
alone. SG3 may start in parallel with SG2. SG4 ∥ SG5 after SG2. SG6, SG7,
SG8 run in order.

Approved by the user on 2026-10-13 with the defaults below; build proceeds
without further approval gates.

## Dispatch protocol

- One thread per sub-goal, branched from the default branch
  (`claude/complete-markdown-tasks-IG7yf`), one PR per sub-goal.
- A thread starts by reading this file, sets its row to `IN PROGRESS`
  with its thread title, and delegates bounded, test-driven tasks to at
  most one or two subagents at a time.
- A thread merges its own PR once its Verify command passes from a clean
  checkout and the full existing suite still passes; it then sets its row
  to `DONE` with the PR link and any open issues, in that same PR.
- When a row turns `DONE`, the coordinator starts every sub-goal whose
  dependencies are all `DONE`, keeping at most two build threads running.
- Blocked threads set their row to `BLOCKED: <reason>` and report in their
  own thread.

## Environment setup (every thread)

```bash
apt-get install -y iverilog                  # 12.0
pip install cocotb==2.0.1 numpy pytest       # 2.1.x breaks the Makefile
curl -sSL -o /tmp/sv2v.zip https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip
python3 -c "import zipfile;zipfile.ZipFile('/tmp/sv2v.zip').extractall('/tmp/sv2v')"
install /tmp/sv2v/*/sv2v /usr/local/bin/
git checkout bench/results                   # tests overwrite these until SG0 lands
```

SG3 additionally needs `pip install torch` (CPU) and a clone of nanoGPT
Extended at `03dbab8` in `$NANOGPT_DIR`.

## Facts the plan is built on

Checked on the default branch @ `d37c7d5` (this repo) and nanoGPT Extended @ `03dbab8`.

- Datapath is 8-bit: registers, ALU, data words, data addresses
  (`src/registers.sv`, `src/alu.sv`, `src/lsu.sv`). Data memory is 256
  words; `DATA_MEM_ADDR_BITS` exists on `gpu` but `lsu` addresses are
  hard-wired `[7:0]`.
- Arithmetic is unsigned. `CMP` sets N = "differs", P is never set
  (ADR-001), so there is no signed compare; `DIV` is unsigned; no shifts.
- Encoding is `op|rd|rs|rt`, 4 bits each. Free opcodes: `1101`, `1110`.
- SMEM: 64 entries per core; LDS/STS/BAR are wired (ADR-007).
  `docs/isa.md` still lists them as "not yet wired" and
  `kernels/tiled_matmul.asm` is a stub.
- No assembler: test programs are hand-encoded in each test file.
- One int8 matmul with K=16 needs ≥20-bit accumulation; SwiGLU weights for
  d=8, hidden=16 are 384 bytes. Neither fits the 8-bit/256-word machine.
- nanoGPT `SwiGLUMLP` rounds hidden to a multiple of 256; `RMSNorm` has a
  learned weight and eps; `Block` = `x + mlp(norm(x))` for the MLP half.
- Toolchain: iverilog 12.0, sv2v 0.0.13, cocotb **2.0.1**. cocotb 2.1.0
  aborts with "No GPI_USERS specified" and `vvp` still exits 0, so
  `make regression` reports green with zero tests run.
- Running the suite at default params overwrites 11 tracked
  `bench/results/*.json` files that were recorded under other params.

All 23 cocotb tests (9 make targets) pass at default params with cocotb 2.0.1. `make harness_selftest` checks the failure gate (SG0).

## Decisions (defaults; user may override at approval)

- **D1 Wide datapath.** Add a `WORD_BITS` parameter (default 8, so all
  existing tests are unchanged). Transformer configs run at `WORD_BITS=32`,
  `DATA_MEM_ADDR_BITS=12`. Alternative rejected: multi-precision int32 on
  8-bit registers (needs MULH/carry/segment ops, triples kernel length).
- **D2 ISA additions, only two, both needed by kernels:**
  `1101 SRA rd, rs, rt` (arithmetic shift right; requantization) and
  `1110 MAX rd, rs, rt` (signed max; ReLU and clamping, `MIN(x,c) =
  -MAX(-x,-c)`). Documented in `docs/isa.md` + ADR-010 in the same commit.
- **D3 Model.** `n_embd=8`, SwiGLU hidden 16, `T=4` tokens, `bias=False`,
  `norm_type='rmsnorm'`, seeded random init. Hidden-dim override is done in
  the exporter (replace the three `nn.Linear`s), not by changing nanoGPT.
  Stretch head: `n_head=1`, `head_dim=8`, `pos_embedding='none'`.
- **D4 Quantization.** Symmetric per-tensor int8 for weights and
  activations, int32 accumulators, fixed-point requant
  `clamp((acc*M + 2^(s-1)) >> s, -128, 127)` with exporter-checked bounds
  so `acc*M` fits in int32. SiLU and exp via 256-entry LUTs indexed by
  `x+128`; rsqrt via LUT.
- **D5 Two-level verification.** Every kernel test is bit-exact against
  the Python integer golden model. Float tolerance against PyTorch is
  asserted only at export (SG3) and end-to-end (SG6, SG8), with the bound
  measured in SG3 and frozen in the contract below.
- **D6 nanoGPT is read-only**, pinned at `03dbab8`, located via
  `NANOGPT_DIR`. cocotb tests read committed fixtures and never import torch.

## Interface contracts (owned by SG1; frozen once merged)

1. **ISA table** — `xf/isa.py`: `OPCODES` (mnemonic → opcode, operand
   format, incl. `SRA`/`MAX`), `FIELDS`, `encode(mnem, **fields)`,
   `decode(word)`. Single source for the assembler and `docs/isa.md`
   (`xf/tests/test_isa.py` checks the doc table).
2. **Assembler** — `xf/asm.py`: `assemble(text) -> list[int]`,
   `disassemble(words)`; labels for `BRnzp`, `%blockIdx/%blockDim/%threadIdx`
   aliases, `#imm` 0–255, writes to R13–R15 rejected. `kernels/*.asm` are
   the canonical kernel sources.
3. **Memory image** — `xf/image.py`: `Image(word_bits=32, addr_bits=12)`,
   `alloc(name, shape, dtype, scale=None, data=None) -> base` (bump
   allocator in call order), `write`, `read(name, words=None)` (signed),
   `to_json`/`from_json`. JSON
   `{word_bits, addr_bits, segments: {name: {base, shape, dtype, scale}}, words}`;
   dtypes `i8`, `i32`, one element per word, words stored two's complement.
   Requant constants (`M`, `s`) live in named segments, so kernels are
   shape- and scale-generic.
4. **Golden integer ops** — `xf/golden.py`, pure numpy, every intermediate
   checked to fit int32 (`OverflowError` otherwise), semantics identical to
   the kernels bit for bit:
   - `matmul_i8(a, b)` → int32 `a @ b`.
   - `requant(acc, M, s)` = `clamp((acc*M + 2^(s-1)) >> s, -128, 127)`, no
     rounding term when `s = 0`, `0 ≤ s ≤ 31`, `>>` arithmetic.
   - `relu(x)`; `residual_add(a, b, Ma, Mb, s)` = `requant(a*Ma + b*Mb, 1, s)`.
   - `swiglu_lut(gate, up, silu_lut, M, s)` = `requant(silu_lut[gate+128] * up, M, s)`,
     256 int8 entries.
   - `rmsnorm_int(x, w, rsqrt_lut, lut_shift, M, s)`: per row
     `r = rsqrt_lut[min(Σx² >> lut_shift, len-1)]`,
     `out = requant(x*w*r, M, s)`; `1/d` and eps are folded into the LUT.
   - `softmax_lut(scores, exp_lut, one, causal=True)`: `d = s - rowmax`
     over unmasked j (`j ≤ i`), `e = exp_lut[d+255]` (256 non-negative
     int32), `p = (e*one + S//2) // S`, masked `p = 0`. Rows sum to `one`
     within `T/2` LSB.
5. **Launch driver** — `test/helpers/runner.py::run_launches(dut, image,
   launches) -> (metrics, data_memory)` with `Launch(program, threads,
   name="", timeout_cycles=20000)`: one data memory loaded from the image
   and persisting across launches, GPU reset before each launch, per-launch
   metrics identical to `run_kernel`'s; rejects an image whose word/address
   width differs from the DUT bus.
6. **Build target** — `make test_xf_<name>` compiles with
   `-Pgpu.WORD_BITS=32 -Pgpu.DATA_MEM_DATA_BITS=32 -Pgpu.DATA_MEM_ADDR_BITS=12`
   (extra `PARAMS` appended) into `build/sim_xf.vvp` and runs
   `test/test_xf_<name>.py`; the default build is untouched. It elaborates
   once SG2 adds `WORD_BITS`.
7. **Fixtures** — `xf/fixture.py` `save`/`load`,
   `test/fixtures/xf/<case>.json`:
   `{case, image, inputs, golden, reference, tol}`; `inputs` are written
   into their segments before launch, `golden` (int) and `reference`
   (float) are keyed by output segment.
8. **Tolerance** — `|dequant(sim) - torch| ≤ TOL` per element, TOL set in
   SG3 from measured quantization error and recorded here.

## Sub-goals

Each sub-goal is one thread and one PR. "Verify" commands must pass from a
clean checkout with cocotb 2.0.1.

### SG0 — Harness hygiene
- Pin `cocotb==2.0.1` (README, CLAUDE.md toolchain).
- `make regression` and `make test_%` fail when cocotb reports any failure
  or runs zero tests (check `results.xml`).
- Tests stop overwriting tracked `bench/results/*.json`; only
  `bench/run.py` writes them.
- Fix the stale LDS/STS/BAR section of `docs/isa.md`.

Accept: all 24 tests pass; a deliberately failing test and a missing
module both make `make regression` exit non-zero; `git status` clean after
the suite.
Verify: `make regression && make test_edge_cases && git status --porcelain`

### SG1 — Shared interfaces
Contracts 1–7 above as code, with no kernels and no RTL.

Accept: the assembler reproduces every hand-encoded program in
`test/test_*.py` bit-identically; image round-trips; golden ops match
straightforward numpy references on random and edge inputs (±127, −128, 0,
saturation); `run_launches` runs two existing kernels back to back on one
memory and matches running them separately.
Verify: `python3 -m pytest xf/tests -q && make test_launches`

### SG2 — Wide datapath + signed ISA ops
`WORD_BITS` through registers, ALU, LSU, controller, SMEM, DCR-derived
regs and the cocotb memory model; `SRA`, `MAX` in decoder and ALU; ADR-010
and `docs/isa.md`.

Accept: all 24 tests pass at default params; the 8 kernel tests also pass
at `WORD_BITS=32`; directed tests for `SRA` (shift 0, 31, negative
operands) and `MAX` (sign boundaries) pass at both widths; old behaviour
of `CMP` unchanged.
Verify: `make regression && make regression_xf && make test_xf_isa_ext`

### SG3 — Quantization + export (M1)
`xf/export.py` builds the D3 model from nanoGPT, calibrates scales,
quantizes, computes golden int outputs and float reference outputs, writes
fixtures. Records TOL in this file.

Accept: deterministic (two runs, identical bytes); golden int pipeline
dequantized vs nanoGPT float within TOL for ≥20 seeded inputs; all
requant bounds asserted; image fits `DATA_MEM_ADDR_BITS=12`.
Verify: `NANOGPT_DIR=... python3 -m xf.export --check && python3 -m pytest xf/tests -q`

### SG4 — Int8 matmul kernels (M2)
`kernels/matmul_naive.asm` (global loads) and `kernels/matmul_tiled.asm`
(SMEM tiles, BAR), both with optional fused requant.

Accept: bit-exact vs `golden.matmul_i8` for shapes 4×8×16, 4×16×8,
1×8×8, non-tile-multiple K, all-extreme and all-zero operands, ≥10 random
seeds; passes under default, `COALESCE=1`, `ICACHE_LINES=16`,
`PIPELINED=1`, `DCACHE_LINES=16`.
Verify: `make test_xf_matmul && make test_xf_matmul PARAMS="-Pgpu.COALESCE=1 -Pgpu.PIPELINED=1"`

### SG5 — Elementwise + RMSNorm kernels (M3)
Requant/clamp, ReLU, SwiGLU (SiLU LUT ⊙ up, requant), residual add with
scale alignment, RMSNorm (sum of squares, rsqrt LUT, weight).

Accept: bit-exact vs golden on random and saturating inputs, including
−128, LUT endpoints, and zero vectors for RMSNorm.
Verify: `make test_xf_elementwise && make test_xf_rmsnorm`

### SG6 — MLP block end-to-end (M4)
Launch chain: RMSNorm → gate, up matmuls → SwiGLU → down matmul →
residual, on one image.

Accept: final int8 output bit-exact vs golden; dequantized output within
TOL of nanoGPT `x + mlp(norm(x))` for every fixture input; passes under
the SG4 param matrix.
Verify: `make test_xf_mlp_block`

### SG7 — Benchmarks (M5)
Extend `bench/run.py` with naive vs tiled × `COALESCE` × `DCACHE_LINES`
for the SG4 shapes and the SG6 block; results in `bench/xf-*.json` and a
table in `docs/xf-bench.md`.

Accept: reproducible numbers (two runs equal); every row's kernel also
passes its correctness check in the same run.
Verify: `python3 bench/run.py --xf --label xf-matmul`

### SG8 — Stretch: single attention head (M6)
QKᵀ (matmul kernel reuse), scale, causal mask, fixed-point softmax
(max-subtract, exp LUT, sum, reciprocal), ×V.

Accept: bit-exact vs `golden.softmax_lut` pipeline; within TOL of nanoGPT
single-head causal attention; rows sum to the fixed-point one within 1 LSB.
Verify: `make test_xf_attention`

## Audit

After SG7 (and again after SG8 if built), run separate audit sessions that
did not write the code, one lens each: correctness vs reference, numerical
tolerance, test coverage, ISA/kernel consistency. Each writes
`AUDIT-<n>.md`, findings ranked by severity, each with a repro or failing
test. Critical and high findings become new SG rows here and go through
the same build-and-verify loop. Rounds repeat until one full round has no
critical or high findings.

## Rules for every thread
- Subagents return only: files changed, tests added, verify command +
  result, open issues.
- Stop and report after 3 failed verifications of one task, or when a fix
  needs a contract change.
- Never break a passing test. Conventional commit prefixes. ISA changes
  update `docs/isa.md` and an ADR in the same commit.
