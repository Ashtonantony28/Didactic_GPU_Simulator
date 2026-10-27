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
| SG2 | Wide datapath + signed ISA ops (RTL)   | SG1             | SG2 wide datapath and ISA | DONE | [#4](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/4) |
| SG3 | Quantization + export (M1)             | SG1             | SG3 | DONE | [#5](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/5) |
| SG4 | Int8 matmul kernels, naive + tiled (M2)| SG2             | Int8 matmul kernels (SG4) | DONE | [#6](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/6) |
| SG5 | Elementwise + RMSNorm kernels (M3)     | SG2             | SG5 elementwise and RMSNorm | DONE | [#7](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/7) |
| SG6 | MLP block end-to-end (M4)              | SG3, SG4, SG5   | MLP block end-to-end (SG6) | DONE | [#8](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/8) |
| SG7 | Benchmarks (M5)                        | SG4, SG6        | SG7 benchmarks | DONE | [#9](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/9) |
| SG8 | Stretch: single attention head (M6)    | SG6             | Attention head (SG8) | DONE | [#10](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/10) |
| SG9 | AUDIT-1 H1: scale-invariant RMSNorm    | SG3, SG5, SG6   | SG9 RMSNorm precision fix | DONE | [#18](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/18) |
| SG10 | AUDIT-2 H1: calibration headroom     | SG3, SG8        | SG10 calibration headroom | DONE | [#14](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/14) |
| SG11 | AUDIT-3 H1: L1-D cache tests         | SG12            | SG11 cache tests | DONE | [#16](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/16) |
| SG12 | AUDIT-3 H2: config matrix + full-suite targets | SG0  | SG12 config matrix | DONE | [#17](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/17) |
| SG13 | AUDIT-5 H1: ln_1 in front of attention | SG8, SG9 | SG13 attention input norm | DONE | [#20](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/20) |
| SG14 | AUDIT-6 H1: attention scores precision | SG13 | SG14 attention TOL bound | DONE | [#25](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/25) |
| SG15 | AUDIT-7 H1: torch-gated checks in the merge gate | SG12 | SG15: make the merge gate run the real tolerance checks | DONE | [#24](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/24) |
| SG16 | AUDIT-7 H2: per-segment gain and RMS checks | SG13, SG14, SG15 | SG16 segment checks | DONE | [#33](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/33) |
| SG17 | Round 3 shared interfaces: ISA model, trace fixtures, multi-warp and SYNC contracts | SG15 | SG17 shared interfaces | DONE | [#27](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/27) |
| SG18 | Controller: parallel channels, exact telemetry | SG17 | SG18 controller fix | DONE | [#28](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/28) |
| SG19 | MeDiC that can install and hit | SG18 | SG19 MeDiC fix | DONE | [#30](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/30) |
| SG20 | Multi-warp core, warp scheduler wired, real BAR | SG17, SG18 | SG20 multi-warp core | DONE | [#31](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/31) |
| SG21 | SIMT-stack control divergence | SG20 | SG21 SIMT stack | DONE | [#32](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/32) |
| SG22 | Kernel and binder robustness (AUDIT-4, AUDIT-8) | SG17 | SG22 kernel and binder robustness | DONE | [#29](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/29) |
| SG23 | Benchmarks under latency and multi-warp; docs refresh | SG19, SG20, SG21 | SG23 benchmarks and docs | DONE | [#34](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/34) |
| SG24 | AUDIT-12 H1–H3: gate coalescing, I-cache counts and STS lane order | SG23 | SG24 coverage gaps | DONE | [#39](https://github.com/Ashtonantony28/Didactic_GPU_Simulator/pull/39) |
| SG25 | AUDIT-14 H1: global STR conflicts agree in model and RTL | SG24 | SG25 store conflicts | TODO | |
| SG26 | AUDIT-13 H1–H2: gate SMEM words ≥ 64 and 253–255-thread launches | SG24 | SG26 SMEM and launch limits | TODO | |
| AUD | Audit rounds (AUDIT-<n>.md)            | SG7 (+SG8)      | —              | TODO | — |

SG1 open issue (resolved in SG8): `softmax_lut` rows sum to `one` within T/2 LSB for an arbitrary LUT, and 2 at T=4 is reachable (four equal exps, `one = 126`). The contract stays as is; SG8 keeps the 1 LSB criterion as a property of the exported table, proved by enumerating every reachable row and asserted on every simulated row.

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
  checkout and the full existing suite (`NANOGPT_DIR=... make test_all` and
  `make -j4 matrix`) still passes; it then sets its row
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

Every thread also needs `pip install torch` (from PyPI; download.pytorch.org
is blocked by the egress proxy) and a clone of nanoGPT Extended at `03dbab8`
in `$NANOGPT_DIR`: `make test_all` fails without them (SG15). Only `xf/export.py`
imports torch.

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
   - `rmsnorm_int(x, w, rsqrt_lut, M, s)` (SG9): per row `ss = Σx²`, `k` the
     least `k ≥ 0` with `ss >> 2k < len`, `r = rsqrt_lut[ss >> 2k]`,
     `out = requant(x*w*r, M, s + k)`; the LUT holds rsqrt of its exact index
     with `1/d` and eps folded in (`golden.rsqrt_index`).
   - `softmax_lut(scores, exp_lut, one, causal=True)`: `d = s - rowmax`
     over unmasked j (`j ≤ i`), `e = exp_lut[d+255]` (256 non-negative
     int32), `p = (e*one + S//2) // S`, masked `p = 0`. Rows sum to `one`
     within `T/2` LSB.
5. **Launch driver** — `test/helpers/runner.py::run_launches(dut, image,
   launches) -> (metrics, data_memory)` with `Launch(program, threads,
   name="", timeout_cycles=20000, max_block=None)`: one data memory loaded from the image
   and persisting across launches, GPU reset before each launch, per-launch
   metrics identical to `run_kernel`'s; rejects an image whose word/address
   width differs from the DUT bus. `max_block` defaults to the program's `max_block`
   (`xf.kernels.Program`; `matmul_tiled` sets 16), and every launch is checked against the DUT's
   `THREADS_PER_BLOCK` with `xf.kernels.check_block` before any runs (SG22).
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
   Every non-input activation scale is max-abs times `HEADROOM` = 1.5
   (SG10), so no activation saturates on the 200k held-out inputs
   `rng(12345)` (peak 93.4 LSB MLP, 97.3 LSB attention); `export --check`
   measures them. Errors are input-quantization-free (the float reference
   runs on `dequant(x_q)`).
   MLP block: **TOL = 2 output LSB** (`2 * scale(out)` = 0.107867, the
   fixture `tol`). Measured max error (SG9) 1.060 LSB over 200 seeded inputs
   and 1.418 LSB over the 200k held-out inputs, both asserted by
   `export --check`; 1.20, 1.20, 1.22, 1.14 LSB on
   `rng(7).standard_normal((2000, 4, 8))` scaled by 1.0, 0.5, 0.3, 0.1.
   Attention head (`attn(ln_1(x))`, SG13): **TOL = 6 output LSB**
   (`6 * scale(y)` = 0.083531; SG14, was 3). Set from an adversarial search, not a Gaussian
   sample: hill-climbing unsaturated int8 inputs reaches 5.35 LSB (`test_audit6.py`; search-backed,
   not a proof), and the Gaussian tail alone exceeds 3 (3.18 LSB in 1M, 3.23 in 200k at seed 2026). No single
   rounding dominates (`h` ~1.1, `v`/`o` 0.6–0.85, `y` 0.5, `p` 0.3–0.4 LSB on the worst inputs;
   unrounded `q`/`k` give the same search maximum). Measured max error 2.400 LSB over 2000 seeded
   inputs and 2.976 LSB over the 200k held-out inputs (peak activation
   86.1 LSB), and 3.337 LSB with channel 5 ×4, asserted by `export --check`; 2.09–2.66 LSB on
   `rng(7).standard_normal((2000, 4, 8))` scaled by 0.05–10, asserted by
   `test_audit5.py`; up to 3.46 LSB with any one channel ×4, and 3.23/3.08/3.02 on fresh
   Gaussian/Laplace/Student-t inputs, asserted by `test_audit6.py`.
   Its scales are calibrated on 2^16 inputs: with the MLP's 4096, `k` and
   `v` saturate on held-out inputs and the worst error reaches 16.6 LSB.
   Segment bounds (SG16, AUDIT-7 H2): on the 200k held-out inputs every golden segment of both
   blocks keeps its least-squares gain `Σ(dequant·ref) / Σ(ref²)` within **±0.0025** of the value
   measured at SG16 and its RMS error within **1.1×** of it (`export.SEGMENTS`, `GAIN_TOL`, `RMS_TOL`),
   asserted by `export --check` and `test_audit7.py`. Reference: nanoGPT's float activation (`kT` is `k`
   transposed, `p` the softmax, `scores` compared on causal entries only). Gain / RMS LSB at SG16:
   MLP `h` 1.0021/0.298, `gate` 1.0024/0.409, `up` 1.0018/0.423, `act` 1.0045/0.369,
   `down` 1.0037/0.753, `out` 1.0001/0.337; attention `h` 0.9997/0.290, `q` 1.0009/0.482,
   `k` = `kT` 0.9998/0.454, `v` 0.9998/0.454, `scores` 0.9998/0.542, `p` 1.0000/0.318,
   `o` 0.9964/0.443, `y` 0.9958/0.501.

`xf/export.py` (SG3) layout of `test/fixtures/xf/mlp_block_<0-3>.json`:
segments in allocation order `x, norm_w, rsqrt_lut[1024],
norm_M, norm_s, h, w_gate, gate_M, gate_s, gate, w_up, up_M, up_s, up,
silu_lut, act_M, act_s, act, w_down, down_M, down_s, down, res_Ma, res_Mb,
res_s, out` (2005 words). Weights are `[in, out]` (`x @ W`); `[1]` i32
segments hold requant constants; `norm_M` is 1 with all rsqrt/eps/1/d/scale
factors folded into `rsqrt_lut`, whose entry 0 (only met by `x = 0`) is 0. `golden` holds `h, gate, up, act, down,
out`; `reference` holds `out`. `export.mlp_block_int(image, x)` is the
golden chain on those segments, and every (M, s) is chosen against the
worst case over all int8 inputs (`export.check_bounds`).

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

Kernel calling convention (SG4, for SG5+): a kernel reads its operands from
an i32 args segment. Its words 0 and 4 are `CONST` immediates of the ARGS
prologue (base = hi*256 + lo), which `xf/kernels.py::program(name, base)`
patches; `kernels.matmul(img, kind, a, b, c, mq=0, s=0)` allocates
`<c>.args` = `[a, b, c, M, K, N, mq, s, rnd]` and returns `(program, threads)`.
`mq = 0` stores int32 accumulators. `matmul_tiled` needs blockDim 4 and
`N % 4 == 0`.

### SG5 — Elementwise + RMSNorm kernels (M3)
Requant/clamp, ReLU, SwiGLU (SiLU LUT ⊙ up, requant), residual add with
scale alignment, RMSNorm (sum of squares, rsqrt LUT, weight).

Accept: bit-exact vs golden on random and saturating inputs, including
−128, LUT endpoints, and zero vectors for RMSNorm.
Verify: `make test_xf_elementwise && make test_xf_rmsnorm`

Bindings (SG5, for SG6): `xf/elementwise.py` `requant(img, x, out, m, s)`,
`relu(img, x, out)`, `swiglu(img, gate, up, lut, out, m, s)`,
`residual_add(img, a, b, out, ma, mb, s)`,
`rmsnorm(img, x, w, lut, out, m, s)`, same convention as
`kernels.matmul`: `<out>.args` allocated, `(program, threads)` returned, one
thread per output element (≤ 255). Kernels `kernels/{requant,relu,swiglu,
residual_add,rmsnorm}.asm`; `rmsnorm` has each thread sum its own row.

### SG6 — MLP block end-to-end (M4)
Launch chain: RMSNorm → gate, up matmuls → SwiGLU → down matmul →
residual, on one image.

Accept: final int8 output bit-exact vs golden; dequantized output within
TOL of nanoGPT `x + mlp(norm(x))` for every fixture input; passes under
the SG4 param matrix.
Verify: `make test_xf_mlp_block`

Chain (SG6, for SG7/SG8): `xf/mlp.py` `launches(img, kind, acts=None)` returns
`[(out segment, (program, threads))]` for the six launches on an exported image,
`kind` picking the matmul kernel; `alloc_acts(img, prefix)` adds a fresh set of
activation segments so several inputs share one image (≤ 5 extra sets fit).

### SG7 — Benchmarks (M5)
Extend `bench/run.py` with naive vs tiled × `COALESCE` × `DCACHE_LINES`
for the SG4 shapes and the SG6 block; results in `bench/xf-*.json` and a
table in `docs/xf-bench.md`.

Accept: reproducible numbers (two runs equal); every row's kernel also
passes its correctness check in the same run.
Verify: `python3 bench/run.py --xf --label xf-matmul`

Bench (SG7): `--xf --label xf-<b>` runs `test/test_xf_bench.py::bench_<b>`
(`matmul`: SG4 shapes; `mlp`: fixture 0, all six launches) for both matmul kinds
under default, `COALESCE=1`, `DCACHE_LINES=16` and both, writes `bench/xf-<b>.json`
and regenerates `docs/xf-bench.md`. Tiled wins wherever M fills the 4-row tile
(MLP block 46619 vs 63077 cycles since SG9) and loses at M = 1 (1×8×8: 4282 vs 1771).

### SG8 — Stretch: single attention head (M6)
QKᵀ (matmul kernel reuse), scale, causal mask, fixed-point softmax
(max-subtract, exp LUT, sum, reciprocal), ×V.

Accept: bit-exact vs `golden.softmax_lut` pipeline; within TOL of nanoGPT
single-head causal attention; rows sum to the fixed-point one within 1 LSB.
Verify: `make test_xf_attention`

Chain (SG8, `ln_1` added by SG13): `xf/attention.py` `launches(img, kind, acts=None)` runs
h (`ln_1`: rmsnorm launch, contract 4, nanoGPT `ln_1` weight) → q, k, v (matmul) → `kT` (`kernels/transpose.asm`) → scores = q @ kT (matmul,
1/√hd folded into `scores_M`) → p (`kernels/softmax.asm`, causal, one thread
per element, each scanning its row) → o = p @ v → y = o @ `w_o` (c_proj) on
`test/fixtures/xf/attention_<0-3>.json` (1847 words: `x, norm_w, rsqrt_lut[1024], norm_M, norm_s, h`, then the SG8 segments); `attention_int` is the
golden chain, `check_bounds` its int32 bounds. `one = 127` so p stays int8
for the p @ v matmul; `exp_lut[d + 255] = round(exp(d·scale(scores))·2^23)`,
2^23 being the largest power keeping `e·one + S/2` in int32 at T = 4.
Bindings `elementwise.transpose(img, x, out)`, `softmax(img, scores, lut,
out, one)`. Row sums: `attention.row_sum_error(lut, one, T)` enumerates all
256^(n-1) rows of each length n ≤ T (≈1 s); `export --check` and
`xf/tests/test_attention.py` require ≤ 1 for the exported LUT, and
`test_xf_attention` asserts it on every simulated p. No ISA changes.

### SG9 — AUDIT-1 H1: scale-invariant integer RMSNorm
The bucketed rsqrt LUT (`Σx² >> 7`, bucket midpoints) mis-normalizes rows with small Σx²
(bucket 0 is wrong by up to 95 h LSB), so the MLP block exceeds TOL against nanoGPT on
in-distribution inputs (7.77 LSB, seed 7) and on most quiet inputs. See AUDIT-1.md H1.
Changes contract 4 (`rmsnorm_int` lookup) and the rmsnorm args, so per Rules it goes to the
user before the build starts.

Accept: rsqrt lookup with bounded relative error for every Σx² in [1, d·128²] (e.g. normalize Σx²
by an even power of two before indexing, no ISA change); kernel bit-exact vs the new golden on
the SG5 rmsnorm cases and the SG6 chain; MLP block within TOL of nanoGPT on
`rng(7).standard_normal((2000, 4, 8))` scaled by 1.0, 0.5, 0.3 and 0.1; the three
`xf/tests/test_audit1.py` xfail markers removed and passing; fixtures regenerated and TOL
re-measured in contract 8; since SG10 is done, also remove `test_mlp_tol_on_heldout`'s marker
and add `"mlp_block"` to `export.HELDOUT_TOL`.
Verify: `NANOGPT_DIR=... python3 -m xf.export --check && NANOGPT_DIR=... python3 -m pytest xf/tests -q && make test_xf_rmsnorm && make test_xf_mlp_block`

Done: contract 4 now normalizes Σx² by powers of 4 (`golden.rsqrt_index`, at most `kmax` = 4 steps at d = 8)
and adds the steps to the requant shift; rsqrt is within 2^-8 relative error for every reachable Σx²
(`test_export.py`). The kernel's normalization is branch-free inside a uniform `kmax`-trip loop, so rows
sharing a block may differ in k; `elementwise.rmsnorm` requires `s + kmax ≤ 30`. `norm_shift` is gone.
MLP TOL re-measured in contract 8 (1.42 LSB held-out); the rmsnorm launch costs 9531 cycles (was 7175).
No ISA change, no open issues.

### SG10 — AUDIT-2 H1: calibration headroom
Max-abs calibration saturates activations on held-out inputs from the calibration distribution:
MLP 13.08 LSB (4096 calibration inputs), attention 8.30 LSB (65,536), against TOLs of 2 and 3.
More calibration inputs only make the tail rarer. See AUDIT-2.md H1 and M1. Exporter-only: scales
live in the image, so no contract or kernel change.

Accept: every non-input activation scale (MLP h, gate, up, act, down, out; attention q, k, v,
scores, o, y) calibrated with headroom so that no activation saturates on
`rng(12345).standard_normal((200000, 4, 8))`; `export --check` measures both blocks on ≥ 200k
held-out inputs and asserts TOL; TOL stays 2 / 3 output LSB with the new absolute values and
measured maxima recorded in contract 8; fixtures regenerated, all xf cocotb tests bit-exact;
`test_attention_tol_on_heldout` xfail marker removed and passing. `test_mlp_tol_on_heldout` also
needs SG9: whichever of SG9/SG10 lands second removes its marker. SG9 and SG10 both regenerate
the fixtures and edit `xf/export.py`, so they do not run concurrently.
Done: `HEADROOM` = 1.5 removes all held-out saturation and attention is within TOL on 200k
(contract 8). Open: the MLP held-out TOL still fails on unsaturated AUDIT-1 H1 rows (12 of 200k,
4.84 LSB), so SG9 removes the `test_mlp_tol_on_heldout` marker and adds `"mlp_block"` to
`export.HELDOUT_TOL`. `attention_int` now also takes a batch of inputs.
Verify: `NANOGPT_DIR=... python3 -m xf.export --check && NANOGPT_DIR=... python3 -m pytest xf/tests -q && make test_xf_mlp_block && make test_xf_attention`

### SG11 — AUDIT-3 H1: L1-D cache tests
`test_l1d_indepth` accepts any hit/miss counts and builds without the cache by default; a cache that
returns stale data after a write hit, or never hits, passes every target. See AUDIT-3.md H1. Test-only:
no RTL change expected (a directed read-after-write test passes at HEAD).

Accept: directed tests for read-after-write by the same thread and for a write by one block read by a
later block, at WORD_BITS 8 and 32; exact hit/miss counts for the hot-address case with and without
MEDIC; `test_l1d_indepth` run under DCACHE_LINES=16 (with and without MEDIC) by the SG12 matrix target.
Both AUDIT-3 H1 mutants (write hit leaves the line stale; a fill never sets `valid`) make the new tests
fail, and the unmutated RTL passes.
Verify: `make test_l1d_indepth PARAMS=-Pgpu.DCACHE_LINES=16 && make test_l1d_indepth PARAMS="-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1"`, then each AUDIT-3 H1 `sed` mutant makes the first command fail.

Done: `test_xf_l1d` runs the same tests at WORD_BITS=32; each test reads DCACHE_LINES, MEDIC and NUM_CORES from the DUT, so the SG12 matrix covers both widths. Both mutants fail at both widths; the tests pass at NUM_CORES 1/2/4, DCACHE_LINES=4 COALESCE=1 and all five. Open issue: under MEDIC=1 the hot-address case is 0 hits / 12 misses. Every lane's hit window starts empty, so it is low-utility, never installs a line and can never leave bypass; MEDIC=1 turns the cache off, which `docs/final-report.md` describes as correct bypass. The test pins this; an RTL fix changes the expected counts.

### SG12 — AUDIT-3 H2: configuration matrix and full-suite targets
No make target runs the whole suite or any parameter-gated RTL; an I-cache that hangs every kernel
passes every documented command, and `make test_xf_rmsnorm PARAMS=-Pgpu.NUM_CORES=1` fails today on the
20,000-cycle launch timeout. See AUDIT-3.md H2.

Accept: a make target that runs every test module once at default parameters (plus
`harness_selftest`), and one that runs `regression`, `regression_xf` and every `test_xf_*` under
default; ICACHE_LINES=16; PIPELINED=1; COALESCE=1; DCACHE_LINES=16; DCACHE_LINES=16 MEDIC=1; all five;
NUM_CORES=1; NUM_CORES=4, plus a run with non-zero `mem_latency`, each configuration in its own build
directory so they can run in parallel; both exit non-zero on any failure; the elementwise/RMSNorm
launch timeout raised so NUM_CORES=1 passes; the AUDIT-3 H2 I-cache mutant fails the matrix target;
the Dispatch protocol's "full existing suite" names the new targets.
Verify: the new full-suite and matrix targets pass from a clean checkout, and the AUDIT-3 H2 `sed` mutant makes the matrix target fail.

Done: `make test_all` (22 cocotb modules, 3 warp-scheduler targets, `harness_selftest`, xf pytest) and `make -j4 matrix` (regression and `test_l1d_indepth` at 8 bits, regression and every `test_xf_*` at 32 bits, per configuration in `build/matrix/<cfg>`; the latency run uses `MEM_LATENCY=3`). Both green; the matrix takes about 53 min at -j4 on 4 cores, NUM_CORES=1 and the latency run being the long poles. The I-cache mutant fails `matrix_icache` (matmul, conv1d, divergent_relu, tiled_matmul time out). `regression`/`regression_xf` now run every module and list the failures instead of stopping at the first.

### SG13 — AUDIT-5 H1: ln_1 in front of the attention head
The head runs on raw `x`, but nanoGPT's block runs `attn(ln_1(x))`. Without the norm its q, k, v and scores
scale with the input, so it exceeds TOL on louder inputs (55.8 LSB at 3× the calibration scale, 98.6 LSB on
uniform int8) and passes vacuously on quiet ones (an all-zero `y` passes 99.9 % of inputs at 0.05×). See
AUDIT-5.md H1. Reuses the rmsnorm kernel and `rmsnorm_int`; no ISA or contract 1–7 change, but the
attention fixture layout (SG8 chain) and contract 8's attention entry change.

Accept: attention chain is `ln_1` (rmsnorm launch, nanoGPT `ln_1` weight) → q, k, v → … → y, with the
reference `blk.attn(blk.ln_1(x))` (adding the residual `x + …` is optional, round-1 M1); scales calibrated on
the new chain; within TOL of nanoGPT on `rng(7).standard_normal((2000, 4, 8))` scaled by 0.1, 0.3, 1, 2 and 3
and on the 200k held-out inputs; attention TOL re-measured in contract 8; fixtures regenerated and
`test_xf_attention` bit-exact; the `xf/tests/test_audit5.py` xfail markers removed and passing (the test
updated to the new reference).
Verify: `NANOGPT_DIR=... python3 -m xf.export --check && NANOGPT_DIR=... python3 -m pytest xf/tests -q && make test_xf_attention`

Done: the head runs `ln_1` through the existing rmsnorm kernel and `rmsnorm_int` (`export.rmsnorm_lut`, shared
with the MLP; MLP fixtures unchanged), calibrated on the new chain; the reference is `blk.attn(blk.ln_1(x))`
without the residual. Within TOL at every scale from 0.05× to 10×, on uniform and ±127 int8 inputs, and on
the 200k held-out inputs (contract 8); an all-zero `y` now fails on every quiet input, so the check is no
longer vacuous. `test_xf_attention` is bit-exact; the bench has no attention entry, so `docs/xf-bench.md`
is unchanged. No ISA or contract 1–7 change. Open: the held-out max (2.976 LSB) sits close to TOL, and 1 of
1M fresh inputs reaches 3.18 LSB (the AUDIT-2 M3 tail, previously 3.614). Both `test_audit6.py` repros now pass
(hill-climbed worst 1.74 LSB, outlier channel 2.83 LSB, 0 of 200k over TOL), so their markers are removed; the
hill climb has not been re-run against this chain, which stays SG14's first step.

### SG14 — AUDIT-6 H1: attention scores precision
`scores` is computed from int8 `q` and `k`, whose rounding errors are multiplied by the other operand
in the dot product; inputs that saturate nothing reach 10.66 y LSB against TOL 3, and one-channel-x4
inputs exceed TOL on 61 of 200k. The Gaussian held-out maximum (2.75 LSB) is not a bound. See AUDIT-6.md H1.
Any fix changes contract 4 (matmul operands), the attention layout, or contract 8's TOL, so per Rules
it goes to the user before the build starts.

Runs after SG13, which may absorb most of it: in a float emulation of SG13's `ln_1` chain the same search
reaches 2.83 LSB. First port `test_audit6.py` to SG13's chain and re-run the AUDIT-6 hill climb; if the chain
stays within TOL, SG14 reduces to the `export --check` addition below.

Accept: either `scores` precision raised (e.g. wider `q`/`k` operands for the scores matmul, or `Wq·Wkᵀ`
folded so one operand is exact) or the attention TOL re-derived from a worst-case error bound, recorded in
contract 8; both `xf/tests/test_audit6.py` xfail markers removed and passing; kernels bit-exact vs the new
golden; `export --check` also asserts TOL on a non-Gaussian unsaturated set (e.g. the outlier-channel inputs).
Verify: `NANOGPT_DIR=... python3 -m xf.export --check && NANOGPT_DIR=... python3 -m pytest xf/tests -q && make test_xf_attention`

Done: on SG13's chain `q`/`k` rounding no longer drives the error: with `q` and `k` unrounded the hill climb
reaches the same maximum, and folding `Wq·Wkᵀ` gives the same. The search over unsaturated inputs reaches 4.9 LSB
(5.35 after refining the best points) from several stacked roundings, and int16 `h` alone would still reach
3.43 LSB (float emulation), so per the user's decision (2026-10-14) the attention TOL is re-derived as 6 LSB (contract 8) with no kernel, layout or
contract 1–7 change; fixtures differ only in `tol`. `test_audit6.py` asserts it on the five climbed inputs (two above 5 LSB), one
channel ×4 for every channel, and fresh Gaussian, Laplace and Student-t inputs (200k each); `export --check`
also asserts TOL on the channel-5 ×4 set for both blocks. Open: AUDIT-6 L1 (the held-out seed tuned `HEADROOM`).

### SG15 — AUDIT-7 H1: torch-gated checks in the merge gate
Without torch and `NANOGPT_DIR`, `make test_all` skips 13 tests, every one that checks more than the 4
committed fixtures per block, and never runs `export --check`; a full revert of SG10 passes it. Contract 8's
no-saturation guarantee is asserted only in `export.check`, so no headroom on `act` passes pytest even with
torch. See AUDIT-7.md H1. Test-only.

Accept: `make test_all` fails (does not skip) when torch or `NANOGPT_DIR` is missing, and runs
`python3 -m xf.export --check`; the held-out saturation assert runs under pytest; the Dispatch protocol's
full-suite command names `NANOGPT_DIR`; each AUDIT-7 H1 mutant (re-exported) fails `make test_all`, and HEAD passes.
Verify: `NANOGPT_DIR=... make test_all`, `env -u NANOGPT_DIR make test_all` exits non-zero, then each H1 mutant fails the first.

### SG16 — AUDIT-7 H2: per-segment gain and RMS checks
Tolerance bounds only the per-element max error of `out` and `y`, so a SiLU LUT indexed one gate LSB off
(`act`, `down` gain +3.4 %), `p` dequantized at 1/128 (`y` −0.9 %) and a 0.5 % residual gain pass every check,
`export --check` included. See AUDIT-7.md H2. Adds bounds to contract 8, so per Rules it goes to the user
before the build starts; after SG13 and SG14, which change the attention chain.

Accept: `export --check` and a pytest compare every non-input golden segment of both blocks against its float
activation on the held-out inputs, asserting least-squares gain and RMS error within bounds measured at HEAD
and recorded in contract 8; each of the three AUDIT-7 H2 mutants (re-exported) fails both.
Verify: `NANOGPT_DIR=... python3 -m xf.export --check && NANOGPT_DIR=... python3 -m pytest xf/tests -q`, then each H2 mutant fails both.

## Round 3 — completing the GPU (approved by the user on 2026-10-19)

Goal: every feature the README and final report claim is wired, measured and tested: parallel memory
channels with exact counters, a MeDiC that can cache, a multi-warp core running the warp scheduler,
real `BAR`, and SIMT-stack divergence. The xf track only gets SG16 and the robustness items.

Facts checked at `2adb710` (iverilog 12.0, sv2v 0.0.13, cocotb 2.0.1):
- **Controller race.** `controller.sv` claims a consumer with `channel_serving_consumer[j] <= 1`; later
  channels in the same loop iteration do not see it, so every idle channel takes the same consumer.
  Probe (8 threads, each `LDR` a distinct address): at most 1 distinct address on the 4 channels in any
  cycle, 16 cycles with duplicate transactions, 145 cycles. Upstream used `=`. With `=` on the three
  claims: 4 distinct addresses in flight, no duplicates, 97 cycles (135 vs 207 at `MEM_LATENCY=3`).
  `mem_requests` then reads 4 for 8 loads, because `<= x + 1` inside the channel loop counts once per
  cycle; `mem_requests_coalesced` likewise counts once per transaction, not per merged consumer.
- **I-cache totals** sum cores 0 and 1 only (`gpu.sv` "Reduction tree").
- **MeDiC** never installs: every lane's hit window starts at 0 (SG11 open issue), so `MEDIC=1` is a
  cache-off switch and the priority pass never fires.
- **Not wired:** `warp_scheduler.sv` (standalone), `decoded_bar` (ignored), SIMT stack (ADR-006 design).
  Branches follow lane 0 (AUDIT-4 M1). No opcode is free.

Order: SG17 alone. Then SG18 ∥ SG22 (∥ SG16, xf-only). SG19 and SG20 after SG18; SG21 after SG20;
SG23 last. At most two sub-goals build at once. Each sub-goal is one PR; its thread splits the work
into bounded, test-driven subagent tasks.

Verification latency budget (measured in SG17, recorded here): every sub-goal's Verify ≤ 10 min on
4 cores; `make -j4 matrix` ≤ 60 min; the differential harness ≤ 2 s of wall time per 1k
simulated cycles. SG17 records simulated cycles/s per configuration as the baseline; a sub-goal that
slows any matrix configuration by > 20 % stops and reports.

Baseline (SG17, 4 cores): `make -j4 matrix` 48 min at `2adb710`, 46 min with SG17's differential
targets added; `make test_all` 15 min. Differential harness, simulated cycles/s (8-bit / 32-bit):

| default | icache | pipelined | coalesce | dcache | medic | all | cores1 | cores4 | latency |
|---|---|---|---|---|---|---|---|---|---|
| 4280 / 4060 | 4290 / 4110 | 4330 / 3790 | 4170 / 3930 | 3340 / 3920 | 3220 / 3350 | 3570 / 3060 | 4700 / 4370 | 3340 / 3140 | 4090 / 4060 |

### SG17 — Round 3 shared interfaces
Contracts 9–12 below as code and tests, with no RTL behaviour change.

9. **ISA reference model** — `xf/isasim.py`: `run(program, data, threads, *, word_bits=8, addr_bits=8,
   threads_per_block=4, warps_per_core=1, smem_size=64, simt=False, stack_depth=8, max_steps=1_000_000,
   warp_policy="lrr") -> Result(memory, steps, divergent, stores_conflict)`; errors raise `IsaSimError`.
   `warp_policy` is `"lrr"` or `"gto"`; missing-`BAR` races show only under `"gto"` (ADR-012). Lockstep blocks,
   per-lane NZP and active mask, SMEM per block, 8/32-bit wrap, the semantics of `docs/isa.md` plus the
   reconvergence encoding of contract 10. It flags divergent branches when the SIMT stack is off.
10. **Reconvergence encoding** — `BR` with `nzp = 000` (never taken today; AUDIT-4 L3) becomes
   `SYNC imm`: push `imm` as the reconvergence PC of the next divergent branch. `xf/isa.py`,
   `xf/asm.py` (`SYNC label`), `docs/isa.md` and ADR-011 change in the same commit. Kernels without
   `SYNC` keep today's semantics.
11. **Trace fixtures** — `test/fixtures/isa/<case>.json`: `{program, data, threads, params, memory,
   cycles?}`; `memory` from the model. A differential cocotb test runs every fixture on the RTL and
   compares memory bit-exactly. Fixtures: every existing kernel test program, the xf kernels on small
   images, and seeded random straight-line programs (AUDIT-8's 92-instruction generator).
12. **Multi-warp and telemetry contract** — `gpu.WARPS_PER_CORE` (default 1, bit-identical cycles),
   `gpu.WARP_POLICY` (0 LRR, 1 GTO, 2 GTLRR; `warp_scheduler.sv` ports unchanged); new outputs
   `warp_issues_total`, `bar_stall_cycles_total`. Counter semantics: `data_mem_requests` = downstream
   transactions; `data_mem_requests_coalesced` = consumer requests served by another's transaction;
   `icache_*_total` = sum over all cores.

Accept: the model reproduces every existing kernel test's memory result and every xf kernel bit-exact
vs `xf.golden`; the differential test passes on all fixtures under the 10 matrix configurations;
assembler round-trips `SYNC`; cycles/s baseline recorded above; no RTL change.
Verify: `python3 -m pytest xf/tests -q && make test_isa_diff && make test_xf_isa_diff`

### SG18 — Controller: parallel channels, exact telemetry
Claim consumers with blocking assignments (as upstream); count requests and merged consumers with
blocking accumulators; sum I-cache counters over `NUM_CORES`.

Accept: a directed test sees `DATA_MEM_NUM_CHANNELS` distinct addresses in flight and no duplicate
transactions; `data_mem_requests` and `data_mem_requests_coalesced` equal counts derived from the
memory model's serviced transactions in every matrix configuration; I-cache totals equal the per-core
sum at `NUM_CORES` 1, 2, 4; all suites pass; `bench/` and `docs/xf-bench.md` re-recorded.
Verify: `make test_controller && make -j4 matrix`, then reverting either claim to `<=` fails `test_controller`.

### SG19 — MeDiC that can install and hit
Lanes start high-utility (or a neutral warm-up of N accesses before classification), so a hot lane
caches and a streaming lane is bypassed; priority applies to high-utility lanes.

Accept: under `MEDIC=1` the hot-address case hits (exact counts, `test_l1d_indepth` updated); a
mixed workload (hot lanes and streaming lanes in one launch) shows the streaming lanes bypassed and the
hot lanes' hit rate ≥ the `MEDIC=0` hit rate; the priority mutant (`can_consider` always 1) fails a test.
Verify: `make test_l1d_indepth PARAMS="-Pgpu.DCACHE_LINES=16 -Pgpu.MEDIC=1" && make test_medic`

### SG20 — Multi-warp core
`WARPS_PER_CORE` blocks resident per core, per-warp PC, NZP, registers, LSU state and SMEM partition;
`warp_scheduler` picks the warp each issue slot; `BAR` waits until every warp of the block arrives.

Accept: `WARPS_PER_CORE=1` matches every current cycle count exactly; at 2 and 4 every fixture
matches the model; a barrier test fails when `BAR` is made a no-op; the three policies give the issue
orders `test_warp_scheduler_*` already assert, now inside the core; added to the matrix.
Verify: `make test_isa_diff PARAMS="-Pgpu.WARPS_PER_CORE=4" && make test_multiwarp && make -j4 matrix`

### SG21 — SIMT-stack control divergence
Per-warp active mask and a stack of `{reconvergence PC, pending PC, mask}`; `SYNC` pushes, a divergent
`BR` splits, reaching the reconvergence PC pops. Disabled lanes do not write registers, NZP or memory.

Accept: the AUDIT-4 M1 program stores `[1, 2, 2, 2]`; nested divergence and loops with per-lane trip
counts match the model on seeded random structured programs; kernels without `SYNC` keep their cycle
counts; stack overflow is a simulation `$fatal`.
Verify: `make test_divergence && make test_isa_diff`

### SG22 — Kernel and binder robustness
AUDIT-4 M2 (`matmul_tiled` works for any `blockDim` that divides 64-word SMEM, or the binder refuses),
L3 (assembler rejects `BR` past the program and `BRp`), L4 (binders reject zero `K`, `d`, `T`),
AUDIT-8 L1/L2 (dtype checks, rmsnorm header), and merge AUDIT-8.

Accept: `test_audit4.py` and `test_audit8.py` xfail markers removed and passing; `make test_xf_matmul
PARAMS=-Pgpu.THREADS_PER_BLOCK=8` passes or is refused by the binder with a clear error.
Verify: `python3 -m pytest xf/tests -q && make test_xf_matmul PARAMS=-Pgpu.THREADS_PER_BLOCK=8`

### SG23 — Benchmarks and docs
`bench/run.py` sweeps `MEM_LATENCY` 0/3/10, `WARPS_PER_CORE` 1/2/4 × policy, and a MeDiC mixed
workload; `docs/final-report.md`, `ROADMAP.md`, `docs/architecture.md` and ADR-006/007/008/009 rewritten
to match the code.

Accept: reproducible numbers (two runs equal); every claim in the final report points at a bench row;
AUDIT-4 L2's stale-doc list is empty.
Verify: `python3 bench/run.py --phase 11 --label phase-11-all` twice, identical JSON.

### SG24 — AUDIT-12 coverage gaps
H1: a test in `test_all` and the `coalesce`/`all` matrix configs where two lanes load `a` and `a + 256k`
in the same cycle at 16-bit addresses. H2: exact `icache_hits_total`/`icache_misses_total` asserted for a
fixed program. H3: `docs/isa.md` states that the highest lane wins an STS conflict (with an ADR), and an
ISA fixture pins it in the RTL.

Accept: AUDIT-12 mutants K08, I02, I04 and S02 are killed by `make test_all` or a matrix config;
the H1–H3 xfails in `xf/tests/test_audit12.py` turn into passing tests.
Verify: `NANOGPT_DIR=... make test_all`; `make -j4 matrix`; each mutant rerun from AUDIT-12.md fails.

### SG25 — AUDIT-14 store conflicts
H1: when several lanes of one STR hit the same word, the RTL keeps the lowest lane of the last channel
group while `xf.isasim` keeps the highest lane. Make them agree on "highest lane wins" (the model's and
SG24's STS rule), state it in `docs/isa.md` with an ADR, and add ISA fixtures with STR conflicts. Also
L1 (the lrr == gto interleaving check admits `sts_conflict*` at more than one warp) and L2 (the RET row).

Accept: AUDIT-14 H1, L1 and L2 xfails pass; STR-conflict fixtures pass at 8 and 32 bits in every matrix
config.
Verify: `NANOGPT_DIR=... make test_all`; `make -j4 matrix`.

### SG26 — AUDIT-13 SMEM and launch limits
H1: no RTL test touches SMEM words ≥ 64; add ISA fixtures that store and load high SMEM words (and the
wrap/mask at `SMEM_SIZE`) so mutants SM1/SM2 die. H2: no test launches 253–255 threads; add fixtures and a
binder test at 253 and 255 threads so dispatcher mutant B02 dies.

Accept: AUDIT-13 mutants SM1, SM2 and B02 are killed by `make test_all` or a matrix config; the H1/H2
xfails in `xf/tests/test_audit13.py` pass.
Verify: `NANOGPT_DIR=... make test_all`; `make -j4 matrix`; each mutant rerun from AUDIT-13.md fails.

### Round 3 audits
After SG23, separate sessions that wrote none of the code, one lens each: **AUDIT-9 encoding
soundness** (ISA/model/RTL agreement, divergence with no false negatives: differential fuzz must catch
every seeded mutant of the stack and the decoder), **AUDIT-10 performance under load** (latency,
4 warps × 4 cores, channel utilisation, sim throughput against the budget), **AUDIT-11 API/SDK
consistency** (`xf/` bindings, assembler, `docs/isa.md`, ADRs, Makefile targets), **AUDIT-12 test
coverage** (mutation). Critical and high findings become SG rows; rounds repeat until one full round has
none.

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
