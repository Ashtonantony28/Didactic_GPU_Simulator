"""ISA trace fixtures (contract 11): test/fixtures/isa/<case>.json, memory from xf.isasim.

    python3 -m xf.isa_fixtures [--check]

{program, data, threads, params, memory}: `params` are xf.isasim.run keywords and `word_bits` picks the
build (8: sim.vvp via test_isa_diff, 32: sim_xf.vvp via test_xf_isa_diff). `data` and `memory` omit
trailing zero words. Cases the model flags as divergent or store-conflicting are not written.
"""
import argparse
import asyncio
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

from .asm import assemble
from .isasim import run
from .kernels import KERNEL_DIR, check_block

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "test" / "fixtures" / "isa"
WIDTHS = {8: 8, 32: 12}   # word_bits -> addr_bits
# Reads another block's writes: the model runs blocks serially, the RTL NUM_CORES at a time.
CROSS_BLOCK = {"l1d_later_block8", "l1d_later_block32"}
EDGE = {8: [0, 1, 7, 8, 9, 127, 128, 255], 32: [0, 31, 32, 33, 2**31 - 1, 2**31, 2**32 - 1]}
OPS = ("ADD", "SUB", "MUL", "DIV", "SRA", "MAX", "CONST")
# Every lane of a block stores its thread id to SMEM[0] in one STS; the highest lane wins (docs/isa.md). BAR makes the
# result independent of the warp interleaving.
STS_CONFLICT = """
    MUL R2, %blockIdx, %blockDim
    ADD R2, R2, %threadIdx
    CONST R0, #0
    STS R0, R2
    BAR
    LDS R1, R0
    STR R2, R1
    RET
"""


def straight_line(seed, w, threads=8, ops=44):
    """AUDIT-8's fuzz shape, 92 instructions: each thread loads R2-R12, runs `ops` random ALU/CONST
    instructions on them and stores R2-R12 back. Operands mix edge words and random words."""
    rng = random.Random(seed)
    regs = range(2, 13)
    src = [f"R{r}" for r in regs] + ["%blockIdx", "%blockDim", "%threadIdx"]
    lines = ["MUL R0, %blockIdx, %blockDim", "ADD R0, R0, %threadIdx", f"CONST R1, #{threads}"]
    lines += [x for r in regs for x in (f"LDR R{r}, R0", "ADD R0, R0, R1")]
    for _ in range(ops):
        op, rd = rng.choice(OPS), rng.choice(regs)
        lines.append(f"CONST R{rd}, #{rng.randrange(256)}" if op == "CONST"
                     else f"{op} R{rd}, {rng.choice(src)}, {rng.choice(src)}")
    lines += [x for r in regs for x in (f"STR R0, R{r}", "ADD R0, R0, R1")] + ["RET"]
    data = [rng.choice(EDGE[w]) if rng.random() < 0.5 else rng.randrange(1 << w) for _ in range(len(regs) * threads)]
    return assemble("\n".join(lines)), data, threads


def _kernel_cases():
    """Every cocotb kernel test program at its own width, the 8-bit ones also at 32 bits."""
    from test import test_isa_ext, test_param_sweep, test_relu
    from xf.tests.test_isasim import LEGACY_CASES

    cases = {}
    for name, (prog, data, threads, _, w, _) in LEGACY_CASES.items():
        for width in ((8, 32) if w == 8 and not name.startswith("l1d") else (w,)):
            cases[name if name.startswith("l1d") else f"{name}{width}"] = (prog, data, threads, width)
    for w in WIDTHS:
        cases[f"param_sweep{w}"] = (test_param_sweep.PROGRAM, test_param_sweep.DATA, 4, w)
        cases[f"relu_branch{w}"] = (test_relu.PROGRAM, test_relu.DATA, test_relu.THREADS, w)
        pairs = test_isa_ext._operand_pairs(w)
        n = len(pairs)
        cases[f"isa_ext_sra_max{w}"] = (assemble(test_isa_ext.SRA_MAX.format(n=n)),
                                        [a for a, _ in pairs] + [b for _, b in pairs], n, w)
        mask = (1 << w) - 1
        vals = [0, 1, 2, mask, 1 << (w - 1), (1 << (w - 1)) - 1]
        pairs = [(a, b) for a in vals for b in vals]
        n = len(pairs)
        cases[f"isa_ext_cmp{w}"] = (assemble(test_isa_ext.CMP_FLAGS.format(n=n)),
                                    [a for a, _ in pairs] + [b for _, b in pairs], 4 * n, w)
    return {k: v for k, v in cases.items() if k not in CROSS_BLOCK}


def _xf_cases():
    """The cheapest launch of each kernel in each xf test module, with the memory image it starts from."""
    from xf.tests import test_isasim

    skel = lambda p: tuple(x & 0xFF00 if i in (0, 4) else x for i, x in enumerate(p))
    names = {skel(assemble(f.read_text())): f.stem for f in sorted(KERNEL_DIR.glob("*.asm"))}
    best = {}

    async def record(dut, image, launches, mem_latency=0):
        words = list(image.words)
        for ln in launches:
            check_block(ln.max_block, 4, ln.name)
            r = run(ln.program, words, ln.threads, word_bits=image.word_bits, addr_bits=image.addr_bits)
            key = f"xf_{mod.__name__.split('test_xf_')[-1]}_{names[skel(ln.program)]}"
            if key not in best or r.steps < best[key][0]:
                best[key] = (r.steps, (list(ln.program), words, ln.threads, image.word_bits))
            words = r.memory
        return [], SimpleNamespace(memory=words)

    mods = test_isasim.XF_MODS
    saved = [vars(m).get("run_launches") for m in mods]
    for m in mods:
        m.run_launches = record
    try:
        for mod, name in test_isasim.XF_TESTS:
            asyncio.run(getattr(mod, name).func(None))
    finally:
        for m, f in zip(mods, saved):
            if f is None:
                del m.run_launches
            else:
                m.run_launches = f
    return {k: v for k, (_, v) in sorted(best.items())}


def _trim(words):
    words = list(words)
    while words and not words[-1]:
        words.pop()
    return words


def cases():
    out = _kernel_cases()
    for w in WIDTHS:
        out.update({f"random{w}_{s}": (*straight_line(s, w), w) for s in range(8)})
        out[f"sts_conflict{w}"] = (assemble(STS_CONFLICT), [], 8, w)
    out.update(_xf_cases())
    return out


def build():
    """-> ({file name: bytes}, [skipped case names])."""
    files, skipped = {}, []
    for name, (prog, data, threads, w) in sorted(cases().items()):
        params = {"word_bits": w, "addr_bits": WIDTHS[w]}
        r = run(prog, data, threads, **params)
        if r.divergent or r.stores_conflict:
            skipped.append(name)
            continue
        d = {"program": [int(x) for x in prog], "data": _trim(int(x) & ((1 << w) - 1) for x in data),
             "threads": threads, "params": params, "memory": _trim(r.memory)}
        files[f"{name}.json"] = (json.dumps(d, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return files, skipped


def load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify the committed fixtures instead of writing them")
    args = ap.parse_args(argv)
    files, skipped = build()
    assert build()[0] == files, "fixture generation is not deterministic"
    have = {p.name for p in FIXTURE_DIR.glob("*.json")}
    if args.check:
        stale = sorted(n for n, b in files.items() if n not in have or (FIXTURE_DIR / n).read_bytes() != b)
        extra = sorted(have - set(files))
        if stale or extra:
            sys.exit(f"stale {stale}, extra {extra}; rerun python3 -m xf.isa_fixtures")
        print(f"ok: {len(files)} ISA fixtures current, skipped {skipped}")
        return
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for n in have - set(files):
        (FIXTURE_DIR / n).unlink()
    for n, b in files.items():
        (FIXTURE_DIR / n).write_bytes(b)
    print(f"wrote {len(files)} ISA fixtures to {FIXTURE_DIR}, skipped {skipped}")


if __name__ == "__main__":
    main()
