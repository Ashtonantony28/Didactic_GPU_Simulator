"""SIMT-stack control divergence (SG21, ADR-011, ADR-015): RTL vs xf.isasim with simt = the DUT's SIMT_STACK.

`make test_divergence` builds SIMT_STACK=1 at WARPS_PER_CORE 1, 2, 4, NUM_CORES 1, 2, 4 and THREADS_PER_BLOCK 8,
and the stack off; test/simt_fatal.py holds the runs that must end in simt_stack's $fatal.
"""
import random

import cocotb

from xf.asm import assemble
from xf.isasim import IsaSimError, run
from xf.tests import test_isasim as cases  # attributes read at test time: test_isasim imports test/ modules
from .helpers.logger import logger
from .helpers.runner import data_bus, run_kernel

SEEDS = 16
IN, OUT, SLOTS, MAX_THREADS = 0, 32, 8, 24   # input word per thread at IN + g; thread g owns OUT + 8g .. OUT + 8g + 7

RET_IN_REGION = """
    SYNC join
    CONST R0, #0
    CONST R5, #5
    CMP %threadIdx, R0
    BRn rest
    STR R0, R5
    RET
rest:
    CONST R1, #7
    STR %threadIdx, R1
join:
    CONST R2, #8
    ADD R3, %threadIdx, R2
    STR R3, R2
    RET
"""
# Lane t runs min(t, cap) + 1 trips: min(tpb - 1, cap) divergent exits, two stack entries each.
CAPPED_LOOP = """
    CONST R0, #0
    CONST R1, #1
    CONST R4, #{cap}
    SUB R4, R0, R4
    SUB R2, R0, %threadIdx
    MAX R2, R2, R4
    SUB R2, R0, R2
    ADD R2, R2, R1
    CONST R3, #0
loop:
    SYNC done
    ADD R3, R3, R2
    SUB R2, R2, R1
    CMP R2, R0
    BRn loop
done:
    STR %threadIdx, R3
    RET
"""


def params(dut):
    return dict(warps=int(dut.WARPS_PER_CORE.value), policy=int(dut.WARP_POLICY.value), cores=int(dut.NUM_CORES.value),
                simt=bool(int(dut.SIMT_STACK.value)), tpb=int(dut.THREADS_PER_BLOCK.value))


def model(dut, prog, data, threads, policy="lrr", **kw):
    p, (bits, addr) = params(dut), data_bus(dut)
    return run(prog, data, threads, word_bits=bits, addr_bits=addr, threads_per_block=p["tpb"],
               warps_per_core=p["warps"], simt=p["simt"], warp_policy=policy, **kw)


async def check(dut, prog, data, threads, name):
    """RTL memory and warp_issues_total equal the model's (which must not depend on the warp interleaving)."""
    want, gto = model(dut, prog, data, threads), model(dut, prog, data, threads, "gto")
    assert (want.memory, want.steps) == (gto.memory, gto.steps), f"{name}: model depends on interleaving"
    _, mem = await run_kernel(dut, prog, data, threads, timeout_cycles=200_000)
    bad = [a for a, (g, w) in enumerate(zip(mem.memory, want.memory)) if g != w]
    assert not bad, f"{name}: {len(bad)} words differ, first at {bad[0]}: RTL {mem.memory[bad[0]]}, model {want.memory[bad[0]]}"
    issues = int(dut.warp_issues_total.value)
    assert issues == want.steps, f"{name}: warp_issues_total {issues} != model steps {want.steps}"
    return want


@cocotb.test()
async def test_audit4_m1(dut):
    """AUDIT-4 M1: [1, 2, 2, 2] with the stack; without it a one-warp block follows thread 0 and stores 1s."""
    p = params(dut)
    r = await check(dut, assemble(cases.M1.format(sync="SYNC done")), [0] * 8, 4, "m1")
    if p["simt"] or p["tpb"] // p["warps"] == 1:
        assert r.memory[:4] == [1, 2, 2, 2], r.memory[:4]
    elif p["warps"] == 1:
        assert r.memory[:4] == [1, 1, 1, 1], r.memory[:4]


@cocotb.test()
async def test_nested(dut):
    """Two levels of if/else inside one block."""
    p = params(dut)
    r = await check(dut, assemble(cases.NESTED), [], 4, "nested")
    if p["simt"]:
        assert r.memory[:4] == [91, 92, 110, 114], r.memory[:4]


@cocotb.test()
async def test_loop_per_lane_trips(dut):
    """Lane t loops t + 1 times."""
    r = await check(dut, assemble(cases.LOOP), [], 4, "loop")
    if params(dut)["simt"]:
        assert r.memory[:4] == [1, 3, 6, 10], r.memory[:4]


@cocotb.test()
async def test_ret_in_divergent_region(dut):
    """Thread 0 returns inside the region; the others reconverge without it."""
    r = await check(dut, assemble(RET_IN_REGION), [], 4, "ret")
    if params(dut)["simt"]:
        assert r.memory[:4] == [5, 7, 7, 7] and r.memory[8:12] == [0, 8, 8, 8], r.memory[:12]


@cocotb.test()
async def test_full_stack(dut):
    """At 8 lanes per warp, 4 divergent loop exits fill the 8 entries exactly (5 is simt_fatal's overflow)."""
    p = params(dut)
    if p["tpb"] // p["warps"] < 8:
        return
    prog = assemble(CAPPED_LOOP.format(cap=4))
    if p["simt"]:
        try:
            model(dut, assemble(CAPPED_LOOP.format(cap=5)), [], 8)
            raise AssertionError("cap 5 does not overflow the model's stack")
        except IsaSimError as e:
            assert "deeper than 8" in str(e), e
    r = await check(dut, prog, [], 8, "full_stack")
    if p["simt"]:
        assert r.memory[:8] == [1, 3, 6, 10, 15, 15, 15, 15], r.memory[:8]


class Structured:
    """Seeded random structured kernel: nested if/else and per-lane loops, each divergent branch right after its
    SYNC, with LDR/STR/LDS/STS on per-thread words only (so no result depends on the warp interleaving).
    R0 = global thread id, R1 = 1, R12 = 0, R2 = input (0-3), R3-R6 values, R7-R9 loop counters, R10-R11 temps."""

    VALS = ["R2", "R3", "R4", "R5", "R6", "R1", "%threadIdx", "%blockIdx"]

    def __init__(self, seed, max_len=170):
        self.rng, self.max_len, self.labels, self.lines = random.Random(seed), max_len, 0, []

    def label(self):
        self.labels += 1
        return f"L{self.labels}"

    def emit(self, *lines):
        self.lines += lines

    def out_addr(self, slot):
        self.emit(f"CONST R10, #{SLOTS}", "MUL R10, R0, R10", f"CONST R11, #{OUT + slot}", "ADD R10, R10, R11")

    def smem_addr(self, slot):
        self.emit("CONST R10, #2", "MUL R10, %threadIdx, R10", f"CONST R11, #{slot}", "ADD R10, R10, R11")

    def condition(self):
        """CMP of a small per-lane value against a constant; returns the branch condition."""
        rng, kind = self.rng, self.rng.randrange(3)
        if kind == 0:
            self.emit(f"CONST R10, #{rng.randrange(4)}", "CMP R2, R10")
        elif kind == 1:
            self.emit(f"CONST R10, #{rng.choice((1, 2))}", "DIV R11, %threadIdx, R10", f"CONST R10, #{rng.randrange(3)}",
                      "CMP R11, R10")
        else:
            self.emit(f"CONST R10, #{rng.randrange(4)}", f"CMP {rng.choice(('R3', 'R4', 'R5', 'R6'))}, R10")
        return rng.choice(("BRz", "BRn"))

    def statement(self, depth, loops):
        rng = self.rng
        kinds = ["alu"] * 4 + ["mem"] * 3
        if depth < 3:
            kinds += ["if"] * 2 + ["loop"] * (loops < 3)
        if depth:
            kinds += ["ret"] * (rng.random() < 0.15)
        kind, rd = rng.choice(kinds), rng.choice(("R3", "R4", "R5", "R6"))
        if kind == "alu":
            op = rng.choice(("ADD", "SUB", "MUL", "DIV", "SRA", "MAX", "CONST"))
            self.emit(f"CONST {rd}, #{rng.randrange(256)}" if op == "CONST"
                      else f"{op} {rd}, {rng.choice(self.VALS)}, {rng.choice(self.VALS)}")
        elif kind == "mem":
            m = rng.randrange(5)
            if m == 0:
                self.emit(f"LDR {rd}, R0")
            elif m in (1, 2):
                self.out_addr(rng.randrange(4))
                self.emit(f"STR R10, {rng.choice(self.VALS[:5])}" if m == 1 else f"LDR {rd}, R10")
            else:
                self.smem_addr(rng.randrange(2))
                self.emit(f"STS R10, {rng.choice(self.VALS[:5])}" if m == 3 else f"LDS {rd}, R10")
        elif kind == "if":
            then, end = self.label(), self.label()
            self.emit(f"SYNC {end}")
            br = self.condition()
            self.emit(f"{br} {then}")
            self.body(depth + 1, loops)
            self.emit(f"BRnzp {end}", f"{then}:")
            self.body(depth + 1, loops)
            self.emit(f"{end}:")
        elif kind == "loop":
            top, end, ctr = self.label(), self.label(), f"R{7 + loops}"
            src = rng.choice(("R2", "%threadIdx"))
            self.emit(f"CONST R10, #{rng.choice((1, 2, 4))}", f"DIV {ctr}, {src}, R10", f"ADD {ctr}, {ctr}, R1",
                      f"{top}:")
            self.body(depth + 1, loops + 1)
            self.emit(f"SUB {ctr}, {ctr}, R1", f"SYNC {end}", f"CMP {ctr}, R12", f"BRn {top}", f"{end}:")
        else:
            self.out_addr(SLOTS - 1)
            self.emit("STR R10, R3", "RET")

    def body(self, depth, loops):
        for _ in range(self.rng.randint(1, 3)):
            if sum(not ln.endswith(":") for ln in self.lines) > self.max_len - 40:
                break
            self.statement(depth, loops)

    def program(self):
        self.emit("MUL R0, %blockIdx, %blockDim", "ADD R0, R0, %threadIdx", "CONST R1, #1", "CONST R12, #0",
                  "LDR R2, R0", "CONST R3, #3", "CONST R4, #4", "CONST R5, #5", "CONST R6, #6")
        while sum(not ln.endswith(":") for ln in self.lines) < self.max_len // 2:
            self.body(0, 0)
        for slot, r in zip(range(4, 8), ("R3", "R4", "R5", "R6")):
            self.out_addr(slot)
            self.emit(f"STR R10, {r}")
        self.emit("RET")
        return assemble("\n".join(self.lines))


def depth_needed(prog, data, threads, tpb):
    """Smallest stack_depth the model runs the program with at one warp."""
    for d in range(0, 2 * tpb):
        try:
            run(prog, data, threads, threads_per_block=tpb, simt=True, stack_depth=d)
            return d
        except IsaSimError as e:
            if "deeper" not in str(e):
                raise
    raise AssertionError("no stack depth runs the program")


@cocotb.test()
async def test_random_structured(dut):
    """Seeded random structured programs match the model; the set diverges, nests and loops per lane."""
    p = params(dut)
    depths, divergent = [], 0
    for seed in range(4 * SEEDS):
        if len(depths) == SEEDS:
            break
        rng = random.Random(1000 + seed)
        threads = rng.randint(p["tpb"], MAX_THREADS)
        data = [rng.randrange(4) for _ in range(MAX_THREADS)]
        prog = Structured(seed).program()
        d = depth_needed(prog, data, threads, p["tpb"])
        if d > 8:
            logger.info(f"seed {seed}: needs stack depth {d}, skipped")
            continue
        depths.append(d)
        divergent += bool(run(prog, data, threads, threads_per_block=p["tpb"]).divergent)
        await check(dut, prog, data, threads, f"seed {seed} ({len(prog)} instructions, {threads} threads)")
    logger.info(f"structured {p}: {len(depths)} programs, stack depths {depths}, {divergent} divergent")
    assert len(depths) == SEEDS and divergent >= len(depths) // 2 and max(depths) >= 4, (depths, divergent)
