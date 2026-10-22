"""Multi-warp core (SG20, ADR-012, ADR-014): BAR, warp_scheduler policies inside the core, exact counters.

Runs at any WARPS_PER_CORE / WARP_POLICY; `make test_multiwarp` builds WARPS_PER_CORE=4 under each policy and
WARPS_PER_CORE=2. Every kernel is also checked against xf.isasim at the DUT's WARPS_PER_CORE.
"""
import cocotb
from cocotb.triggers import ReadOnly, RisingEdge

from xf.asm import assemble
from xf.isasim import run
from .helpers.logger import logger
from .helpers.memory import Memory
from .helpers.runner import MEM_LATENCY, _run, data_bus
from .helpers.setup import setup

IDLE, FETCH, WAIT, UPDATE = 0, 1, 4, 6
OP_BAR, OP_RET = 0b1100, 0b1111

# Thread t waits 3*t loop iterations, writes SMEM[t] = t + 1, then (after BAR) stores SMEM[3 - t] to out[i].
# Without a working BAR the early warps read slots the late warps have not written yet.
BARRIER = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #1
    CONST R2, #3
    MUL R3, %threadIdx, R2
    CONST R4, #0
loop:
    CMP R4, R3
    BRz stage
    ADD R4, R4, R1
    BRnzp loop
stage:
    ADD R5, %threadIdx, R1
    STS %threadIdx, R5
    BAR
    SUB R6, R2, %threadIdx
    LDS R7, R6
    CONST R8, #16
    ADD R8, R8, R0
    STR R8, R7
    RET
"""
# ORDER, AGES and LONG load a[i] and store to n + i (n = threads), so blocks on different cores never touch
# each other's words (the model runs blocks one after another).
# A straight ALU prefix (no stalls), one load, then more ALU: shows each policy's issue order.
ORDER = """
    CONST R1, #1
    ADD R2, R1, R1
    ADD R3, R2, R1
    ADD R4, R3, R1
    ADD R5, R4, R1
    ADD R6, R5, R1
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    LDR R7, R0
    ADD R7, R7, R6
    CONST R8, #{n}
    ADD R8, R8, R0
    STR R8, R7
    RET
"""
# Thread t spins blockDim - t times before two loads, so warps arrive with different ages.
AGES = """
    CONST R1, #1
    CONST R3, #0
    SUB R2, %blockDim, %threadIdx
spin:
    SUB R2, R2, R1
    CMP R2, R3
    BRn spin
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    LDR R4, R0
    ADD R4, R4, R1
    CONST R5, #{n}
    ADD R5, R5, R0
    STR R5, R4
    LDR R6, R5
    ADD R6, R6, R4
    STR R0, R6
    RET
"""
# 100 loop iterations (300 issues) per warp before a load: ages saturate (ADR-014).
LONG = """
    CONST R1, #1
    CONST R2, #100
    CONST R3, #0
spin:
    ADD R3, R3, R1
    CMP R3, R2
    BRn spin
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    LDR R4, R0
    ADD R4, R4, R3
    CONST R5, #{n}
    ADD R5, R5, R0
    STR R5, R4
    RET
"""
# Branch on threadIdx / (blockDim / 2): uniform within a warp at WARPS_PER_CORE >= 2, divergent at 1.
WARP_BRANCH = """
    MUL R0, %blockIdx, %blockDim
    ADD R0, R0, %threadIdx
    CONST R1, #2
    DIV R2, %blockDim, R1
    DIV R2, %threadIdx, R2
    CONST R3, #0
    CONST R4, #10
    CMP R2, R3
    BRz low
    CONST R4, #20
low:
    ADD R4, R4, %threadIdx
    STR R0, R4
    RET
"""


def params(dut):
    return int(dut.WARPS_PER_CORE.value), int(dut.WARP_POLICY.value), int(dut.NUM_CORES.value)


def model(dut, program, data, threads, policy="lrr"):
    bits, addr = data_bus(dut)
    return run(program, data, threads, word_bits=bits, addr_bits=addr,
               threads_per_block=int(dut.THREADS_PER_BLOCK.value), warps_per_core=params(dut)[0], warp_policy=policy)


class Trace:
    """Per-core issue slots, completed instructions and barrier stalls, from core_state and the instruction."""

    def __init__(self, dut):
        warps, _, cores = params(dut)
        self.warps, self.lanes = warps, int(dut.THREADS_PER_BLOCK.value) // warps
        self.cores = [dut.cores[c].core_instance for c in range(cores)]
        self.sched = [c.scheduler_instance for c in self.cores]
        self.issues = [[] for _ in self.cores]     # [warp, ready mask at the slot, yielded in WAIT]
        self.updates = 0
        self.bar_stall = 0
        self._state = [IDLE] * len(self.cores)
        self._arrived = [{} for _ in self.cores]   # warp -> cycle its BAR completed
        self._done = [set() for _ in self.cores]
        self._ready = [0] * len(self.cores)
        self.cycle = 0

    def sample(self):
        for c, (core, s) in enumerate(zip(self.cores, self.sched)):
            state = int(s.core_state.value)
            if state == FETCH and self._state[c] in (IDLE, UPDATE):
                self.issues[c].append([int(s.current_warp.value), self._ready[c], False])
            if state == IDLE and self._state[c] == WAIT:
                self.issues[c][-1][2] = True
            if self.warps > 1 and state == IDLE:
                self._ready[c] = int(s.warp_ready.value)
            if state == UPDATE:
                self.updates += 1
                op, w = int(core.instruction.value) >> 12, int(s.current_warp.value)
                if op == OP_BAR:
                    self._arrived[c][w] = self.cycle + 1
                if op == OP_RET:
                    self._done[c].add(w)
            if state == IDLE and self._state[c] != IDLE and int(s.done.value) == 0:
                live = {w for w in range(self.warps) if w * self.lanes < int(core.thread_count.value)} - self._done[c]
                if live and live <= set(self._arrived[c]):
                    self.bar_stall += sum(self.cycle - a for w, a in self._arrived[c].items())
                    self._arrived[c] = {}
            if int(core.reset.value):
                self._arrived[c], self._done[c] = {}, set()
            self._state[c] = state
        self.cycle += 1


async def run_traced(dut, program, data, threads):
    bits, addr = data_bus(dut)
    pm = Memory(dut, 8, 16, 1, "program", latency=MEM_LATENCY)
    dm = Memory(dut, addr, bits, len(dut.data_mem_read_ready), "data", latency=MEM_LATENCY)
    tr = Trace(dut)

    async def watch():
        while True:
            await ReadOnly()
            tr.sample()
            await RisingEdge(dut.clk)

    await setup(dut, pm, program, dm, data, threads)
    task = cocotb.start_soon(watch())
    metrics = await _run(dut, pm, dm, threads, 50_000)
    task.cancel()
    return metrics, dm, tr


async def check_against_model(dut, src, data, threads, policies=("lrr", "gto")):
    prog = assemble(src)
    want = [model(dut, prog, data, threads, p) for p in policies]
    assert all((r.memory, r.steps) == (want[0].memory, want[0].steps) for r in want), "model depends on interleaving"
    _, dm, tr = await run_traced(dut, prog, data, threads)
    assert dm.memory == want[0].memory, f"RTL {dm.memory[:32]} != model {want[0].memory[:32]}"
    issues = int(dut.warp_issues_total.value)
    assert issues == want[0].steps == tr.updates, f"warp_issues_total {issues}, model {want[0].steps}, UPDATEs {tr.updates}"
    stall = int(dut.bar_stall_cycles_total.value)
    logger.info(f"multiwarp {params(dut)}: {issues} issues, {stall} bar stall cycles, issue order {tr.issues}")
    assert stall == tr.bar_stall, f"bar_stall_cycles_total {stall} != traced {tr.bar_stall}"
    return want[0], tr


@cocotb.test()
async def test_barrier(dut):
    """BAR orders SMEM between warps: out[i] = 4 - threadIdx; counters exact; stalls only with several warps."""
    threads = 8
    r, tr = await check_against_model(dut, BARRIER, [], threads)
    assert r.memory[16:16 + threads] == [4 - t % 4 for t in range(threads)], r.memory[16:16 + threads]
    stall = int(dut.bar_stall_cycles_total.value)
    assert (stall > 0) == (params(dut)[0] > 1), f"bar_stall_cycles_total {stall}"


def reference_order(tr, warps, policy):
    """Replay warp_scheduler.sv on the core's ready mask at each issue slot; assert it picks what the core issued.
    Returns the number of slots where GTO and GTLRR would pick differently."""
    differ = 0
    for c, slots in enumerate(tr.issues):
        ptr = cur = 0
        age, pending = [0] * warps, set()
        for i, (w, ready, yielded) in enumerate(slots):
            rdy = [ready >> k & 1 for k in range(warps)]
            rot = [(ptr + k) % warps for k in range(warps)]
            picks = (next(k for k in rot if rdy[k]),
                     cur if rdy[cur] else min((k for k in range(warps) if rdy[k]), key=lambda k: age[k]),
                     cur if rdy[cur] else next(k for k in rot if rdy[k]))
            assert w == picks[policy], (f"core {c} slot {i}: issued warp {w}, policy {policy} reference picks "
                                        f"{picks[policy]} (ready {rdy}, ages {age})")
            differ += picks[1] != picks[2]
            ptr, cur = (w + 1) % warps, w
            if w not in pending:
                age[w] = min(age[w] + 1, 254)
            pending.discard(w)
            if yielded:
                pending.add(w)
        assert not pending, f"core {c}: warps {pending} yielded and never resumed"
    return differ


@cocotb.test()
async def test_policy_order(dut):
    """Issue order inside the core is what test_warp_scheduler_* assert of each policy (LRR balanced, GTO sticks),
    and the reference of warp_scheduler.sv reproduces it slot by slot."""
    warps, policy, cores = params(dut)
    threads = cores * int(dut.THREADS_PER_BLOCK.value)
    _, tr = await check_against_model(dut, ORDER.format(n=threads), list(range(threads)), threads)
    for c, slots in enumerate(tr.issues):
        hist = [w for w, _, _ in slots]
        assert len(hist) >= 14 * warps, f"core {c}: {len(hist)} issue slots"
        if warps == 1:
            assert set(hist) == {0}
        elif policy == 0:
            # The ALU prefix is 6 stall-free instructions per warp.
            assert hist[:2 * warps] == list(range(warps)) * 2, f"core {c}: LRR order {hist[:2 * warps]}"
        else:
            assert hist[:6] == [0] * 6, f"core {c}: policy {policy} did not stick: {hist[:6]}"
    if warps > 1:
        reference_order(tr, warps, policy)


@cocotb.test()
async def test_policy_ages(dut):
    """Warps reach their loads after different instruction counts, so GTO (oldest = fewest issued) and GTLRR
    (round-robin) choose differently at some slot (with more than two warps); the core still matches the reference for its policy."""
    warps, policy, cores = params(dut)
    if warps == 1:
        return
    threads = cores * int(dut.THREADS_PER_BLOCK.value)
    _, tr = await check_against_model(dut, AGES.format(n=threads), list(range(threads)), threads)
    differ = reference_order(tr, warps, policy)
    assert differ or warps == 2, "GTO and GTLRR agree on every slot: the trace does not tell them apart"


@cocotb.test()
async def test_age_saturation(dut):
    """More than 255 issues per warp before every warp stalls on a load: the core still issues (GTO compares
    ages with <, so an age of 255 would never be picked) and follows the reference."""
    warps, policy, cores = params(dut)
    threads = cores * int(dut.THREADS_PER_BLOCK.value)
    _, tr = await check_against_model(dut, LONG.format(n=threads), list(range(threads)), threads)
    if warps > 1:
        reference_order(tr, warps, policy)


@cocotb.test()
async def test_warp_branch(dut):
    """A branch uniform within each warp but not within the block: each warp follows its own path."""
    warps, _, cores = params(dut)
    if warps == 1:
        return
    threads = cores * int(dut.THREADS_PER_BLOCK.value)
    r, _ = await check_against_model(dut, WARP_BRANCH, [], threads)
    assert not r.divergent, r.divergent
    half = int(dut.THREADS_PER_BLOCK.value) // 2
    assert r.memory[:threads] == [(10 if t % (2 * half) < half else 20) + t % (2 * half) for t in range(threads)]


@cocotb.test()
async def test_partial_block(dut):
    """6 threads: the second block has fewer threads than lanes x warps, so some warps have no threads."""
    await check_against_model(dut, BARRIER, [], 6)
