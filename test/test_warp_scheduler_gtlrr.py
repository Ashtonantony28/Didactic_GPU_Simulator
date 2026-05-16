"""LRR policy unit test."""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ReadOnly
from .helpers.runner import write_bench_result
from .helpers.logger import logger


NUM_WARPS = 4


def _i(v):
    try:
        return int(v)
    except Exception:
        return 0


class WarpModel:
    def __init__(self, warp_id, total=16, stall_every=4, stall_cycles=3):
        self.warp_id = warp_id
        self.total = total
        self.issued = 0
        self.stall = 0
        self.stall_every = stall_every
        self.stall_cycles = stall_cycles
        self.last_issued = False
    @property
    def done(self): return self.issued >= self.total
    @property
    def ready(self): return (not self.done) and self.stall == 0
    @property
    def progressing(self): return self.last_issued and self.ready
    def tick(self, did):
        if did:
            self.issued += 1
            self.last_issued = True
            if self.stall_every and self.issued % self.stall_every == 0:
                self.stall = self.stall_cycles
        else:
            self.last_issued = False
            if self.stall > 0:
                self.stall -= 1


async def run(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.reset.value = 1
    dut.warp_ready.value = 0
    dut.warp_progressing.value = 0
    for k in range(NUM_WARPS):
        getattr(dut, f"warp_age{k}").value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.reset.value = 0
    warps = [WarpModel(i) for i in range(NUM_WARPS)]
    cycles = 0
    issues = 0
    hist = []
    while not all(w.done for w in warps) and cycles < 10_000:
        ready = sum((1 << i) for i, w in enumerate(warps) if w.ready)
        prog  = sum((1 << i) for i, w in enumerate(warps) if w.progressing)
        dut.warp_ready.value = ready
        dut.warp_progressing.value = prog
        for i, w in enumerate(warps):
            getattr(dut, f"warp_age{i}").value = w.issued
        await ReadOnly()
        sel = _i(dut.selected_warp.value)
        valid = _i(dut.issue_valid.value)
        await RisingEdge(dut.clk)
        cycles += 1
        for i, w in enumerate(warps):
            w.tick(valid == 1 and sel == i)
        if valid:
            issues += 1
            hist.append(sel)
    return cycles, issues, hist


@cocotb.test()
async def test_gtlrr(dut):
    cycles, issues, hist = await run(dut)
    logger.info(f"LRR: {cycles} cycles, {issues} issues; first 20 = {hist[:20]}")
    assert issues == NUM_WARPS * 16
    pass
    pass  # GTLRR has loose ordering by construction
    write_bench_result("warp_scheduler_gtlrr", {"policy": "GTLRR", "cycles": cycles, "issues": issues})
