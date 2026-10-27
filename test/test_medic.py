"""MeDiC on a mixed workload: hot lanes and streaming lanes in one launch (SG19, ADR-009).

Threads 0, 1 of each block stream: a new address every iteration, block 0's on its hot lanes' lines.
Threads 2, 3 re-read one hot address each; block 1's sit on other lines and move to a new address every
3 iterations, so they miss while block 0's hit. Every iteration every lane write-throughs its running sum. Data memory is slower
than program memory, so the two cores' requests overlap and the controller has to choose.
A monitor counts per-lane lookups, installs and bypasses at the L1-D's ports, and checks that a
bypassed fill leaves no line behind and that the controller never gives a channel to a write or a
low-utility read while a high-utility read waits.

`make test_medic` runs this module on the configuration at MEDIC=0 (writing the hot lanes' hit count
to $MEDIC_BASELINE), then at MEDIC=1 (comparing against it).
"""
import json
import os
import cocotb
from cocotb.triggers import RisingEdge, ReadOnly
from xf.asm import assemble
from xf.isasim import run
from .helpers.memory import Memory
from .helpers.runner import MEM_LATENCY, _run, _safe_int, data_bus, run_kernel
from .helpers.setup import setup
from .test_l1d_indepth import encode, ADD, MUL, LDR, STR, CONST, RET, BLOCKIDX, BLOCKDIM, THREADIDX

SUB, DIV = 0b0100, 0b0110
ITERS, MOVE, OUT, BLOCKS, TPB = 7, 3, 240, 2, 4
WARMUP = 2       # installs before a streaming lane turns low-utility: 4-lookup window, MEDIC_THRESHOLD = 2
DATA_LATENCY = MEM_LATENCY + 4


def program():
    p = [
        encode(CONST, rd=12, imm=2),
        encode(DIV,   rd=0, rs=THREADIDX, rt=12),   # R0 = 1 on hot lanes, 0 on streaming lanes
        encode(CONST, rd=7, imm=4),
        encode(MUL,   rd=7, rs=7, rt=BLOCKIDX),
        encode(MUL,   rd=7, rs=7, rt=0),            # R7 = hot-address move: 4 on block 1's hot lanes
        encode(SUB,   rd=2, rs=THREADIDX, rt=12),
        encode(ADD,   rd=2, rs=2, rt=7),            # R2 = hot address t - 2 + 4b
        encode(CONST, rd=11, imm=16),
        encode(MUL,   rd=1, rs=BLOCKIDX, rt=12),
        encode(ADD,   rd=1, rs=1, rt=11),
        encode(ADD,   rd=1, rs=1, rt=THREADIDX),    # R1 = streaming address 16 + 2b + t
        encode(SUB,   rd=5, rs=2, rt=1),
        encode(MUL,   rd=5, rs=5, rt=0),
        encode(ADD,   rd=5, rs=1, rt=5),            # R5 = hot ? R2 : R1
        encode(CONST, rd=3, imm=32),
        encode(MUL,   rd=6, rs=3, rt=0),
        encode(SUB,   rd=3, rs=3, rt=6),            # R3 = stride: 0 hot, 32 streaming (same line index)
        encode(CONST, rd=4, imm=0),
        encode(CONST, rd=10, imm=OUT),
        encode(MUL,   rd=9, rs=BLOCKIDX, rt=BLOCKDIM),
        encode(ADD,   rd=10, rs=10, rt=9),
        encode(ADD,   rd=10, rs=10, rt=THREADIDX),  # R10 = &out[4b + t]
    ]
    for i in range(ITERS):
        if i and i % MOVE == 0:
            p.append(encode(ADD, rd=5, rs=5, rt=7))
        p += [
            encode(LDR, rd=6, rs=5),
            encode(ADD, rd=4, rs=4, rt=6),
            encode(STR, rs=10, rt=4),
            encode(ADD, rd=5, rs=5, rt=3),
        ]
    return p + [encode(RET)]


def addresses(b, t):
    if t >= 2:
        return [t - 2 + 4 * b * (1 + i // MOVE) for i in range(ITERS)]
    return [16 + 2 * b + t + 32 * i for i in range(ITERS)]


async def monitor(dut, s):
    n = len(dut.lsu_read_ready)
    abits = len(dut.ctl_read_address) // n
    coalesce, medic, lines = int(dut.COALESCE.value), int(dut.MEDIC.value), int(dut.DCACHE_LINES.value)
    l1d = dut.with_l1d.l1d

    def sample():
        a = _safe_int(dut.ctl_read_address.value)
        return dict(rv=_safe_int(dut.ctl_read_valid.value), rr=_safe_int(dut.ctl_read_ready.value),
                    lr=_safe_int(dut.lsu_read_ready.value),
                    hi=_safe_int(dut.warp_high_util.value),
                    srv=_safe_int(dut.data_memory_controller.channel_serving_consumer.value),
                    addr=[(a >> (abits * i)) & ((1 << abits) - 1) for i in range(n)])

    await ReadOnly()
    prev = sample()
    while True:
        await RisingEdge(dut.clk)
        await ReadOnly()
        cur = sample()
        for i in range(n):
            if (cur["lr"] >> i) & 1 and not (prev["lr"] >> i) & 1:
                s["reads"][i] += 1
            if (prev["rv"] >> i) & (prev["rr"] >> i) & 1:
                bypass = medic and not (prev["hi"] >> i) & 1
                s["bypass" if bypass else "install"][i] += 1
                a = prev["addr"][i]
                if bypass and _safe_int(l1d.valid_arr[a % lines].value) and \
                        _safe_int(l1d.tag_arr[a % lines].value) == a // lines:
                    s["violations"].append((s["cycle"], f"lane {i} installed {a} while low-utility"))
        claimed = cur["srv"] & ~prev["srv"]
        waiting = prev["rv"] & ~prev["srv"]   # priority orders reads only (ADR-016)
        hi = prev["hi"]
        if claimed and waiting & ~claimed and waiting & hi and waiting & ~hi:
            s["decided"] += 1
        if coalesce:   # a rider shares another claimant's read transaction and takes no channel
            for i in range(n):
                if (claimed >> i) & (prev["rv"] >> i) & 1 and any(
                        (claimed >> j) & (prev["rv"] >> j) & 1 and prev["addr"][j] == prev["addr"][i]
                        for j in range(i)):
                    claimed &= ~(1 << i)
        if waiting & hi & ~claimed and claimed & ~(hi & prev["rv"]):
            s["violations"].append((s["cycle"], f"{waiting:b}", f"{claimed:b}", f"{hi:b}"))
        s["cycle"] += 1
        prev = cur


@cocotb.test()
async def test_medic_mixed(dut):
    """Streaming lanes bypass after warm-up; hot lanes hit at least as often as at MEDIC=0; priority holds."""
    lines, medic, tpb = int(dut.DCACHE_LINES.value), int(dut.MEDIC.value), int(dut.THREADS_PER_BLOCK.value)
    assert lines >= 16 and tpb == TPB, f"test_medic needs DCACHE_LINES >= 16, THREADS_PER_BLOCK = {TPB}"
    baseline = os.environ.get("MEDIC_BASELINE")
    assert baseline, "MEDIC_BASELINE is unset; run this module through `make test_medic`"
    n, channels = len(dut.lsu_read_ready), len(dut.data_mem_read_ready)
    bits, abits = data_bus(dut)
    mask = (1 << bits) - 1
    data = [(3 * a + 1) & mask for a in range(OUT)]
    s = dict(reads=[0] * n, install=[0] * n, bypass=[0] * n, decided=0, violations=[], cycle=0)
    pm = Memory(dut, 8, 16, 1, "program", latency=MEM_LATENCY)
    dm = Memory(dut, abits, bits, channels, "data", latency=DATA_LATENCY)
    await setup(dut, pm, program(), dm, data, TPB * BLOCKS)
    mon = cocotb.start_soon(monitor(dut, s))
    metrics = await _run(dut, pm, dm, TPB * BLOCKS, 20_000)
    mon.cancel()

    for b in range(BLOCKS):
        for t in range(TPB):
            want = sum(data[a] for a in addresses(b, t)) & mask
            got = dm.memory[OUT + TPB * b + t]
            assert got == want, f"block {b} thread {t}: {got}, expected {want}"
    misses = [s["install"][i] + s["bypass"][i] for i in range(n)]
    assert sum(misses) == metrics["dcache_misses"], f"{misses} vs {metrics}"
    assert sum(s["reads"]) - sum(misses) == metrics["dcache_hits"], f"{s['reads']} vs {metrics}"
    hot = [i for i in range(n) if i % TPB >= 2 and s["reads"][i]]
    stream = [i for i in range(n) if i % TPB < 2 and s["reads"][i]]
    for i in stream:
        assert misses[i] == s["reads"][i], f"streaming lane {i} hit: {s}"
    hot_reads = sum(s["reads"][i] for i in hot)
    hot_hits = hot_reads - sum(misses[i] for i in hot)
    assert sum(s["reads"][i] for i in stream) == hot_reads == ITERS * BLOCKS * 2
    report = dict(hot_hits=hot_hits, hot_reads=hot_reads, cycles=metrics["cycles"],
                  install=s["install"], bypass=s["bypass"], decided=s["decided"],
                  dcache_hits=metrics["dcache_hits"], dcache_misses=metrics["dcache_misses"],
                  data_mem_requests=metrics["data_mem_requests"])
    dut._log.info(f"MEDIC={medic} {report}")
    if os.environ.get("BENCH_OUT"):   # bench/run.py --phase 11
        with open(os.path.join(os.environ["BENCH_OUT"], f"medic{medic}.json"), "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

    if not medic:
        assert sum(s["bypass"]) == 0, f"bypass at MEDIC=0: {s['bypass']}"
        assert hot_hits < hot_reads, f"streaming lanes never evicted a hot line: {report}"
        with open(baseline, "w") as f:
            json.dump(report, f)
        return

    with open(baseline) as f:
        base = json.load(f)
    assert base["hot_reads"] == hot_reads, f"baseline {base} is for another workload"
    for i in stream:
        assert (s["install"][i], s["bypass"][i]) == (WARMUP, s["reads"][i] - WARMUP), \
            f"streaming lane {i}: {s['install'][i]} installs, {s['bypass'][i]} bypasses"
    assert hot_hits >= base["hot_hits"], f"hot lanes hit {hot_hits}/{hot_reads} at MEDIC=1, {base['hot_hits']} at MEDIC=0"
    assert not s["violations"], f"bypass or priority broken: {s['violations'][:4]}"
    if n > channels:
        assert s["decided"], "the controller never chose between high- and low-utility lanes"


# Odd lanes re-read one word (high utility), even lanes read a new word each time (low utility), then every lane stores
# to one word: writes skip the priority pass, so the highest lane still wins (ADR-016).
STR_AFTER_MIXED = """
    CONST R1, #1
    CONST R3, #2
    DIV R4, %threadIdx, R3
    MUL R4, R4, R3
    SUB R4, %threadIdx, R4
    SUB R4, R1, R4              ; 1 on even lanes
    CONST R5, #16
    MUL R6, R4, R5              ; stride: 16 even, 0 odd
    CONST R7, #64
    MUL R7, R4, R7
    ADD R7, R7, %threadIdx
""" + "    LDR R8, R7\n    ADD R7, R7, R6\n" * 5 + """
    CONST R0, #200
    ADD R2, %threadIdx, R1
    STR R0, R2
    RET"""


@cocotb.test()
async def test_medic_str_conflict(dut):
    """Lanes of one STR with mixed utility: the highest lane wins, as in xf.isasim."""
    tpb = int(dut.THREADS_PER_BLOCK.value)
    bits, abits = data_bus(dut)
    prog = assemble(STR_AFTER_MIXED)
    _, mem = await run_kernel(dut, prog, [], tpb)
    hi = _safe_int(dut.warp_high_util.value) & ((1 << tpb) - 1)
    assert hi == sum(1 << t for t in range(1, tpb, 2)), f"lane utility {hi:b}: odd lanes high, even lanes low expected"
    want = run(prog, [], tpb, word_bits=bits, addr_bits=abits, threads_per_block=tpb).memory[200]
    assert mem.memory[200] == want == tpb, (mem.memory[200], want)
