"""Instruction-level reference model of the GPU (docs/isa.md, ADR-011, ADR-012), bit for bit with the RTL.

    run(program, data, threads, word_bits=8, addr_bits=8, ...) -> Result

Each instruction is atomic across a warp: every active lane reads its operands before any lane writes
registers, NZP, memory or SMEM. Within one STR lanes store in lane order (the last lane wins); within one
STS the highest lane wins. Blocks run one after another in blockIdx order on one memory, so races
between blocks (a block reading another block's writes) are out of scope.

simt=False is today's RTL: the warp follows its lowest enabled lane and `divergent` records branches
where enabled lanes disagree; SYNC (BR with nzp = 000) is a no-op. simt=True is the ADR-011 SIMT stack.
warps_per_core > 1 splits a block into warps of consecutive lanes that share SMEM and issue one
instruction at a time: `warp_policy` "lrr" (round-robin, ADR-012) or "gto" (stay on a warp until it
waits at BAR or finishes, then the lowest ready warp).
"""
from dataclasses import dataclass, field

PC_MASK = 0xFF


class IsaSimError(RuntimeError):
    pass


@dataclass
class Result:
    memory: list
    steps: int = 0
    divergent: list = field(default_factory=list)        # (block, pc), simt=False only
    stores_conflict: list = field(default_factory=list)  # (block, pc, addr)


class _Warp:
    def __init__(self, mask):
        self.stack = [[0, None, mask]]   # entries [pc, rpc, lane mask]; the top runs
        self.sync = None
        self.bar = False
        self.done = not mask


def run(program, data, threads, *, word_bits=8, addr_bits=8, threads_per_block=4, warps_per_core=1,
        smem_size=64, simt=False, stack_depth=8, max_steps=1_000_000, warp_policy="lrr"):
    wmask, amask, sign = (1 << word_bits) - 1, (1 << addr_bits) - 1, 1 << (word_bits - 1)
    if smem_size < 1 or smem_size & (smem_size - 1):
        raise ValueError(f"smem_size {smem_size} is not a power of two")
    if threads_per_block % warps_per_core:
        raise ValueError(f"{warps_per_core} warps do not divide {threads_per_block} threads")
    if len(data) > 1 << addr_bits or len(program) > 1 << 8:
        raise ValueError("data or program larger than its memory")
    if warp_policy not in ("lrr", "gto"):
        raise ValueError(f"unknown warp policy {warp_policy!r}")

    def signed(x):
        return x - ((x & sign) << 1)

    alu = {
        0b0011: lambda a, b: (a + b) & wmask,
        0b0100: lambda a, b: (a - b) & wmask,
        0b0101: lambda a, b: (a * b) & wmask,
        0b0110: lambda a, b: a // b if b else 0,
        0b1101: lambda a, b: (signed(a) >> b) & wmask,
        0b1110: lambda a, b: a if signed(a) > signed(b) else b,
    }
    prog = [int(w) for w in program] + [0] * ((1 << 8) - len(program))
    res = Result([int(v) & wmask for v in data] + [0] * ((1 << addr_bits) - len(data)))
    mem, tpb, lanes_per_warp = res.memory, threads_per_block, threads_per_block // warps_per_core

    for b in range(-(-threads // tpb)):
        n = min(tpb, threads - b * tpb)
        regs = [[0] * 13 + [b & wmask, tpb & wmask, t & wmask] for t in range(tpb)]
        nzp, smem = [0] * tpb, [0] * smem_size
        warps = [_Warp(sum(1 << t for t in range(w * lanes_per_warp, min((w + 1) * lanes_per_warp, n))))
                 for w in range(warps_per_core)]
        cur = 0
        while True:
            live = [w for w in warps if not w.done]
            if not live:
                break
            if all(w.bar for w in live):
                for w in live:
                    w.bar = False
            order = range(cur, cur + warps_per_core) if warp_policy == "lrr" else [cur, *range(warps_per_core)]
            cur = next(i % warps_per_core for i in order if not warps[i % warps_per_core].done
                       and not warps[i % warps_per_core].bar)
            warp = warps[cur]
            if warp_policy == "lrr":
                cur = (cur + 1) % warps_per_core
            res.steps += 1
            if res.steps > max_steps:
                raise IsaSimError(f"more than {max_steps} steps")

            stack = warp.stack
            top = stack[-1]
            pc, mask = top[0], top[2]
            lanes = [t for t in range(tpb) if mask >> t & 1]
            word = prog[pc]
            op, rd, rs, rt, imm = word >> 12, (word >> 8) & 15, (word >> 4) & 15, word & 15, word & 0xFF
            top[0] = (pc + 1) & PC_MASK

            if op in alu:
                f = alu[op]
                vals = [f(regs[t][rs], regs[t][rt]) for t in lanes]
                if rd < 13:
                    for t, v in zip(lanes, vals):
                        regs[t][rd] = v
            elif op in (0b0111, 0b1001, 0b1010):   # LDR, CONST, LDS
                if op == 0b0111:
                    vals = [mem[regs[t][rs] & amask] for t in lanes]
                elif op == 0b1001:
                    vals = [imm] * len(lanes)
                else:
                    vals = [smem[regs[t][rs] & 0xFF & (smem_size - 1)] for t in lanes]
                if rd < 13:
                    for t, v in zip(lanes, vals):
                        regs[t][rd] = v
            elif op == 0b1000:                     # STR
                seen, clash = {}, set()
                for a, v in [(regs[t][rs] & amask, regs[t][rt]) for t in lanes]:
                    if seen.get(a, v) != v and a not in clash:
                        clash.add(a)
                        res.stores_conflict.append((b, pc, a))
                    seen[a] = mem[a] = v
            elif op == 0b1011:                     # STS
                for a, v in [(regs[t][rs] & 0xFF & (smem_size - 1), regs[t][rt]) for t in lanes]:
                    smem[a] = v
            elif op == 0b0010:                     # CMP
                for t in lanes:
                    nzp[t] = 0b100 if regs[t][rs] != regs[t][rt] else 0b010
            elif op == 0b1100:                     # BAR
                warp.bar = True
            elif op == 0b1111:                     # RET
                for e in stack:
                    e[2] &= ~mask
                stack[:] = [e for e in stack if e[2]]
                if not stack:
                    warp.done, warp.bar = True, False
                    continue
            elif op == 0b0001:
                cond = (word >> 9) & 7
                if cond == 0:                      # SYNC
                    if simt:
                        warp.sync = imm
                else:
                    taken = sum(1 << t for t in lanes if nzp[t] & cond)
                    if not simt:
                        if taken and taken != mask:
                            res.divergent.append((b, pc))
                        if taken & (mask & -mask):
                            top[0] = imm
                    elif taken == mask:
                        top[0] = imm
                    elif taken:
                        if warp.sync is None:
                            raise IsaSimError(f"block {b}: divergent branch at pc {pc} without SYNC")
                        r, warp.sync = warp.sync, None
                        top[0] = r
                        stack += [[(pc + 1) & PC_MASK, r, mask & ~taken], [imm, r, taken]]
                        if len(stack) - 1 > stack_depth:
                            raise IsaSimError(f"block {b}: SIMT stack deeper than {stack_depth} at pc {pc}")
            while len(stack) > 1 and stack[-1][0] == stack[-1][1]:
                stack.pop()
    return res
