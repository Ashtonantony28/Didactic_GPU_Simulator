"""Behavioral external-memory model for cocotb 2.x.

The DUT drives `<name>_mem_read_valid/address` per channel; we respond with
ready/data the same cycle. Optional latency lets us emulate slower memories.
"""
from typing import List, Optional
from cocotb.types import LogicArray
from .logger import logger


def _to_int(v) -> int:
    """cocotb 2.x: BinaryValue → LogicArray. Return as integer; X/Z treated as 0."""
    try:
        return int(v)
    except Exception:
        s = str(v).replace("x", "0").replace("z", "0")
        return int(s, 2) if s else 0


def _unpack(value, width: int, n: int) -> List[int]:
    """Decode either a packed bus or an unpacked array signal into n integers."""
    if isinstance(value, LogicArray) or hasattr(value, "binstr"):
        raw = _to_int(value)
        out = []
        for i in range(n):
            out.append((raw >> ((n - 1 - i) * width)) & ((1 << width) - 1))
        return out
    # Unpacked array handle
    try:
        return [_to_int(value[i]) for i in range(n)]
    except Exception:
        return [_to_int(value)] * n


class Memory:
    def __init__(self, dut, addr_bits: int, data_bits: int, channels: int,
                 name: str, latency: int = 0):
        self.dut = dut
        self.addr_bits = addr_bits
        self.data_bits = data_bits
        self.channels = channels
        self.name = name
        self.latency = latency
        self.memory = [0] * (2 ** addr_bits)

        self.mem_read_valid   = getattr(dut, f"{name}_mem_read_valid")
        self.mem_read_address = getattr(dut, f"{name}_mem_read_address")
        self.mem_read_ready   = getattr(dut, f"{name}_mem_read_ready")
        self.mem_read_data    = getattr(dut, f"{name}_mem_read_data")

        self.has_writes = (name != "program")
        if self.has_writes:
            self.mem_write_valid   = getattr(dut, f"{name}_mem_write_valid")
            self.mem_write_address = getattr(dut, f"{name}_mem_write_address")
            self.mem_write_data    = getattr(dut, f"{name}_mem_write_data")
            self.mem_write_ready   = getattr(dut, f"{name}_mem_write_ready")

        # Initialize outputs to 0 so the DUT sees deterministic ready signals.
        self.mem_read_ready.value = 0
        if self.has_writes:
            self.mem_write_ready.value = 0

        self._pending_read = [0] * channels   # countdown until ready
        self._pending_write = [0] * channels

    def _pack_data(self, vals: List[int]) -> int:
        packed = 0
        for v in vals:
            packed = (packed << self.data_bits) | (v & ((1 << self.data_bits) - 1))
        return packed

    def run(self) -> int:
        """One step. Returns number of external requests serviced this cycle."""
        serviced = 0
        read_valids   = _unpack(self.mem_read_valid.value,   1, self.channels)
        read_addrs    = _unpack(self.mem_read_address.value, self.addr_bits, self.channels)

        data_out  = [0] * self.channels
        ready_out = [0] * self.channels

        for i in range(self.channels):
            if read_valids[i] == 1:
                if self._pending_read[i] < self.latency:
                    self._pending_read[i] += 1
                else:
                    addr = read_addrs[i] & ((1 << self.addr_bits) - 1)
                    data_out[i]  = self.memory[addr]
                    ready_out[i] = 1
                    self._pending_read[i] = 0
                    serviced += 1
            else:
                self._pending_read[i] = 0

        self.mem_read_data.value  = self._pack_data(data_out)
        self.mem_read_ready.value = int("".join(str(r) for r in ready_out), 2) if self.channels else 0

        if self.has_writes:
            write_valids = _unpack(self.mem_write_valid.value,   1, self.channels)
            write_addrs  = _unpack(self.mem_write_address.value, self.addr_bits, self.channels)
            write_datas  = _unpack(self.mem_write_data.value,    self.data_bits, self.channels)
            wready_out = [0] * self.channels
            for i in range(self.channels):
                if write_valids[i] == 1:
                    if self._pending_write[i] < self.latency:
                        self._pending_write[i] += 1
                    else:
                        addr = write_addrs[i] & ((1 << self.addr_bits) - 1)
                        self.memory[addr] = write_datas[i] & ((1 << self.data_bits) - 1)
                        wready_out[i] = 1
                        self._pending_write[i] = 0
                        serviced += 1
                else:
                    self._pending_write[i] = 0
            self.mem_write_ready.value = int("".join(str(r) for r in wready_out), 2) if self.channels else 0
        return serviced

    def write(self, address: int, data: int):
        if 0 <= address < len(self.memory):
            self.memory[address] = data & ((1 << self.data_bits) - 1)

    def load(self, rows: List[int]):
        for addr, val in enumerate(rows):
            self.write(addr, val)

    def display(self, rows: int, decimal: bool = True):
        logger.info(f"\n{self.name.upper()} MEMORY (first {rows} rows)")
        for i in range(min(rows, len(self.memory))):
            if decimal:
                logger.info(f"  [{i:3d}] = {self.memory[i]}")
            else:
                logger.info(f"  [{i:3d}] = {self.memory[i]:0{self.data_bits}b}")
