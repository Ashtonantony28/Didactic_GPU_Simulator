"""run_launches: transpose then matadd on one memory equals running them separately."""
import cocotb

from xf.image import Image
from .helpers.runner import Launch, run_kernel, run_launches
from .test_matadd import PROGRAM as MATADD
from .test_transpose import PROGRAM as TRANSPOSE, INPUT

# transpose: x (2x4) at 0 -> xt (4x2) at 8; matadd: y[i] = mem[i] + mem[8+i] at 16.
EXPECTED = [a + b for a, b in zip(INPUT, [INPUT[r * 4 + c] for c in range(4) for r in range(2)])]
_separate = {}


def _image():
    img = Image(word_bits=8, addr_bits=8)
    img.alloc("x", [2, 4], "i8", data=[INPUT[:4], INPUT[4:]])
    img.alloc("xt", [4, 2], "i8")
    img.alloc("y", [8], "i8")
    return img


@cocotb.test()
async def test_separate_transpose(dut):
    _separate["m1"], mem = await run_kernel(dut, TRANSPOSE, _image().words, 8)
    _separate["after1"] = list(mem.memory)


@cocotb.test()
async def test_separate_matadd(dut):
    _separate["m2"], mem = await run_kernel(dut, MATADD, _separate["after1"], 8)
    _separate["after2"] = list(mem.memory)
    assert mem.memory[16:24] == EXPECTED


@cocotb.test()
async def test_chained(dut):
    img = _image()
    metrics, mem = await run_launches(dut, img, [Launch(TRANSPOSE, 8, "transpose"), Launch(MATADD, 8, "matadd")])
    assert mem.memory == _separate["after2"]
    assert img.read("y", mem.memory).tolist() == EXPECTED
    for got, want in zip(metrics, (_separate["m1"], _separate["m2"])):
        assert {k: got[k] for k in want} == want, f"{got['name']}: {got} != {want}"


@cocotb.test()
async def test_width_mismatch(dut):
    try:
        await run_launches(dut, Image(word_bits=32, addr_bits=12), [Launch(MATADD, 8)])
    except ValueError:
        return
    assert False, "32-bit image accepted by the 8-bit build"
