import re
from pathlib import Path

import pytest

from xf.asm import AsmError, assemble, disassemble
from xf.tests.legacy import ROOT, hand_encoded


def _sections():
    text = (Path(__file__).parent / "legacy.asm").read_text()
    parts = re.split(r"^;; == (\S+) ==$", text, flags=re.M)
    return dict(zip(parts[1::2], parts[2::2]))


LEGACY = hand_encoded()
SECTIONS = _sections()


def test_every_hand_encoded_program_has_a_source():
    assert len(LEGACY) >= 20
    assert set(SECTIONS) == set(LEGACY)


@pytest.mark.parametrize("key", sorted(LEGACY))
def test_reproduces_hand_encoded(key):
    assert assemble(SECTIONS[key]) == LEGACY[key]


@pytest.mark.parametrize("path", sorted((ROOT / "kernels").glob("*.asm")), ids=lambda p: p.name)
def test_kernels_assemble(path):
    assert assemble(path.read_text())


def test_vector_dot_kernel_matches_test():
    assert assemble((ROOT / "kernels" / "vector_dot.asm").read_text()) == LEGACY["test_vector_dot.module"]


def test_disassemble_round_trip():
    for words in LEGACY.values():
        assert assemble("\n".join(disassemble(words))) == words


def test_labels_forward_and_backward():
    src = """
    top:  CONST R0, #1     ; 0
          BRz end          ; 1
    mid:  BRnzp top        ; 2
    end:
          RET              ; 3
    """
    assert assemble(src) == [0x9001, 0b0001_010_0_00000011, 0b0001_111_0_00000000, 0xF000]


def test_syntax_variants():
    assert assemble("const r1, #0x80\nadd\tR2, R1, %threadIdx\nbrnp 0\nnop\nbar") == [
        0x9180, 0x321F, 0b0001_101_0_00000000, 0x0000, 0xC000]
    assert assemble("CONST R1, 255") == [0x91FF]
    assert assemble("SRA R1, R2, R3\nMAX R4, R5, R6") == [0xD123, 0xE456]


@pytest.mark.parametrize("src", [
    "FOO R1", "ADD R1, R2", "ADD R1, R2, R16", "CONST R1, #256", "CONST R1, #-1",
    "CONST %threadIdx, #1", "LDR R13, R0", "BRn nowhere", "a: NOP\na: NOP", "RET R1", "BRq 0",
    "BRn 256",
])
def test_errors(src):
    with pytest.raises(AsmError):
        assemble(src)


def test_error_names_line():
    with pytest.raises(AsmError, match="line 2"):
        assemble("NOP\nADD R1")
