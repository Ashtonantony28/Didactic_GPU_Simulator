import re
from pathlib import Path

import pytest

from xf.isa import FIELDS, OPCODES, decode, encode, operands

# Opcodes defined in xf/isa.py but not yet in docs/isa.md (SG2 wires and documents them).
PENDING_DOCS = {"SRA", "MAX"}


def test_opcodes_unique_and_cover_4_bits():
    assert sorted(op for op, _ in OPCODES.values()) == list(range(16))


def test_decode_inverts_encode_exactly_on_canonical_words():
    for w in range(1 << 16):
        mnem, fields = decode(w)
        used = 0xF000 | sum(((1 << FIELDS[f][1]) - 1) << FIELDS[f][0] for f in operands(mnem))
        assert (encode(mnem, **fields) == w) == (w & ~used == 0), f"{w:016b}"


def test_br_layout():
    assert encode("BR", nzp=0b100, imm=12) == 0b0001_100_0_00001100


@pytest.mark.parametrize("mnem,fields", [("ADD", {"rd": 16}), ("CONST", {"imm": 256}),
                                         ("BR", {"nzp": 8}), ("RET", {"rd": 1}), ("CMP", {"rd": 1})])
def test_encode_rejects_bad_fields(mnem, fields):
    with pytest.raises(ValueError):
        encode(mnem, **fields)


def test_docs_opcode_table_matches():
    doc = (Path(__file__).resolve().parents[2] / "docs" / "isa.md").read_text()
    rows = {m.replace("BRnzp", "BR"): int(op, 2) for op, m in re.findall(r"^\| `([01]{4})`\s*\| (\w+)", doc, re.M)}
    assert rows == {m: op for m, (op, _) in OPCODES.items() if m not in PENDING_DOCS}
