"""ISA table: opcode map, operand formats and 16-bit encoding (docs/isa.md)."""

# mnemonic: (opcode, operand format)
OPCODES = {
    "NOP":   (0b0000, ""),
    "BR":    (0b0001, "nzp,imm"),
    "CMP":   (0b0010, "rs,rt"),
    "ADD":   (0b0011, "rd,rs,rt"),
    "SUB":   (0b0100, "rd,rs,rt"),
    "MUL":   (0b0101, "rd,rs,rt"),
    "DIV":   (0b0110, "rd,rs,rt"),
    "LDR":   (0b0111, "rd,rs"),
    "STR":   (0b1000, "rs,rt"),
    "CONST": (0b1001, "rd,imm"),
    "LDS":   (0b1010, "rd,rs"),
    "STS":   (0b1011, "rs,rt"),
    "BAR":   (0b1100, ""),
    "SRA":   (0b1101, "rd,rs,rt"),
    "MAX":   (0b1110, "rd,rs,rt"),
    "RET":   (0b1111, ""),
}
BY_OPCODE = {op: m for m, (op, _) in OPCODES.items()}

REG_ALIASES = {"%blockIdx": 13, "%blockDim": 14, "%threadIdx": 15}
READ_ONLY_REGS = frozenset(REG_ALIASES.values())

# field: (shift, width)
FIELDS = {"op": (12, 4), "rd": (8, 4), "rs": (4, 4), "rt": (0, 4), "nzp": (9, 3), "imm": (0, 8)}


def operands(mnemonic):
    fmt = OPCODES[mnemonic][1]
    return fmt.split(",") if fmt else []


def encode(mnemonic, **fields):
    op, _ = OPCODES[mnemonic]
    names = operands(mnemonic)
    if set(fields) - set(names):
        raise ValueError(f"{mnemonic}: unexpected fields {sorted(set(fields) - set(names))}")
    word = op << 12
    for name in names:
        shift, width = FIELDS[name]
        v = fields.get(name, 0)
        if not 0 <= v < (1 << width):
            raise ValueError(f"{mnemonic}: {name}={v} out of range [0, {(1 << width) - 1}]")
        word |= v << shift
    return word


def decode(word):
    """Inverse of encode for canonical words: (mnemonic, {field: value})."""
    mnemonic = BY_OPCODE[word >> 12]
    return mnemonic, {n: (word >> FIELDS[n][0]) & ((1 << FIELDS[n][1]) - 1) for n in operands(mnemonic)}
