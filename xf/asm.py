"""Two-pass assembler for the tiny-gpu ISA: `assemble(text) -> list[int]`.

Syntax: one instruction per line, `;` comments, `label:` prefixes,
registers R0-R15 or %blockIdx/%blockDim/%threadIdx, immediates `#5`/`#0x80`,
branches `BR<n|z|p...> target` where target is a label or instruction index,
`SYNC target` sets the reconvergence PC of the next divergent branch (encoded as BR with nzp = 000).
"""
import re
import sys

from .isa import OPCODES, READ_ONLY_REGS, REG_ALIASES, encode, decode

_LABEL = re.compile(r"^([A-Za-z_]\w*):")
_BR = re.compile(r"^BR(N?)(Z?)(P?)$")


class AsmError(ValueError):
    pass


def _reg(tok):
    if tok in REG_ALIASES:
        return REG_ALIASES[tok]
    m = re.fullmatch(r"[Rr](\d+)", tok)
    if not m or int(m.group(1)) > 15:
        raise AsmError(f"bad register {tok!r}")
    return int(m.group(1))


def _imm(tok, labels=None):
    if labels is not None and tok in labels:
        return labels[tok]
    try:
        return int(tok[1:] if tok.startswith("#") else tok, 0)
    except ValueError:
        raise AsmError(f"bad immediate or unknown label {tok!r}") from None


def _lines(text):
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split(";", 1)[0].strip()
        labels = []
        while (m := _LABEL.match(line)):
            labels.append(m.group(1))
            line = line[m.end():].strip()
        yield n, labels, line


def assemble(text):
    labels, body = {}, []
    for n, names, line in _lines(text):
        for name in names:
            if name in labels:
                raise AsmError(f"line {n}: duplicate label {name!r}")
            labels[name] = len(body)
        if line:
            body.append((n, line))

    words = []
    for n, line in body:
        mnem, _, rest = line.replace("\t", " ").partition(" ")
        mnem = mnem.upper()
        args = [a.strip() for a in rest.split(",")] if rest.strip() else []
        try:
            if (m := _BR.match(mnem)) or mnem == "SYNC":
                if len(args) != 1:
                    raise AsmError(f"{mnem} takes one target")
                nzp = int("".join("1" if g else "0" for g in m.groups()), 2) if m else 0
                if m and not nzp:
                    raise AsmError("BR needs a condition (n, z, p); use SYNC for nzp = 000")
                words.append(encode("BR", nzp=nzp, imm=_imm(args[0], labels)))
                continue
            if mnem not in OPCODES or mnem == "BR":
                raise AsmError(f"unknown mnemonic {mnem!r}")
            names = OPCODES[mnem][1].split(",") if OPCODES[mnem][1] else []
            if len(args) != len(names):
                raise AsmError(f"{mnem} expects {len(names)} operands, got {len(args)}")
            fields = {f: (_imm(a) if f == "imm" else _reg(a)) for f, a in zip(names, args)}
            if fields.get("rd") in READ_ONLY_REGS:
                raise AsmError(f"R{fields['rd']} is read-only")
            words.append(encode(mnem, **fields))
        except (AsmError, ValueError) as e:
            raise AsmError(f"line {n}: {line!r}: {e}") from None
    return words


_ALIAS_NAMES = {v: k for k, v in REG_ALIASES.items()}


def disassemble(words):
    out = []
    for w in words:
        mnem, f = decode(w)
        if mnem == "BR" and not f["nzp"]:
            out.append(f"SYNC {f['imm']}")
            continue
        if mnem == "BR":
            out.append("BR" + "".join(c for c, b in zip("nzp", f"{f['nzp']:03b}") if b == "1") + f" {f['imm']}")
            continue
        args = [f"#{f[k]}" if k == "imm" else _ALIAS_NAMES.get(f[k], f"R{f[k]}") for k in f]
        out.append(f"{mnem} {', '.join(args)}".strip())
    return out


if __name__ == "__main__":
    for w in assemble(open(sys.argv[1]).read()):
        print(f"{w:016b}")
