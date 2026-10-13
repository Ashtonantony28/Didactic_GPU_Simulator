"""Exit non-zero unless a cocotb results.xml exists, ran >=1 test, and has no failures.

vvp exits 0 even when cocotb fails a test or never loads, so make gates on this.
"""
import sys
import xml.etree.ElementTree as ET


def main(path: str) -> int:
    try:
        cases = ET.parse(path).getroot().findall(".//testcase")
    except (OSError, ET.ParseError) as e:
        print(f"FAIL: no cocotb results ({e})")
        return 1
    bad = [c.get("name") for c in cases if c.find("failure") is not None or c.find("error") is not None]
    ran = [c for c in cases if c.find("skipped") is None]
    if bad:
        print(f"FAIL: {len(bad)} failing test(s) in {path}: {', '.join(bad)}")
        return 1
    if not ran:
        print(f"FAIL: zero tests ran ({path})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
