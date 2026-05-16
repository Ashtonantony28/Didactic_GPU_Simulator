# Phase 0 — Bootstrap

## Status
Done.

## What landed
- Cloned `adam-maj/tiny-gpu` into `baseline/`.
- Toolchain verified: `iverilog 12.0`, `sv2v v0.0.13`, `cocotb 2.0.1`,
  Python 3.11.
- Baseline compiles via `make compile` (sv2v + iverilog).
- Baseline `make test_matadd` reaches simulation start under cocotb 2.x;
  upstream's `format_cycle` helper hits a cocotb 2.0 API mismatch
  (LogicArray no longer supports arithmetic). The toolchain is verified;
  the cocotb 2.0 compatibility lives in our `src/` test harness.

## Architecture summary
See [docs/architecture.md](../architecture.md) for the full first-principles
write-up of upstream's design.

## Reproducibility
```
apt-get install -y iverilog
pip3 install cocotb
curl -L -o sv2v.zip https://github.com/zachjs/sv2v/releases/download/v0.0.13/sv2v-Linux.zip
unzip sv2v.zip && cp sv2v-Linux/sv2v /usr/local/bin/

cd baseline
make compile
# baseline tests use upstream cocotb 1.x conventions; see Phase 1 for
# our cocotb 2.x compatible test harness.
```
