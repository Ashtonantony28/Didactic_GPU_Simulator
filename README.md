# tiny-gpu-extended

Replication of [adam-maj/tiny-gpu](https://github.com/adam-maj/tiny-gpu)
extended with bench harness, parameterised instruction cache, memory
coalescing, divergent-kernel verification, and design specs for shared
memory, a pluggable warp scheduler, and MeDiC memory-divergence
correction.

See [`docs/final-report.md`](docs/final-report.md) for results,
[`docs/architecture.md`](docs/architecture.md) for the design, and
[`ROADMAP.md`](ROADMAP.md) for per-phase status.

## Quick start

```bash
apt-get install -y iverilog
pip3 install cocotb
# sv2v 0.0.13 binary into /usr/local/bin/

make test_matadd
make test_matmul
python3 bench/run.py --phase 5 --label phase-5-coalesce \
    --params "ICACHE_LINES=16 COALESCE=1"
```
