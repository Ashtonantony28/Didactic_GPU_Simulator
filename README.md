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
pip3 install cocotb==2.0.1   # 2.1.x breaks the Makefile
# sv2v 0.0.13 binary into /usr/local/bin/

make test_matadd
make test_matmul
make regression                # exits non-zero on any failure
make test_all                  # every test module at default params
make -j4 matrix                # kernel and xf tests under 10 configurations
python3 bench/run.py --phase 5 --label phase-5-coalesce \
    --params "ICACHE_LINES=16 COALESCE=1"
```
