"""nanoGPT-backed tests take `nanogpt_dir` or `blk`. `make test_all` sets XF_REQUIRE_NANOGPT=1, so a missing
torch or NANOGPT_DIR fails those tests instead of skipping them (AUDIT-7 H1)."""
import importlib.util
import os

import pytest

from xf import export


@pytest.fixture(scope="session")
def nanogpt_dir():
    missing = [n for n, ok in (("torch", importlib.util.find_spec("torch")), ("NANOGPT_DIR", os.environ.get("NANOGPT_DIR"))) if not ok]
    if missing:
        (pytest.fail if os.environ.get("XF_REQUIRE_NANOGPT") else pytest.skip)(f"{' and '.join(missing)} missing")
    return os.environ["NANOGPT_DIR"]


@pytest.fixture(scope="session")
def blk(nanogpt_dir):
    return export.build_block(nanogpt_dir)
