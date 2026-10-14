"""AUDIT-4 repros: expected failures until the finding is fixed (strict, so a fix flips them)."""
import pytest

from xf.asm import AsmError, assemble


@pytest.mark.xfail(strict=True, reason="AUDIT-4 L3: `BR` with no condition assembles to a never-taken branch")
def test_br_without_condition_rejected():
    with pytest.raises(AsmError):
        assemble("top: BR top\nRET")


@pytest.mark.xfail(strict=True, reason="AUDIT-4 L3: branch targets past the end of the program are accepted")
def test_br_target_past_end_rejected():
    with pytest.raises(AsmError):
        assemble("BRnzp 200\nRET")
