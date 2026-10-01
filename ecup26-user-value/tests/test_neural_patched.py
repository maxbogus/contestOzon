"""RED tests for T3.19 SequenceModelPatched (patch-token Transformer).

Per .clinerules: R1 (PyTorch BSD), R6 (seeded/deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from neural_models import SequenceModelPatched  # noqa: E402


class TestSequenceModelPatched:
    def test_forward_shape_full_history(self) -> None:
        m = SequenceModelPatched(d_in=16, hidden=64, layers=2, patch=7)
        m.eval()
        out = m(torch.randn(8, 409, 16))  # L=409 -> 59 patches
        assert out.shape == (8,), f"expected [8], got {out.shape}"

    def test_forward_shape_non_multiple(self) -> None:
        """L not divisible by patch -> trailing zero-pad path must work."""
        m = SequenceModelPatched(d_in=16, hidden=64, layers=2, patch=7)
        m.eval()
        out = m(torch.randn(4, 400, 16))  # 400 = 57*7 + 1 -> pads to 58 patches
        assert out.shape == (4,)

    def test_determinism(self) -> None:
        torch.manual_seed(42)
        m1 = SequenceModelPatched(d_in=16, hidden=64, layers=2, patch=7)
        torch.manual_seed(42)
        m2 = SequenceModelPatched(d_in=16, hidden=64, layers=2, patch=7)
        x = torch.randn(4, 100, 16)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x), atol=1e-6)

    def test_batch_independence(self) -> None:
        torch.manual_seed(0)
        m = SequenceModelPatched(d_in=16, hidden=32, layers=1, patch=7)
        m.eval()
        a = torch.randn(1, 45, 16)
        b = torch.randn(1, 45, 16)
        with torch.no_grad():
            p_single = m(a)
            p_batch = m(torch.cat([a, b], dim=0))
        assert torch.allclose(p_single[0], p_batch[0], atol=1e-6)

    def test_finite_output(self) -> None:
        torch.manual_seed(1)
        m = SequenceModelPatched(d_in=16, hidden=64, layers=2, patch=14)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(16, 180, 16))
        assert torch.isfinite(out).all()
