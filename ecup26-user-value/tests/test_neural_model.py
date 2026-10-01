"""RED tests for T3.19 neural models (GRU forward / determinism / batch-independence).

Per .clinerules: R1 (PyTorch, BSD), R6 (seeded/deterministic), R8 test-only.
"""

from __future__ import annotations

import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from experiment_neural_gru import GRUModel  # noqa: E402


class TestGRUModel:
    def test_forward_shape(self) -> None:
        m = GRUModel(d_in=16, hidden=64, layers=2)
        m.eval()
        x = torch.randn(8, 90, 16)
        out = m(x)
        assert out.shape == (8,), f"expected [8], got {out.shape}"

    def test_determinism(self) -> None:
        torch.manual_seed(42)
        m1 = GRUModel(d_in=16, hidden=64, layers=2)
        torch.manual_seed(42)
        m2 = GRUModel(d_in=16, hidden=64, layers=2)
        x = torch.randn(4, 30, 16)
        m1.eval(); m2.eval()
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x), atol=1e-6)  # same seed -> same weights/output

    def test_batch_independence(self) -> None:
        """Predictions of a single sample do not depend on the batch neighbours (eval mode)."""
        torch.manual_seed(0)
        m = GRUModel(d_in=16, hidden=32, layers=1)
        m.eval()
        a = torch.randn(1, 45, 16)
        b = torch.randn(1, 45, 16)
        with torch.no_grad():
            p_single = m(a)
            p_batch = m(torch.cat([a, b], dim=0))
        assert torch.allclose(p_single[0], p_batch[0], atol=1e-6)

    def test_finite_output(self) -> None:
        torch.manual_seed(1)
        m = GRUModel(d_in=16, hidden=64, layers=2)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(16, 180, 16))
        assert torch.isfinite(out).all()
