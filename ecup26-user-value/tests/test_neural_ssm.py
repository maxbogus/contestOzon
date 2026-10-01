"""RED tests for T3.20 sequence encoders: Mamba (S6), RWKV, xLSTM (mLSTM).

Self-implemented pure-PyTorch blocks (R1: no external deps, BSD-compatible).
Same interface as SequenceModel: embedding -> encoder -> attention pool -> MLP head,
d_in=16 (the neural-sweep config that produced the LB winner).

Per .clinerules: R1 (PyTorch, BSD), R6 (seeded/deterministic).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from neural_models import (  # noqa: E402
    SequenceModelMamba,
    SequenceModelRWKV,
    SequenceModelXLSTM,
)

MODELS = [
    SequenceModelMamba,
    SequenceModelRWKV,
    SequenceModelXLSTM,
]


@pytest.mark.parametrize("cls", MODELS)
class TestSequenceSSM:
    def test_forward_shape_full_history(self, cls) -> None:
        m = cls(d_in=16, hidden=128, layers=2)
        m.eval()
        out = m(torch.randn(8, 409, 16))  # L=409, d_in=16 (production shape)
        assert out.shape == (8,), f"expected [8], got {out.shape}"

    def test_forward_shape_short(self, cls) -> None:
        m = cls(d_in=16, hidden=64, layers=1)
        m.eval()
        out = m(torch.randn(4, 30, 16))
        assert out.shape == (4,)

    def test_layers_variants(self, cls) -> None:
        for layers in (1, 2, 3):
            m = cls(d_in=16, hidden=64, layers=layers)
            m.eval()
            assert m(torch.randn(2, 50, 16)).shape == (2,)

    def test_finite_output(self, cls) -> None:
        torch.manual_seed(1)
        m = cls(d_in=16, hidden=64, layers=2)
        m.eval()
        with torch.no_grad():
            out = m(torch.randn(16, 180, 16))
        assert torch.isfinite(out).all()

    def test_determinism(self, cls) -> None:
        torch.manual_seed(42)
        m1 = cls(d_in=16, hidden=64, layers=2)
        torch.manual_seed(42)
        m2 = cls(d_in=16, hidden=64, layers=2)
        x = torch.randn(4, 100, 16)
        m1.eval()
        m2.eval()
        with torch.no_grad():
            assert torch.allclose(m1(x), m2(x), atol=1e-6)

    def test_batch_independence(self, cls) -> None:
        """No leaky internal state: repeated calls on the same batch are identical."""
        torch.manual_seed(0)
        m = cls(d_in=16, hidden=32, layers=1)
        m.eval()
        x = torch.randn(4, 45, 16)
        with torch.no_grad():
            o1 = m(x).clone()
            o2 = m(x)
        assert torch.allclose(o1, o2, atol=1e-6)

    def test_training_step_stable(self, cls) -> None:
        """One Adam step must keep outputs finite (no exp blowup in scan/gates)."""
        torch.manual_seed(0)
        m = cls(d_in=16, hidden=64, layers=2)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        x = torch.randn(8, 120, 16)
        y = torch.randn(8)
        loss = torch.nn.functional.mse_loss(m(x), y)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss).item()
        for p in m.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {p.shape}"
        loss2 = torch.nn.functional.mse_loss(m(x), y)
        assert torch.isfinite(loss2).item()

    def test_gradients_flow(self, cls) -> None:
        m = cls(d_in=16, hidden=32, layers=1)
        x = torch.randn(4, 60, 16, requires_grad=True)
        out = m(x)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        n_grad = sum(1 for p in m.parameters() if p.grad is not None)
        assert n_grad == len(list(m.parameters())), "some parameters got no gradient"
