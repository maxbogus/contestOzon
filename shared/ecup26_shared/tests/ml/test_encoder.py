"""Tests for shared.ml.encoder and shared.ml.model_loader."""
from __future__ import annotations

import numpy as np


class TestEncoderMRL:
    """TDD: encoder must support MRL truncation."""

    def test_default_dim_is_2560(self):
        from ecup26_shared.ml.encoder import EncoderConfig
        c = EncoderConfig(model_path="dummy")
        assert c.dim == 2560

    def test_mrl_dim_truncates(self):
        from ecup26_shared.ml.encoder import EncoderConfig
        c = EncoderConfig(model_path="dummy", mrl_dim=1024)
        assert c.dim == 1024


class TestL2Normalize:
    """Pure function: l2_normalize"""

    def test_unit_norm_rows(self):
        from ecup26_shared.ml.encoder import l2_normalize
        vecs = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
        out = l2_normalize(vecs)
        norm = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norm, [1.0], atol=1e-5)

    def test_zero_vector_handled(self):
        from ecup26_shared.ml.encoder import l2_normalize
        vecs = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        out = l2_normalize(vecs)
        # Should not divide by zero
        assert np.all(np.isfinite(out))


class TestDeviceDetection:
    """detect_device: cuda if available else cpu."""

    def test_returns_string(self):
        from ecup26_shared.ml.model_loader import detect_device
        dev = detect_device()
        assert dev in ("cuda", "cpu")
