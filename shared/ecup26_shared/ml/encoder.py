"""BiEncoder-style encoder with MRL (Matryoshka Representation Learning).

Reusable by ecup26-matching (Qwen3-Embedding-4B), ecup26-user-value (search
query embeddings, T3.5.3), and other tracks needing multilingual embeddings.

Adapted from mlaw-rag backend/src/application/embedding_service.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class EncoderConfig:
    """Encoder configuration.

    Attributes:
        model_path: Path to local model directory (no HF download).
        dim: Output dimension. Default 2560 (Qwen3-Embedding-4B native).
              Can be truncated via MRL to 1024/512/etc.
        normalize: L2-normalize output (default True, required for cosine).
        batch_size: Batch size for encode() — keep small for 8GB VRAM.
    """

    model_path: str
    dim: int = 2560
    mrl_dim: int | None = None
    normalize: bool = True
    batch_size: int = 16
    max_length: int = 512
    device: str | None = None  # auto-detect
    quantization: str = "none"  # none | int8 | int4

    def __post_init__(self) -> None:
        if self.mrl_dim is not None:
            object.__setattr__(self, "dim", self.mrl_dim)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    """L2-normalize each row to unit length. Zero vectors stay zero."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


class BiEncoder:
    """Lazy-loaded bi-encoder with MRL + L2 normalization.

    Usage:
        cfg = EncoderConfig(model_path="models/Qwen3-Embedding-4B", mrl_dim=1024)
        encoder = BiEncoder(cfg)
        encoder.load()
        vecs = encoder.encode(["text1", "text2"])  # (2, 1024)
    """

    def __init__(self, config: EncoderConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    def load(self) -> BiEncoder:
        """Lazy load model + tokenizer (HF AutoModel)."""
        if self._model is not None:
            return self
        from ecup26_shared.ml.model_loader import detect_device, load_model_and_tokenizer

        self._device = self.config.device or detect_device()
        self._model, self._tokenizer = load_model_and_tokenizer(
            self.config.model_path,
            device=self._device,
            quantization=self.config.quantization,
        )
        return self

    def _mean_pool(self, outputs: Any, mask: Any) -> np.ndarray:
        """Mask-aware mean pooling over token embeddings."""
        mask_f = mask.unsqueeze(-1).float()
        summed = (outputs.last_hidden_state * mask_f).sum(dim=1)
        counts = mask_f.sum(dim=1).clamp(min=1e-9)
        return (summed / counts).cpu().numpy()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode list of texts to (n, dim) embeddings.

        - Mean pooling over tokens (mask-aware)
        - MRL truncation if mrl_dim set
        - L2 normalization if normalize=True
        """
        if self._model is None:
            self.load()

        if not texts:
            return np.zeros((0, self.config.dim), dtype=np.float32)

        try:
            import torch  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "torch required. Install with `uv add torch transformers`"
            ) from e

        all_vecs = []
        bs = self.config.batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self._model(**inputs)
            vecs = self._mean_pool(outputs, inputs["attention_mask"]).astype(np.float32)
            if self.config.normalize:
                vecs = l2_normalize(vecs)
            all_vecs.append(vecs)
        result = np.vstack(all_vecs).astype(np.float32)
        # MRL truncation
        if result.shape[1] > self.config.dim:
            result = result[:, : self.config.dim]
            if self.config.normalize:
                result = l2_normalize(result)
        return result
