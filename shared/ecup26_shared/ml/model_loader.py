"""Model loader with device detection (CUDA / CUDA+CPU offload / CPU).

Adapted from mlaw-rag backend/src/infrastructure/embedding_model.py.
Supports quantization: none | int8 | int4 via bitsandbytes.
"""
from __future__ import annotations

from typing import Any


def detect_device(requested: str | None = None) -> str:
    """Resolve device: requested if available, else CPU.

    For E-CUP we always want explicit CUDA when available (10-100x faster
    for Qwen3-Embedding-4B). CPU fallback for smoke tests / no-GPU servers.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"
    if requested in ("cuda", "cpu"):
        if requested == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(
    model_path: str,
    device: str = "cuda",
    quantization: str = "none",
    max_memory: dict | None = None,
) -> tuple[Any, Any]:
    """Load HF AutoModel + AutoTokenizer.

    Args:
        model_path: local path (no network in submission Docker).
        device: cuda | cpu (auto-offload для больших моделей на 8GB VRAM).
        quantization: none (FP16/BF16) | int8 | int4 (bitsandbytes).
        max_memory: per-device memory budget, e.g. {0: "6GiB", "cpu": "16GiB"}.

    Returns:
        (model, tokenizer) tuple.
    """
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "transformers required. Install with `uv add transformers`"
        ) from e

    kwargs: dict[str, Any] = {}

    if device == "cuda":
        try:
            import torch  # type: ignore
            kwargs["torch_dtype"] = torch.float16
        except ImportError:
            pass
        # Use device_map for auto-offload (handles models >VRAM gracefully)
        if quantization in ("int8", "int4"):
            kwargs["device_map"] = "auto"
            kwargs["quantization_config"] = _make_quant_config(quantization)
        else:
            kwargs["device_map"] = "auto"  # safer than .cuda() — handles offload
        if max_memory:
            kwargs["max_memory"] = max_memory
        else:
            # Default: 8GB VRAM budget per ADR-001
            kwargs["max_memory"] = {0: "6GiB", "cpu": "16GiB"}

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, **kwargs)
    model.eval()
    return model, tokenizer


def _make_quant_config(quantization: str) -> Any:
    """Build bitsandbytes quantization config."""
    try:
        from transformers import BitsAndBytesConfig  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "bitsandbytes required for int8/int4. Install with `uv add bitsandbytes`"
        ) from e

    if quantization == "int4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if quantization == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"Unknown quantization: {quantization}")
