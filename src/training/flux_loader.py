"""Helpers for loading FLUX.1-dev components."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


def _resolve_model_source(model_id: str) -> str:
    """Return local path or repo id for the model."""
    local_path = Path(model_id)
    if local_path.exists():
        logger.info("Используется локальная модель Flux: %s", local_path)
        return str(local_path)
    return model_id


def load_flux_components(
    model_id: str,
    dtype: torch.dtype,
    *,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> SimpleNamespace:
    """Load Flux pipeline using native components only."""
    source = _resolve_model_source(model_id)
    logger.info("Загрузка FLUX.1-dev из %s (dtype=%s)", source, dtype)

    pipeline = DiffusionPipeline.from_pretrained(
        source,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_safetensors=True,
        revision=revision,
        variant=variant,
    )

    processor = getattr(pipeline, "image_processor", getattr(pipeline, "processor", None))
    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        except Exception:
            processor = None

    # Prefer FLUX T5 text encoder/tokenizer (text_encoder_2/tokenizer_2) when available.
    text_encoder = getattr(pipeline, "text_encoder_2", None) or getattr(pipeline, "text_encoder", None)
    tokenizer = getattr(pipeline, "tokenizer_2", None) or getattr(pipeline, "tokenizer", None)

    components = SimpleNamespace(
        transformer=getattr(pipeline, "transformer", None),
        text_encoder=text_encoder,
        vae=getattr(pipeline, "vae", None),
        tokenizer=tokenizer,
        processor=processor,
        image_processor=processor,
        scheduler=getattr(pipeline, "scheduler", None),
        dtype=dtype,
    )
    return components


__all__ = ["load_flux_components"]
