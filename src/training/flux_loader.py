"""Helpers for loading FLUX.1-dev components."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from transformers import AutoTokenizer, AutoProcessor

logger = logging.getLogger(__name__)


def _resolve_model_source(base_model_id: str, local_override: Optional[Path] = None) -> str:
    """Return path or repo id that should be used for loading."""
    if local_override and local_override.exists():
        logger.info("Используется локальная модель Flux: %s", local_override)
        return str(local_override)
    if Path(base_model_id).exists():
        logger.info("Обнаружена локальная директория модели: %s", base_model_id)
        return str(Path(base_model_id))
    return base_model_id


def load_flux_components(
    model_id: str,
    dtype: torch.dtype,
    *,
    tokenizer_path: Optional[Path] = None,
    local_dir: Optional[Path] = None,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> SimpleNamespace:
    """
    Load FLUX.1-dev pipeline components without aggressive device/offload logic.
    """
    if tokenizer_path is not None and not isinstance(tokenizer_path, Path):
        tokenizer_path = Path(tokenizer_path)

    source = _resolve_model_source(model_id, local_override=local_dir)
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

    tokenizer = None
    if tokenizer_path and tokenizer_path.exists():
        logger.info("Используется кастомный токенайзер: %s", tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = getattr(pipeline, "tokenizer", None)
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            except Exception:
                tokenizer = None

    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        except Exception:
            processor = None

    components = SimpleNamespace(
        transformer=getattr(pipeline, "transformer", None),
        text_encoder=getattr(pipeline, "text_encoder", None),
        vae=getattr(pipeline, "vae", None),
        tokenizer=tokenizer,
        processor=processor,
        image_processor=processor,
        scheduler=getattr(pipeline, "scheduler", None),
        dtype=dtype,
    )
    return components


__all__ = ["load_flux_components"]
