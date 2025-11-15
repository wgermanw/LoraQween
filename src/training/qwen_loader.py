"""Helpers for loading Qwen/Qwen-Image components in low-memory mode."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Any, Dict

import torch
from diffusers import DiffusionPipeline, DDPMScheduler
from transformers import AutoProcessor, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class LoadedComponents:
    """Container for the minimal set of modules required for LoRA training."""

    transformer: Any
    text_encoder: Any
    vae: Any
    tokenizer: Any
    processor: Optional[Any]
    scheduler: Optional[Any]


def _build_device_map(preferred_device: str) -> str:
    """Return diffusers-compatible device_map representation."""
    if preferred_device == "cpu":
        return "cpu"
    return "balanced"


def load_qwen_components(
    model_id: str,
    dtype: torch.dtype,
    *,
    tokenizer_path: Optional[Path] = None,
    max_cpu_ram_gb: int = 40,
    max_vram_gb: int = 24,
    device: str = "cuda",
    offload_state_dict: bool = True,
) -> SimpleNamespace:
    """
    Load Qwen-Image components with as little RAM pressure as possible.

    Args:
        model_id: Hugging Face id or local path with the base model weights.
        dtype: torch dtype used for the transformer.
        tokenizer_path: Optional override for tokenizer (dataset-specific tokens).
        max_cpu_ram_gb: Cap for CPU RAM during loading.
        max_vram_gb: Cap for GPU memory.
        device: Preferred device for the trainable transformer block.
        offload_state_dict: Whether to offload weights back to CPU after loading.

    Returns:
        SimpleNamespace with transformer, text_encoder, vae, tokenizer, processor, scheduler.
    """
    if tokenizer_path is not None and not isinstance(tokenizer_path, Path):
        tokenizer_path = Path(tokenizer_path)

    loading_kwargs: Dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "use_safetensors": True,
        "low_cpu_mem_usage": True,
        "device_map": _build_device_map(device if torch.cuda.is_available() else "cpu"),
    }

    max_memory: Dict[Any, str] = {}
    if torch.cuda.is_available() and max_vram_gb:
        max_memory[0] = f"{max_vram_gb}GiB"
    if max_cpu_ram_gb:
        max_memory["cpu"] = f"{max_cpu_ram_gb}GiB"
    if max_memory:
        loading_kwargs["max_memory"] = max_memory
    if offload_state_dict:
        loading_kwargs["offload_state_dict"] = True

    logger.info(
        "Loading Qwen-Image with low_cpu_mem_usage=%s, device_map=%s, max_memory=%s",
        loading_kwargs["low_cpu_mem_usage"],
        loading_kwargs["device_map"],
        loading_kwargs.get("max_memory"),
    )

    pipeline = DiffusionPipeline.from_pretrained(model_id, **loading_kwargs)

    processor = getattr(pipeline, "image_processor", getattr(pipeline, "processor", None))
    scheduler = getattr(pipeline, "scheduler", None)

    components = SimpleNamespace(
        transformer=getattr(pipeline, "transformer", getattr(pipeline, "unet", None)),
        text_encoder=getattr(pipeline, "text_encoder", None),
        vae=getattr(pipeline, "vae", None),
        tokenizer=getattr(pipeline, "tokenizer", None),
        processor=processor,
        image_processor=processor,
        scheduler=scheduler,
    )

    # Replace tokenizer if dataset provided its own vocabulary
    tokenizer_source = tokenizer_path if tokenizer_path and tokenizer_path.exists() else None
    if tokenizer_source:
        components.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    elif components.tokenizer is None:
        components.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    if components.processor is None:
        try:
            components.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            components.image_processor = components.processor
        except Exception:
            components.processor = None
            components.image_processor = None

    if components.scheduler is None:
        components.scheduler = DDPMScheduler.from_pretrained(
            model_id, subfolder="scheduler", trust_remote_code=True
        )

    # Drop the heavy pipeline container to free RAM immediately
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "Loaded Qwen components: transformer=%s, text_encoder=%s, vae=%s",
        type(components.transformer).__name__ if components.transformer else "N/A",
        type(components.text_encoder).__name__ if components.text_encoder else "N/A",
        type(components.vae).__name__ if components.vae else "N/A",
    )
    return components


__all__ = ["load_qwen_components", "LoadedComponents"]
