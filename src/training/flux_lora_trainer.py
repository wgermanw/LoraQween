"""Flux-specific LoRA trainer."""

from __future__ import annotations

import json
import logging
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from .flux_loader import load_flux_components

logger = logging.getLogger(__name__)


class FluxPersonDataset(Dataset):
    """Dataset returning pixel tensors and raw captions from metadata.jsonl."""

    def __init__(self, dataset_dir: Path, resolution: int = 512):
        from PIL import Image
        import torchvision.transforms as transforms

        self.dataset_dir = Path(dataset_dir)
        self.items = []
        self.Image = Image
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        metadata = self.dataset_dir / "metadata.jsonl"
        with open(metadata, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                image_path = self.dataset_dir / "images" / obj["file_name"]
                if image_path.exists():
                    self.items.append({
                        "image_path": image_path,
                        "caption": obj["caption"],
                    })
        logger.info("Flux dataset loaded: %s samples", len(self.items))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = self.Image.open(item["image_path"]).convert("RGB")
        return {
            "pixel_values": self.transform(image),
            "captions": item["caption"],
        }


class FluxLoRATrainer:
    """Dedicated trainer for FLUX.1-dev."""

    def __init__(self, model_config: Dict, dataset_dir: Path, output_dir: Path, config: Dict):
        self.model_config = model_config
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
            mixed_precision=config.get("mixed_precision", "bf16"),
        )

    def setup(self):
        dtype = torch.bfloat16 if self.model_config.get("dtype", "bf16") == "bf16" else torch.float16
        components = load_flux_components(
            self.model_config.get("base_model_id"),
            dtype,
            revision=self.model_config.get("revision"),
            variant=self.model_config.get("variant"),
        )

        target_modules = self.config.get("target_modules", ["to_q", "to_k", "to_v", "to_out"])
        lora_cfg = LoraConfig(
            r=int(self.config.get("lora_rank", 16)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.0)),
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        components.transformer = get_peft_model(components.transformer, lora_cfg)
        # Patch Flux transformer forward to discard any stray/unsupported kwargs (e.g., input_ids)
        try:
            peft_tx = components.transformer
            base_tx = getattr(peft_tx, "base_model", None)
            if base_tx is None and hasattr(peft_tx, "model"):
                base_tx = peft_tx.model
            if base_tx is None:
                base_tx = peft_tx
            tx_orig_forward = base_tx.forward
            tx_allowed_keys = {
                "hidden_states",
                "timestep",
                "encoder_hidden_states",
                "pooled_projections",
                "guidance",
                "txt_ids",
                "img_ids",
                "controlnet_block_samples",
                "controlnet_single_block_samples",
                "return_dict",
            }

            def tx_forward_filtered(*args, **kwargs):
                removed = []
                for key in list(kwargs.keys()):
                    if key not in tx_allowed_keys:
                        kwargs.pop(key, None)
                        removed.append(key)
                if removed:
                    logger.warning(
                        "Dropping unsupported kwargs %s for FluxTransformer2DModel.forward",
                        ", ".join(sorted(set(removed))),
                    )
                return tx_orig_forward(*args, **kwargs)

            base_tx.forward = tx_forward_filtered
        except Exception:
            # If patching fails, proceed without filtering
            pass

        if components.text_encoder is not None:
            # Determine appropriate target modules for text encoder (CLIP vs T5)
            requested_targets = self.config.get("text_encoder_target_modules")
            # Collect module names to detect availability
            module_names = [name for name, _ in components.text_encoder.named_modules()]

            def has_mod(mod_name: str) -> bool:
                dot = f".{mod_name}"
                return any(n.endswith(dot) or n == mod_name for n in module_names)

            text_targets: list[str] = []
            if requested_targets:
                # Use requested only if at least one target exists
                if any(has_mod(t) for t in requested_targets):
                    text_targets = list(requested_targets)
                else:
                    logger.warning("Requested text_encoder_target_modules %s not found, auto-detecting...",
                                   requested_targets)
            if not text_targets:
                # Prefer CLIP-style if present
                out_name = "out_proj" if has_mod("out_proj") else ("o_proj" if has_mod("o_proj") else None)
                if has_mod("q_proj") and has_mod("k_proj") and has_mod("v_proj") and out_name:
                    text_targets = ["q_proj", "k_proj", "v_proj", out_name]
                # Else try T5-style names
                elif has_mod("q") and has_mod("k") and has_mod("v") and has_mod("o"):
                    text_targets = ["q", "k", "v", "o"]
                else:
                    # As a last resort, include whichever of common names are present
                    common = ["q_proj", "k_proj", "v_proj", "out_proj", "o_proj", "q", "k", "v", "o"]
                    text_targets = [t for t in common if has_mod(t)]

            if text_targets:
                text_cfg = LoraConfig(
                    r=int(self.config.get("lora_rank", 16)),
                    lora_alpha=int(self.config.get("lora_alpha", 32)),
                    lora_dropout=float(self.config.get("lora_dropout", 0.0)),
                    target_modules=text_targets,
                    bias="none",
                    task_type=TaskType.FEATURE_EXTRACTION,
                )
                components.text_encoder = get_peft_model(components.text_encoder, text_cfg)

                # Patch text encoder forward to discard unsupported kwargs such as inputs_embeds.
                peft_model = components.text_encoder
                base_model = getattr(peft_model, "base_model", None)
                if base_model is None and hasattr(peft_model, "model"):
                    base_model = peft_model.model
                if base_model is None:
                    base_model = peft_model

                orig_forward = base_model.forward
                allowed_keys = {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "return_dict",
                    "output_attentions",
                    "output_hidden_states",
                }

                def forward_filtered(*args, **kwargs):
                    removed = []
                    for key in list(kwargs.keys()):
                        if key not in allowed_keys:
                            kwargs.pop(key, None)
                            removed.append(key)
                    if removed:
                        logger.warning(
                            "Dropping unsupported kwargs %s for text_encoder.forward",
                            ", ".join(sorted(set(removed))),
                        )
                    return orig_forward(*args, **kwargs)

                base_model.forward = forward_filtered
            else:
                logger.warning("No matching target modules found in text_encoder; skipping LoRA on text_encoder.")

        return components, dtype

    def train(self):
        components, dtype = self.setup()
        clip_tokenizer = components.tokenizer
        t5_tokenizer = getattr(components, "tokenizer_2", None)
        vae = components.vae.to(self.accelerator.device, dtype=dtype)
        clip_text_encoder = components.text_encoder.to(self.accelerator.device, dtype=dtype) if components.text_encoder is not None else None
        t5_text_encoder = getattr(components, "text_encoder_2", None)
        if t5_text_encoder is not None:
            t5_text_encoder = t5_text_encoder.to(self.accelerator.device, dtype=dtype)
        transformer = components.transformer
        scheduler = components.scheduler or DDPMScheduler.from_pretrained(
            self.model_config.get("base_model_id"), subfolder="scheduler"
        )

        if clip_text_encoder is not None:
            clip_text_encoder.eval()
        vae.eval()
        if clip_text_encoder is not None:
            for param in clip_text_encoder.parameters():
                param.requires_grad = False
        if t5_text_encoder is not None:
            t5_text_encoder.eval()
            for param in t5_text_encoder.parameters():
                param.requires_grad = False
        for param in vae.parameters():
            param.requires_grad = False

        dataset = FluxPersonDataset(self.dataset_dir, resolution=self.config.get("resolution", 512))
        dataloader = DataLoader(
            dataset,
            batch_size=int(self.config.get("batch_size", 1)),
            shuffle=True,
            num_workers=0,
        )

        optimizer = torch.optim.AdamW(
            [p for p in transformer.parameters() if p.requires_grad],
            lr=float(self.config.get("learning_rate", 5e-5)),
        )

        transformer, optimizer, dataloader = self.accelerator.prepare(transformer, optimizer, dataloader)

        max_steps = int(self.config.get("max_steps", 1000))
        progress_bar = tqdm(range(max_steps), disable=not self.accelerator.is_local_main_process)

        # Try to detect expected transformer x-embed input size once
        try:
            raw_tx = self.accelerator.unwrap_model(transformer)
            x_embedder = getattr(raw_tx, "x_embedder", None)
            x_in_features = getattr(x_embedder, "in_features", None)
        except Exception:
            x_in_features = None
        
        # Try to detect pooled_projection expected dimension for time_text_embed
        pooled_proj_dim = None
        try:
            if "raw_tx" not in locals():
                raw_tx = self.accelerator.unwrap_model(transformer)
            time_text_embed = getattr(raw_tx, "time_text_embed", None)
            if time_text_embed is not None:
                # Common attribute name
                proj_layer = getattr(time_text_embed, "pooled_projection_proj", None)
                if hasattr(proj_layer, "in_features"):
                    pooled_proj_dim = int(getattr(proj_layer, "in_features"))
                else:
                    # Heuristic: find a Linear with 'pooled' in its name
                    for name, module in time_text_embed.named_modules():
                        if "pooled" in name and hasattr(module, "in_features"):
                            pooled_proj_dim = int(getattr(module, "in_features"))
                            break
        except Exception:
            pooled_proj_dim = None

        step = 0
        while step < max_steps:
            for batch in dataloader:
                captions = batch["captions"]
                caption_texts = [c.strip() if isinstance(c, str) else "" for c in captions]
                # Tokenize for CLIP (for pooled_projections, dim 768)
                clip_tok = clip_tokenizer(
                    caption_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=min(getattr(clip_tokenizer, "model_max_length", 77), 77),
                    return_tensors="pt",
                )
                clip_input_ids = clip_tok["input_ids"].to(self.accelerator.device)
                clip_attention_mask = clip_tok["attention_mask"].to(self.accelerator.device)

                # Tokenize for T5 (for encoder_hidden_states, dim 4096)
                if t5_tokenizer is not None:
                    t5_tok = t5_tokenizer(
                        caption_texts,
                        padding="max_length",
                        truncation=True,
                        max_length=min(getattr(t5_tokenizer, "model_max_length", 512), 512),
                        return_tensors="pt",
                    )
                    t5_input_ids = t5_tok["input_ids"].to(self.accelerator.device)
                    t5_attention_mask = t5_tok["attention_mask"].to(self.accelerator.device)
                else:
                    # Fallback: use CLIP tokens if T5 tokenizer missing
                    t5_input_ids = clip_input_ids
                    t5_attention_mask = clip_attention_mask
                pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * getattr(vae.config, "scaling_factor", 0.18215)
                    # T5 for encoder_hidden_states (B, seq, 4096)
                    if t5_text_encoder is not None:
                        t5_out = t5_text_encoder(
                            input_ids=t5_input_ids,
                            attention_mask=t5_attention_mask,
                            return_dict=True,
                        )
                        encoder_hidden_states = getattr(t5_out, "last_hidden_state", None)
                        if encoder_hidden_states is None:
                            encoder_hidden_states = t5_out[0]
                    else:
                        # Fallback to CLIP last_hidden_state (dim 768)
                        clip_out_fallback = clip_text_encoder(
                            input_ids=clip_input_ids,
                            attention_mask=clip_attention_mask,
                            return_dict=True,
                        )
                        encoder_hidden_states = getattr(clip_out_fallback, "last_hidden_state", None)
                        if encoder_hidden_states is None:
                            encoder_hidden_states = clip_out_fallback[0]
                    # CLIP pooled for pooled_projections (B, 768)
                    pooled_from_clip = None
                    if clip_text_encoder is not None:
                        clip_out = clip_text_encoder(
                            input_ids=clip_input_ids,
                            attention_mask=clip_attention_mask,
                            return_dict=True,
                        )
                        pooled_from_clip = getattr(clip_out, "pooler_output", None)
                        if pooled_from_clip is None:
                            # mean pool across sequence
                            last_hidden = getattr(clip_out, "last_hidden_state", None)
                            if last_hidden is None:
                                last_hidden = clip_out[0]
                            pooled_from_clip = last_hidden.mean(dim=1)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                    dtype=torch.long,
                )
                # Add noise depending on scheduler capabilities
                if hasattr(scheduler, "add_noise"):
                    timesteps = timesteps.to(latents.device)
                    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                elif hasattr(scheduler, "sigmas"):
                    indices = timesteps
                    if indices.dtype != torch.long:
                        indices = indices.to(torch.long)
                    indices = indices.to(device=scheduler.sigmas.device)
                    sigmas = scheduler.sigmas[indices]
                    sigmas = sigmas.to(latents.device)
                    while sigmas.ndim < latents.ndim:
                        sigmas = sigmas.view(-1, *([1] * (latents.ndim - 1)))
                    noisy_latents = latents + sigmas * noise
                else:
                    noisy_latents = latents + noise
                
                # Scale input for scheduler if required
                model_input = noisy_latents
                if hasattr(scheduler, "scale_model_input"):
                    model_input = scheduler.scale_model_input(noisy_latents, timesteps)

                # If transformer expects flattened patch vectors (e.g., in_features=3072),
                # and we currently have BCHW tensors, convert to (B, tokens, in_features)
                if x_in_features is not None and model_input.ndim == 4:
                    bsz, channels, height, width = model_input.shape
                    if x_in_features % channels == 0:
                        required_area = x_in_features // channels
                        # Find (kh, kw) such that kh*kw == required_area and divides HxW grid
                        kh, kw = None, None
                        # Prefer square-ish factors, but accept any valid tiling
                        for h_factor in range(int(required_area**0.5), 0, -1):
                            if required_area % h_factor != 0:
                                continue
                            w_factor = required_area // h_factor
                            if h_factor <= height and w_factor <= width and height % h_factor == 0 and width % w_factor == 0:
                                kh, kw = h_factor, w_factor
                                break
                        # Fallback to row-wise tiling if possible
                        if kh is None and required_area <= width and width % required_area == 0:
                            kh, kw = 1, required_area
                        # Apply unfold if we found a feasible tiling
                        if kh is not None and kw is not None:
                            unfolded = F.unfold(model_input, kernel_size=(kh, kw), stride=(kh, kw))  # [B, C*kh*kw, L]
                            model_input = unfolded.transpose(1, 2).contiguous()  # [B, L, x_in_features]
                        else:
                            # As a last resort, keep BCHW; the model may still support it
                            pass

                # Prepare required conditioning for Flux time/text embedding
                bsz = model_input.shape[0]
                try:
                    guidance_value = float(self.config.get("guidance", 3.5))
                except Exception:
                    guidance_value = 3.5
                guidance_vec = torch.full(
                    (bsz,),
                    guidance_value,
                    device=self.accelerator.device,
                    dtype=dtype,
                )
                # Prepare pooled_projections from CLIP pooled output (expected dim 768 for Flux)
                if pooled_proj_dim is not None and pooled_from_clip is not None and pooled_from_clip.shape[-1] == pooled_proj_dim:
                    pooled_projections = pooled_from_clip
                else:
                    # Fallback to zeros if dimension mismatch
                    target_dim = pooled_proj_dim if pooled_proj_dim is not None else (pooled_from_clip.shape[-1] if pooled_from_clip is not None else 768)
                    pooled_projections = torch.zeros(
                        (bsz, target_dim),
                        device=self.accelerator.device,
                        dtype=dtype,
                    )

                with self.accelerator.accumulate(transformer):
                    # Flux transformer expects hidden_states + timestep + encoder_hidden_states
                    model_out = transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        guidance=guidance_vec,
                        pooled_projections=pooled_projections,
                        return_dict=False,
                    )
                    model_pred = model_out[0] if isinstance(model_out, (tuple, list)) else model_out
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                progress_bar.set_description(f"loss: {loss.item():.4f}")
                progress_bar.update(1)
                if step >= max_steps:
                    break

        progress_bar.close()
        self._save_lora(transformer, text_encoder)

        manifest = {
            "training_date": datetime.now().isoformat(),
            "backend": "flux",
            "dataset_dir": str(self.dataset_dir),
            "config": self.config,
        }
        with open(self.output_dir / "training_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        return self.output_dir / "training_manifest.json"

    def _save_lora(self, transformer, text_encoder):
        ckpt = self.output_dir / "checkpoint-final"
        ckpt.mkdir(parents=True, exist_ok=True)
        unwrap = self.accelerator.unwrap_model
        if hasattr(transformer, "save_pretrained"):
            unwrap(transformer).save_pretrained(ckpt / "transformer")
        if text_encoder is not None and hasattr(text_encoder, "save_pretrained"):
            unwrap(text_encoder).save_pretrained(ckpt / "text_encoder")


__all__ = ["FluxLoRATrainer", "FluxPersonDataset"]
