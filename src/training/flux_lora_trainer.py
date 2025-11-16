"""Flux-specific LoRA trainer."""

from __future__ import annotations

import json
import logging
import inspect
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from accelerate import Accelerator
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

        if self.config.get("text_encoder_target_modules"):
            text_cfg = LoraConfig(
                r=int(self.config.get("lora_rank", 16)),
                lora_alpha=int(self.config.get("lora_alpha", 32)),
                lora_dropout=float(self.config.get("lora_dropout", 0.0)),
                target_modules=self.config.get("text_encoder_target_modules", []),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            components.text_encoder = get_peft_model(components.text_encoder, text_cfg)

            # Patch CLIP text encoder forward to discard unsupported kwargs such as inputs_embeds.
            peft_model = components.text_encoder
            base_model = getattr(peft_model, "base_model", None)
            if base_model is None and hasattr(peft_model, "model"):
                base_model = peft_model.model
            if base_model is None:
                base_model = peft_model

            orig_forward = base_model.forward
            # Use an allowlist of known-safe kwargs for CLIPTextModel to avoid
            # accidentally dropping required ones due to wrapper layers.
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
                        "Dropping unsupported kwargs %s for CLIPTextModel.forward",
                        ", ".join(sorted(set(removed))),
                    )
                return orig_forward(*args, **kwargs)

            base_model.forward = forward_filtered

        return components, dtype

    def train(self):
        components, dtype = self.setup()
        tokenizer = components.tokenizer
        vae = components.vae.to(self.accelerator.device, dtype=dtype)
        text_encoder = components.text_encoder.to(self.accelerator.device, dtype=dtype)
        transformer = components.transformer
        scheduler = components.scheduler or DDPMScheduler.from_pretrained(
            self.model_config.get("base_model_id"), subfolder="scheduler"
        )

        text_encoder.eval()
        vae.eval()
        for param in text_encoder.parameters():
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

        step = 0
        while step < max_steps:
            for batch in dataloader:
                captions = batch["captions"]
                caption_texts = [c.strip() if isinstance(c, str) else "" for c in captions]
                tokenized = tokenizer(
                    caption_texts,
                    padding="max_length",
                    truncation=True,
                    max_length=min(tokenizer.model_max_length, 77),
                    return_tensors="pt",
                )
                input_ids = tokenized["input_ids"].to(self.accelerator.device)
                attention_mask = tokenized["attention_mask"].to(self.accelerator.device)
                pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=dtype)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * getattr(vae.config, "scaling_factor", 0.18215)
                    encoder_hidden_states = text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )[0]

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

                with self.accelerator.accumulate(transformer):
                    # Flux transformer expects hidden_states + timestep + encoder_hidden_states
                    model_out = transformer(
                        hidden_states=model_input,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
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
