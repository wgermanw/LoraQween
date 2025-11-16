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
import math
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
                # 1) Отфильтровать мусорные kwargs
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

                # 2) Достать hidden_states и encoder_hidden_states из args/kwargs
                hidden_states = kwargs.get("hidden_states", None)
                if hidden_states is None and len(args) > 0:
                    hidden_states = args[0]
                encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

                # 3) Определить device для новых тензоров
                device = None
                if isinstance(hidden_states, torch.Tensor):
                    device = hidden_states.device
                elif isinstance(encoder_hidden_states, torch.Tensor):
                    device = encoder_hidden_states.device
                else:
                    device = torch.device("cpu")

                # 4) Гарантировать txt_ids: 2D long-тензор [T_txt, 3]
                txt_ids = kwargs.get("txt_ids", None)
                if txt_ids is None:
                    if isinstance(encoder_hidden_states, torch.Tensor) and encoder_hidden_states.ndim >= 2:
                        txt_len = int(encoder_hidden_states.shape[1])
                    else:
                        txt_len = 1
                    txt_pos = torch.arange(txt_len, device=device, dtype=torch.long)
                    # Простейшая схема: t=0, h=0, w=индекс
                    txt_ids = torch.stack(
                        [
                            torch.zeros_like(txt_pos),  # time
                            torch.zeros_like(txt_pos),  # height
                            txt_pos,                    # width / порядковый индекс токена
                        ],
                        dim=-1,
                    )
                    kwargs["txt_ids"] = txt_ids

                # 5) Гарантировать img_ids: 2D long-тензор [T_img, 3]
                img_ids = kwargs.get("img_ids", None)
                if img_ids is None:
                    if isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3:
                        img_len = int(hidden_states.shape[1])
                    else:
                        img_len = 1
                    img_pos = torch.arange(img_len, device=device, dtype=torch.long)
                    # Прикидываем квадратичную "сетку" по высоте/ширине
                    side = int(math.ceil(math.sqrt(float(img_len))))
                    h = img_pos // side
                    w = img_pos % side
                    img_ids = torch.stack(
                        [
                            torch.ones_like(img_pos),  # time = 1 для картинки
                            h,
                            w,
                        ],
                        dim=-1,
                    )
                    kwargs["img_ids"] = img_ids

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

                # Patch text encoder forward to drop only unsupported 'inputs_embeds'
                peft_model = components.text_encoder
                base_model = getattr(peft_model, "base_model", None)
                if base_model is None and hasattr(peft_model, "model"):
                    base_model = peft_model.model
                if base_model is None:
                    base_model = peft_model

                orig_forward = base_model.forward

                def forward_no_inputs_embeds(*args, **kwargs):
                    if "inputs_embeds" in kwargs:
                        logger.warning(
                            "Dropping unsupported kwarg 'inputs_embeds' for CLIPTextModel.forward"
                        )
                        kwargs.pop("inputs_embeds", None)
                    return orig_forward(*args, **kwargs)

                base_model.forward = forward_no_inputs_embeds
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
            # держим T5 на CPU в float32, чтобы экономить VRAM
            t5_text_encoder = t5_text_encoder.to("cpu", dtype=torch.float32)
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

        # Определить in_features, которые ожидает x_embedder на последней оси
        x_in_features = None
        try:
            raw_tx = self.accelerator.unwrap_model(transformer)
            x_emb = getattr(raw_tx, "x_embedder", None)
            if x_emb is not None:
                if hasattr(x_emb, "in_features"):
                    x_in_features = int(getattr(x_emb, "in_features"))
                else:
                    weight = getattr(x_emb, "weight", None)
                    if weight is not None and hasattr(weight, "shape"):
                        # weight shape == (out_features, in_features)
                        x_in_features = int(weight.shape[1])
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
                    # T5 на CPU
                    t5_input_ids = t5_tok["input_ids"]
                    t5_attention_mask = t5_tok["attention_mask"]
                else:
                    # Fallback: use CLIP tokens if T5 tokenizer missing
                    t5_input_ids = clip_input_ids
                    t5_attention_mask = clip_attention_mask
                pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=dtype)

                with torch.no_grad():
                    # Encode to FLUX AE latents when available (preferred over raw pixels)
                    try:
                        enc_out = vae.encode(pixel_values)
                        if hasattr(enc_out, "latent_dist"):
                            pixel_latents = enc_out.latent_dist.sample()
                        elif hasattr(enc_out, "latents"):
                            pixel_latents = enc_out.latents
                        else:
                            pixel_latents = enc_out
                    except Exception:
                        # Fallback to raw pixels if AE encode is unavailable
                        pixel_latents = pixel_values.to(self.accelerator.device, dtype=dtype)
                    pixel_latents = pixel_latents.to(self.accelerator.device, dtype=dtype)
                    # T5 for encoder_hidden_states (B, seq, 4096)
                    if t5_text_encoder is not None:
                        t5_out = t5_text_encoder(
                            input_ids=t5_input_ids,
                            attention_mask=t5_attention_mask,
                        )
                        # перенести результат на девайс акселератора и нужный dtype
                        encoder_hidden_states = t5_out[0].to(self.accelerator.device, dtype=dtype)
                    else:
                        # Fallback to CLIP last_hidden_state (dim 768)
                        clip_out_fallback = clip_text_encoder(
                            input_ids=clip_input_ids,
                            attention_mask=clip_attention_mask,
                        )
                        encoder_hidden_states = clip_out_fallback[0]

                    # CLIP pooled for pooled_projections (B, 768)
                    pooled_from_clip = None
                    if clip_text_encoder is not None:
                        clip_out = clip_text_encoder(
                            input_ids=clip_input_ids,
                            attention_mask=clip_attention_mask,
                        )
                        # try pooler_output, else mean of last_hidden_state
                        pooled_from_clip = getattr(clip_out, "pooler_output", None)
                        if pooled_from_clip is None:
                            last_hidden = clip_out[0]
                            pooled_from_clip = last_hidden.mean(dim=1)

                # Flow Matching noise schedule for FLUX: mix latents with noise using random t in (0,1)
                bsz = pixel_latents.shape[0]
                t = torch.sigmoid(torch.randn((bsz,), device=self.accelerator.device, dtype=dtype))
                noise = torch.randn_like(pixel_latents)
                model_input = (1 - t.view(bsz, 1, 1, 1)) * pixel_latents + t.view(bsz, 1, 1, 1) * noise
                timesteps = t  # pass continuous t to transformer
                
                # Упаковать латенты как в официальном FluxPipeline: [B, C, H, W] -> [B, (H//2)*(W//2), C*4]
                def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
                    b, c, h, w = latents.shape
                    # гарантировать чётные H и W
                    if (h % 2) != 0:
                        latents = latents[:, :, :h - 1, :]
                        h = h - 1
                    if (w % 2) != 0:
                        latents = latents[:, :, :, :w - 1]
                        w = w - 1
                    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
                    latents = latents.permute(0, 2, 4, 1, 3, 5)
                    latents = latents.reshape(b, (h // 2) * (w // 2), c * 4)
                    return latents

                hidden_states = _pack_latents(model_input)
                if x_in_features is not None and hidden_states.shape[-1] != x_in_features:
                    raise RuntimeError(
                        f"packed latents last dim {hidden_states.shape[-1]} != x_embedder.in_features {x_in_features}"
                    )

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

                # Не передаём txt_ids/img_ids — пусть Flux сам их построит

                with self.accelerator.accumulate(transformer):
                    # Flux transformer expects hidden_states + timestep + encoder_hidden_states
                    model_out = transformer(
                        hidden_states=hidden_states,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        guidance=guidance_vec,
                        pooled_projections=pooled_projections,
                        return_dict=False,
                    )
                    model_pred = model_out[0] if isinstance(model_out, (tuple, list)) else model_out

                    # ------------------------------------------------------------
                    # Распаковать предсказание обратно в латентную сетку [B, C, H, W]
                    # чтобы оно совпадало по форме с noise / pixel_latents.
                    b, c, h, w = pixel_latents.shape
                    h_half, w_half = h // 2, w // 2

                    # Ожидаем форму [B, (H/2 * W/2), C*4]
                    if model_pred.shape[0] != b or model_pred.shape[1] != h_half * w_half or model_pred.shape[2] != c * 4:
                        raise RuntimeError(
                            f"Unexpected transformer output shape {tuple(model_pred.shape)} "
                            f"for latents shape {(b, c, h, w)} (expected tokens={(h_half * w_half, c * 4)})"
                        )

                    # [B, (H/2*W/2), C*4] -> [B, H/2, W/2, C, 2, 2] -> [B, C, H, W]
                    noise_pred = model_pred.view(b, h_half, w_half, c, 2, 2)
                    noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)

                    # Flow-matching таргет: (noise - latents)
                    target = (noise - pixel_latents).detach()

                    loss = F.mse_loss(noise_pred.float(), target.float(), reduction="mean")
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1
                progress_bar.set_description(f"loss: {loss.item():.4f}")
                progress_bar.update(1)
                if step >= max_steps:
                    break

        progress_bar.close()
        self._save_lora(transformer, clip_text_encoder)

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
