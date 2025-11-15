"""Обучение LoRA для персоны."""

import json
import logging
from pathlib import Path
import os
from typing import Optional, Dict
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoProcessor
from diffusers import DDPMScheduler
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from tqdm import tqdm
from .qwen_loader import load_qwen_components

logger = logging.getLogger(__name__)


def _to_number(value, default=None, number_type=float):
    """Преобразовать значение в число, если это строка."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return number_type(value)
    if isinstance(value, str):
        try:
            return number_type(value)
        except ValueError:
            return default
    return value


class PersonDataset(Dataset):
    """Датасет для обучения LoRA."""
    
    def __init__(self, dataset_dir: Path, tokenizer, processor, resolution: int = 768):
        """
        Инициализировать датасет.
        
        Args:
            dataset_dir: Директория с данными
            tokenizer: Токенайзер
            processor: Процессор изображений
            resolution: Разрешение изображений
        """
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        self.processor = processor
        self.resolution = resolution
        
        # Загрузить метаданные
        metadata_file = self.dataset_dir / "metadata.jsonl"
        self.items = []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                image_path = self.dataset_dir / "images" / item['file_name']
                if image_path.exists():
                    self.items.append({
                        'image_path': image_path,
                        'caption': item['caption']
                    })
        
        logger.info(f"Загружено {len(self.items)} изображений из датасета")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Загрузить изображение
        from PIL import Image
        import torch
        import torchvision.transforms as transforms
        
        image = Image.open(item['image_path']).convert('RGB')
        
        # Обработать изображение
        # Для Qwen Image processor может быть VaeImageProcessor, который нужно использовать правильно
        try:
            # Попробовать использовать processor напрямую (может быть callable)
            if callable(self.processor):
                try:
                    # Попробовать с return_tensors
                    processed = self.processor(image, return_tensors="pt")
                    if isinstance(processed, dict):
                        image = processed.get('pixel_values', processed.get('images', None))
                        if image is not None:
                            image = image[0] if len(image.shape) > 3 else image
                    elif isinstance(processed, (list, tuple)):
                        image = processed[0]
                    else:
                        image = processed
                except TypeError:
                    # Если не поддерживает return_tensors, попробовать без него
                    try:
                        processed = self.processor(image)
                        if isinstance(processed, dict):
                            image = processed.get('pixel_values', processed.get('images', None))
                            if image is not None:
                                image = image[0] if len(image.shape) > 3 else image
                        elif isinstance(processed, (list, tuple)):
                            image = processed[0]
                        else:
                            image = processed
                        # Преобразовать в tensor если нужно
                        if not isinstance(image, torch.Tensor):
                            import numpy as np
                            if isinstance(image, np.ndarray):
                                image = torch.from_numpy(image)
                            else:
                                image = torch.tensor(image)
                    except Exception as e:
                        raise e
            elif hasattr(self.processor, 'preprocess'):
                # Попробовать метод preprocess без return_tensors
                try:
                    processed = self.processor.preprocess(image)
                    if isinstance(processed, dict):
                        image = processed.get('pixel_values', processed.get('images', None))
                        if image is not None:
                            image = image[0] if len(image.shape) > 3 else image
                    elif isinstance(processed, (list, tuple)):
                        image = processed[0]
                    else:
                        image = processed
                    # Преобразовать в tensor если нужно
                    if not isinstance(image, torch.Tensor):
                        import numpy as np
                        if isinstance(image, np.ndarray):
                            image = torch.from_numpy(image)
                        else:
                            image = torch.tensor(image)
                except Exception:
                    raise
            else:
                raise AttributeError("Processor не имеет метода preprocess или __call__")
        except Exception as e:
            # Fallback: использовать стандартную обработку через torchvision
            logger.debug(f"Ошибка при использовании processor: {e}. Используется fallback.")
            transform = transforms.Compose([
                transforms.Resize((self.resolution, self.resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
            ])
            image = transform(image)
        
        # Убедиться, что это tensor правильной формы
        if not isinstance(image, torch.Tensor):
            import numpy as np
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            else:
                image = torch.tensor(image)
        
        # Убедиться, что изображение в правильном формате [C, H, W]
        if len(image.shape) == 4:
            image = image[0]  # Убрать batch dimension если есть
        if len(image.shape) == 2:
            image = image.unsqueeze(0)  # Добавить channel dimension если нужно
        
        # Обработать текст
        caption = item['caption']
        # Определить max_length из модели или использовать значение по умолчанию
        max_length = getattr(self.tokenizer, 'model_max_length', 77)
        text_inputs = self.tokenizer(
            caption,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'pixel_values': image,
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze()
        }


class LoRATrainer:
    """Тренер для обучения LoRA."""
    
    def __init__(self, config: dict, model_config: dict, base_model_path: str, dataset_dir: Path, output_dir: Path, paths: Optional[Dict[str, Path]] = None):
        """
        Инициализировать тренер.
        
        Args:
            config: Конфигурация обучения
            base_model_path: Путь к базовой модели
            dataset_dir: Директория с датасетом
            output_dir: Директория для сохранения результатов
        """
        self.config = config
        self.model_config = model_config or {}
        self.base_model_path = base_model_path
        self.model_id = self.model_config.get('base_model_id') or self.base_model_path
        self.model_backend = self.model_config.get('backend', 'flux').lower()
        self.paths = paths or {}
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hardware = config.get('hardware', {})
        self.max_cpu_ram_gb = self.hardware.get('max_cpu_ram_gb', 40)
        self.max_vram_gb = self.hardware.get('max_vram_gb', 24)
        self.dataloader_workers = max(0, int(self.hardware.get('dataloader_workers', 2)))
        self.prefetch_factor = max(1, int(self.hardware.get('prefetch_factor', 1)))
        self.max_batch_size = self.hardware.get('max_batch_size')
        self.offload_state_dict = self.hardware.get('offload_state_dict', True)
        
        accelerator_kwargs = {
            "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 4),
            "mixed_precision": config.get('mixed_precision', 'bf16')
        }
        if not torch.cuda.is_available():
            accelerator_kwargs["cpu"] = True
            accelerator_kwargs["mixed_precision"] = "no"
        self.accelerator = Accelerator(**accelerator_kwargs)
        self.device = self.accelerator.device
        logger.info("Инициализирован тренер LoRA")
        logger.info(f"  Mixed precision: {accelerator_kwargs['mixed_precision']}")
        logger.info(f"  Gradient accumulation: {accelerator_kwargs['gradient_accumulation_steps']}")
    
    def setup_model(self):
        """Настроить модель и LoRA."""
        logger.info("Загрузка базовой модели...")
        
        # Загрузить pipeline (упрощённая версия, в реальности нужна поддержка Qwen-Image)
        # ВАЖНО: Здесь нужно использовать правильную модель Qwen-Image 20B
        # Пока используем заглушку для структуры кода
        
        # Для RTX 5060 Ti лучше использовать float16 вместо bfloat16 для лучшей совместимости
        mixed_precision = self.config.get('mixed_precision', 'bf16')
        if mixed_precision == 'bf16':
            # Попробовать bfloat16, но если не работает - использовать float16
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        

        # Загрузить токенайзер из датасета (если он уже расширен)
        tokenizer_path = self.dataset_dir / "tokenizer"
        tokenizer_override = tokenizer_path if tokenizer_path.exists() else None

        model_dtype = dtype
        if dtype == torch.bfloat16:
            try:
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA недоступна")
                capability = torch.cuda.get_device_capability()
                if capability[0] < 8:
                    raise RuntimeError("GPU не поддерживает bfloat16")
            except Exception as exc:
                logger.info("bfloat16 не поддерживается (%s), используем float16", exc)
                model_dtype = torch.float16

        model_id = self.model_id
        pipeline = None

        if self.model_backend == "flux":
            from .flux_loader import load_flux_components
            local_flux_dir = None
            base_model_dir = self.paths.get('base_model_dir')
            if base_model_dir:
                local_flux_dir = Path(base_model_dir) / "flux"
            pipeline = load_flux_components(
                model_id,
                model_dtype,
                tokenizer_path=tokenizer_override,
                local_dir=local_flux_dir,
                revision=self.model_config.get('revision'),
                variant=self.model_config.get('variant'),
            )
            logger.info("✓ Компоненты FLUX.1-dev загружены")
        else:
            pipeline = load_qwen_components(
                model_id,
                model_dtype,
                tokenizer_path=tokenizer_override,
                max_cpu_ram_gb=self.max_cpu_ram_gb,
                max_vram_gb=self.max_vram_gb,
                device="cuda" if torch.cuda.is_available() else "cpu",
                offload_state_dict=self.offload_state_dict,
            )
            logger.info("✓ Компоненты Qwen-Image загружены")

        tokenizer = pipeline.tokenizer
        primary_dtype = getattr(pipeline, "dtype", model_dtype)
        pipeline_type = f"{self.model_backend}:{type(pipeline).__name__}"
        trainable_attr = "transformer" if getattr(pipeline, "transformer", None) is not None else "unet"
        logger.info(f"Trainable компонент: {trainable_attr}")

        # Проверить доступные компоненты более надёжным способом
        has_transformer = False
        has_unet = False
        
        try:
            _ = pipeline.transformer
            has_transformer = True
            logger.info("✓ Найден компонент: transformer")
        except (AttributeError, KeyError):
            pass
        
        try:
            _ = pipeline.unet
            has_unet = True
            logger.info("✓ Найден компонент: unet")
        except (AttributeError, KeyError):
            pass
        
        # Настроить LoRA
        # В Qwen-Image используется transformer вместо unet
        if has_transformer:
            # Qwen-Image использует transformer
            target_modules = self.config.get('target_modules', ['to_q', 'to_k', 'to_v', 'to_out.0'])
            lora_config = LoraConfig(
                r=_to_number(self.config.get('lora_rank', 16), 16, int),
                lora_alpha=_to_number(self.config.get('lora_alpha', 32), 32, int),
                target_modules=target_modules,
                lora_dropout=_to_number(self.config.get('lora_dropout', 0.1), 0.1, float),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            transformer = get_peft_model(pipeline.transformer, lora_config)
            pipeline.transformer = transformer
            logger.info("✓ LoRA применена к transformer (Qwen-Image)")
        elif has_unet:
            # Стандартные модели используют unet
            lora_config = LoraConfig(
                r=_to_number(self.config.get('lora_rank', 16), 16, int),
                lora_alpha=_to_number(self.config.get('lora_alpha', 32), 32, int),
                target_modules=self.config.get('target_modules', []),
                lora_dropout=_to_number(self.config.get('lora_dropout', 0.1), 0.1, float),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            unet = get_peft_model(pipeline.unet, lora_config)
            pipeline.unet = unet
            logger.info("✓ LoRA применена к unet")
        else:
            # Вывести доступные атрибуты для отладки
            available_attrs = [attr for attr in dir(pipeline) if not attr.startswith('_')]
            logger.error(f"Доступные атрибуты pipeline: {available_attrs[:20]}...")  # Первые 20
            raise ValueError(f"Не найден компонент для LoRA (ни transformer, ни unet). Тип pipeline: {pipeline_type}")
        
        # Если нужно, применить LoRA к текстовому энкодеру
        if self.config.get('text_encoder_target_modules') and hasattr(pipeline, 'text_encoder'):
            text_encoder_lora_config = LoraConfig(
                r=_to_number(self.config.get('lora_rank', 16), 16, int),
                lora_alpha=_to_number(self.config.get('lora_alpha', 32), 32, int),
                target_modules=self.config.get('text_encoder_target_modules', []),
                lora_dropout=_to_number(self.config.get('lora_dropout', 0.1), 0.1, float),
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION
            )
            text_encoder = get_peft_model(pipeline.text_encoder, text_encoder_lora_config)
            pipeline.text_encoder = text_encoder
            logger.info("✓ LoRA применена к text_encoder")

            try:
                peft_model = text_encoder
                base_model = getattr(peft_model, "base_model", None)
                if base_model is None and hasattr(peft_model, "model"):
                    base_model = peft_model.model
                if base_model is None:
                    base_model = peft_model

                orig_clip_forward = base_model.forward

                def clip_forward_no_inputs_embeds(*args, **kwargs):
                    if "inputs_embeds" in kwargs:
                        logger.warning("Убрал лишний kwargs inputs_embeds перед CLIPTextModel.forward")
                        kwargs.pop("inputs_embeds", None)
                    return orig_clip_forward(*args, **kwargs)

                base_model.forward = clip_forward_no_inputs_embeds
            except Exception as e:
                logger.warning(f"Не удалось запатчить base_model.forward: {e}")
        
        return pipeline, tokenizer, model_dtype
    
    def train(self):
        """Запустить обучение."""
        logger.info("Начало обучения LoRA...")
        
        # Настроить модель
        pipeline, tokenizer, primary_dtype = self.setup_model()
        
        # Получить processor из pipeline
        processor = None
        processor_type = None
        
        if hasattr(pipeline, 'image_processor'):
            processor = pipeline.image_processor
            processor_type = type(processor).__name__
        elif hasattr(pipeline, 'processor'):
            processor = pipeline.processor
            processor_type = type(processor).__name__
        else:
            # Попробовать загрузить processor отдельно
            try:
                processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
                processor_type = type(processor).__name__
            except Exception as e:
                logger.warning(f"Не удалось загрузить processor: {e}")
                logger.info("Использование стандартной обработки изображений через torchvision")
                processor = None  # Будем использовать fallback
        
        if processor:
            logger.info(f"✓ Processor загружен: {processor_type}")
        else:
            logger.info("✓ Используется стандартная обработка изображений")
        
        # Настроить scheduler (остаётся на CPU, не занимает много памяти)
        if getattr(pipeline, "scheduler", None) is not None:
            scheduler = pipeline.scheduler
            logger.info("✓ Scheduler получен из компонентов модели")
        else:
            scheduler = DDPMScheduler.from_pretrained(
                self.model_id,
                subfolder="scheduler",
                trust_remote_code=True
            )
            logger.info("✓ Scheduler загружен (на CPU)")
        
        # Получить trainable модель (transformer или unet)
        trainable_model = None
        if hasattr(pipeline, 'transformer'):
            trainable_model = pipeline.transformer
        elif hasattr(pipeline, 'unet'):
            trainable_model = pipeline.unet
        else:
            raise ValueError("Не найден trainable компонент модели")
        
        # Включить gradient checkpointing если нужно
        if self.config.get('gradient_checkpointing', False):
            if hasattr(trainable_model, 'enable_gradient_checkpointing'):
                trainable_model.enable_gradient_checkpointing()
                logger.info("✓ Gradient checkpointing включен")
        
        # Загрузить датасет
        resolution = _to_number(self.config.get('resolution', 768), 768, int)
        requested_batch = _to_number(self.config.get('batch_size', 1), 1, int)
        batch_size = requested_batch
        if self.max_batch_size and batch_size > self.max_batch_size:
            logger.warning(
                "Batch size %s превышает лимит %s — уменьшено автоматически", requested_batch, self.max_batch_size
            )
            batch_size = self.max_batch_size
        elif self.max_vram_gb and self.max_vram_gb <= 24 and batch_size > 1:
            logger.info("Профиль 24GB VRAM: batch_size=1 для стабильности")
            batch_size = 1
        
        dataset = PersonDataset(self.dataset_dir, tokenizer, processor, resolution)
        worker_cap = max(0, min(self.dataloader_workers, max((os.cpu_count() or 1) - 1, 0)))
        pin_memory = torch.cuda.is_available()
        dataloader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=worker_cap,
            pin_memory=pin_memory,
            persistent_workers=worker_cap > 0,
        )
        if worker_cap > 0:
            dataloader_kwargs["prefetch_factor"] = self.prefetch_factor
        dataloader = DataLoader(**dataloader_kwargs)
        logger.info(f"✓ Датасет загружен: {len(dataset)} изображений")
        
        # Настроить оптимизатор только для trainable параметров
        trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"✓ Trainable параметров: {num_trainable:,}")
        
        if num_trainable == 0:
            raise ValueError("Нет trainable параметров! Проверьте конфигурацию LoRA.")
        
        # Преобразовать learning_rate в float (может быть строкой из YAML)
        learning_rate = _to_number(self.config.get('learning_rate', 5e-5), 5e-5, float)
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        logger.info(f"✓ Оптимизатор настроен (lr={learning_rate})")
        
        # Настроить lr scheduler
        max_steps = _to_number(self.config.get('max_steps', 4000), 4000, int)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=1e-6
        )
        
        # Подготовить для accelerator
        trainable_model, optimizer, dataloader, lr_scheduler = self.accelerator.prepare(
            trainable_model, optimizer, dataloader, lr_scheduler
        )
        
        # Получить trainable_params после prepare (они могут измениться)
        trainable_params = [p for p in trainable_model.parameters() if p.requires_grad]
        
        # Получить device
        device = self.accelerator.device
        
        # Проверить совместимость CUDA
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                compute_capability = torch.cuda.get_device_capability(0)
                compute_capability_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
                
                # Проверить, поддерживается ли архитектура
                if compute_capability[0] >= 12:
                    logger.warning(f"⚠️  Обнаружена GPU: {gpu_name} с архитектурой {compute_capability_str}")
                    logger.warning("⚠️  Эта архитектура может не поддерживаться текущей версией PyTorch.")
                    logger.warning("⚠️  Если возникнут ошибки CUDA, будет использован CPU fallback.")
                    logger.warning("⚠️  Для полной поддержки рекомендуется обновить PyTorch:")
                    logger.warning("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
                
                torch.cuda.empty_cache()
                logger.info(f"✓ CUDA кэш очищен. Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            except Exception as e:
                logger.warning(f"Не удалось получить информацию о GPU: {e}")
        
        # Получить text_encoder и vae для forward pass
        # НЕ перемещаем их на GPU сразу - будем перемещать только когда нужно
        text_encoder = pipeline.text_encoder
        vae = pipeline.vae
        
        # Установить в eval mode для inference
        text_encoder.eval()
        vae.eval()
        
        # Заморозить параметры
        for param in text_encoder.parameters():
            param.requires_grad = False
        for param in vae.parameters():
            param.requires_grad = False
        
        # Сохранить оригинальный dtype моделей для восстановления
        try:
            vae_dtype = next(vae.parameters()).dtype if len(list(vae.parameters())) > 0 else primary_dtype
        except Exception:
            vae_dtype = primary_dtype
        
        try:
            text_encoder_dtype = next(text_encoder.parameters()).dtype if len(list(text_encoder.parameters())) > 0 else primary_dtype
        except Exception:
            text_encoder_dtype = primary_dtype
        
        # Переместить text_encoder и vae на CPU для экономии памяти
        # Будем перемещать их на GPU только во время forward pass
        text_encoder = text_encoder.to("cpu")
        vae = vae.to("cpu")
        
        logger.info("✓ Модели подготовлены для обучения")
        logger.info("  - Transformer: на GPU (trainable)")
        logger.info("  - Text encoder: на CPU (будет перемещён при необходимости)")
        logger.info("  - VAE: на CPU (будет перемещён при необходимости)")
        
        # Training loop
        global_step = 0
        save_steps = _to_number(self.config.get('save_steps', 500), 500, int)
        logging_steps = _to_number(self.config.get('logging_steps', 50), 50, int)
        gradient_accumulation_steps = _to_number(self.config.get('gradient_accumulation_steps', 4), 4, int)
        
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        logger.info(f"Начало обучения на {max_steps} шагов...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {gradient_accumulation_steps}")
        logger.info(f"  Effective batch size: {effective_batch_size}")
        
        progress_bar = tqdm(
            total=max_steps,
            disable=not self.accelerator.is_local_main_process,
            desc="Training"
        )
        
        trainable_model.train()
        
        while global_step < max_steps:
            for batch in dataloader:
                if global_step >= max_steps:
                    break
                
                # Переместить batch на устройство
                pixel_values = batch['pixel_values']
                # Приводим к формату BCHW
                if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
                    pixel_values = pixel_values[:, 0]
                elif pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                elif pixel_values.ndim != 4:
                    raise ValueError(f"Unexpected pixel_values shape {pixel_values.shape}, expected BCHW")
                pixel_values = pixel_values.to(device, dtype=primary_dtype)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Encode изображения в latent space через VAE
                # Переместить VAE на GPU только для forward pass
                with torch.no_grad():
                    # Попробовать использовать GPU сначала
                    vae_device = device
                    use_cpu_fallback = False
                    
                    try:
                        # Переместить VAE на GPU
                        vae = vae.to(vae_device)
                        # Убедиться, что VAE в правильном dtype для GPU
                        # Если bfloat16 не работает, попробуем float16
                        try:
                            if primary_dtype == torch.bfloat16:
                                # Проверить, поддерживается ли bfloat16 на этой GPU
                                test_tensor = torch.randn(1, 1, device=vae_device, dtype=torch.bfloat16)
                                del test_tensor
                        except Exception:
                            # bfloat16 не поддерживается, используем float16
                            logger.debug("bfloat16 не поддерживается на GPU, используем float16 для VAE")
                            primary_dtype = torch.float16
                            vae = vae.to(dtype=torch.float16)
                        
                        # VAE encoder
                        latents = vae.encode(pixel_values).latent_dist.sample()
                    except RuntimeError as e:
                        error_str = str(e)
                        if "CUDA" in error_str or "kernel" in error_str.lower() or "no kernel" in error_str.lower():
                            use_cpu_fallback = True
                            cuda_error = e  # Сохранить ошибку для логирования
                        else:
                            raise
                    
                    if use_cpu_fallback:
                            # CUDA ошибка - использовать CPU
                            logger.warning(f"CUDA ошибка при использовании VAE: {cuda_error}")
                            logger.warning("⚠️  Ваша GPU (RTX 5060 Ti) имеет архитектуру sm_120, которая не поддерживается текущей версией PyTorch.")
                            logger.warning("⚠️  Используется CPU для VAE (будет медленнее). Рекомендуется обновить PyTorch:")
                            logger.warning("   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124")
                            vae_device = "cpu"
                            # Переместить VAE на CPU и преобразовать в float32 для совместимости
                            vae = vae.to(vae_device)
                            
                            # Преобразовать VAE в float32 для CPU (bfloat16 может не поддерживаться на CPU)
                            # Сохранить оригинальный dtype для восстановления
                            try:
                                original_vae_dtype = next(vae.parameters()).dtype if len(list(vae.parameters())) > 0 else torch.float32
                            except Exception:
                                original_vae_dtype = torch.float32
                            
                            # Преобразовать модель в float32
                            try:
                                # Использовать to() с dtype для преобразования всей модели
                                vae = vae.to(dtype=torch.float32)
                            except Exception as dtype_error:
                                logger.debug(f"Не удалось преобразовать VAE в float32: {dtype_error}")
                                # Если не удалось, попробуем преобразовать параметры вручную
                                try:
                                    for param in vae.parameters():
                                        if param.dtype == torch.bfloat16:
                                            param.data = param.data.to(torch.float32)
                                except Exception:
                                    pass
                            
                            # Преобразовать входные данные в float32 для совместимости
                            pixel_values_cpu = pixel_values.to(vae_device)
                            # Убедиться, что dtype совпадает с моделью
                            if pixel_values_cpu.dtype != torch.float32:
                                pixel_values_cpu = pixel_values_cpu.to(torch.float32)
                            
                            # Выполнить encode
                            latents = vae.encode(pixel_values_cpu).latent_dist.sample()
                            
                            # Вернуть latents на GPU
                            latents = latents.to(device)
                            
                            # Восстановить оригинальный dtype VAE если нужно (для экономии памяти)
                            # Но только если это не bfloat16 (может не работать на CPU)
                            try:
                                if original_vae_dtype == torch.bfloat16:
                                    # Оставить в float32 для CPU
                                    pass
                                elif original_vae_dtype != torch.float32:
                                    vae = vae.to(dtype=original_vae_dtype)
                            except Exception:
                                pass
                    
                    # Убрать dimension num_frames если он был добавлен
                    if len(latents.shape) == 5:
                        # [batch, channels, num_frames, height, width] -> [batch, channels, height, width]
                        latents = latents.squeeze(2)
                    
                    # Применить scaling factor если он есть в конфиге
                    if hasattr(vae.config, 'scaling_factor'):
                        latents = latents * vae.config.scaling_factor
                    elif hasattr(vae, 'scaling_factor'):
                        latents = latents * vae.scaling_factor
                    
                    # Вернуть VAE на CPU
                    vae = vae.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Очистить кэш после VAE
                
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long
                )
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
                
                # Encode text
                # Переместить text_encoder на GPU только для forward pass
                with torch.no_grad():
                    text_encoder_device = device
                    try:
                        text_encoder = text_encoder.to(text_encoder_device)
                        encoder_hidden_states = text_encoder(
                            input_ids,
                            attention_mask=attention_mask
                        )[0]
                    except RuntimeError as e:
                        if "CUDA" in str(e) or "kernel" in str(e).lower():
                            # CUDA ошибка - использовать CPU
                            logger.warning(f"CUDA ошибка при использовании text_encoder: {e}")
                            text_encoder_device = "cpu"
                            text_encoder = text_encoder.to(text_encoder_device)
                            # Преобразовать text_encoder в float32 для CPU
                            if hasattr(text_encoder, 'to'):
                                try:
                                    text_encoder = text_encoder.to(dtype=torch.float32)
                                except Exception:
                                    pass
                            input_ids_cpu = input_ids.to(text_encoder_device)
                            attention_mask_cpu = attention_mask.to(text_encoder_device)
                            encoder_hidden_states = text_encoder(
                                input_ids_cpu,
                                attention_mask=attention_mask_cpu
                            )[0]
                            encoder_hidden_states = encoder_hidden_states.to(device)  # Вернуть на GPU
                        else:
                            raise
                    
                    # Вернуть text_encoder на CPU
                    text_encoder = text_encoder.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # Очистить кэш после text_encoder
                
                # Forward pass через transformer/unet
                with self.accelerator.accumulate(trainable_model):
                    # Predict noise
                    # Для Qwen Image transformer используем стандартный API
                    try:
                        # Попробовать стандартный API для transformer
                        if hasattr(trainable_model, 'forward'):
                            # Qwen Image transformer
                            model_pred = trainable_model(
                                sample=noisy_latents,
                                timestep=timesteps,
                                encoder_hidden_states=encoder_hidden_states,
                                return_dict=False
                            )
                            # Может вернуть tuple или tensor
                            if isinstance(model_pred, tuple):
                                model_pred = model_pred[0]
                        else:
                            # Fallback для unet
                            model_pred = trainable_model(
                                noisy_latents,
                                timesteps,
                                encoder_hidden_states=encoder_hidden_states
                            )
                            if hasattr(model_pred, 'sample'):
                                model_pred = model_pred.sample
                    except Exception as e:
                        logger.error(f"Ошибка в forward pass: {e}")
                        raise
                    
                    # Calculate loss
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(trainable_params, 1.0)
                    
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Logging
                if global_step % logging_steps == 0:
                    logs = {
                        "loss": loss.detach().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "step": global_step
                    }
                    progress_bar.set_postfix(**logs)
                    logger.info(f"Step {global_step}: loss={loss.item():.4f}, lr={lr_scheduler.get_last_lr()[0]:.2e}")
                
                # Save checkpoint
                if global_step % save_steps == 0 and global_step > 0:
                    self._save_checkpoint(pipeline, global_step)
                
                global_step += 1
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Final save
        logger.info("Сохранение финальной модели...")
        self._save_checkpoint(pipeline, global_step, is_final=True)
        
        logger.info("✅ Обучение завершено!")
        
        # Сохранить манифест обучения
        manifest = {
            'training_date': datetime.now().isoformat(),
            'config': self.config,
            'base_model': self.model_id,
            'dataset_dir': str(self.dataset_dir),
            'status': 'completed',
            'total_steps': global_step,
            'final_loss': loss.item() if 'loss' in locals() else None
        }
        
        manifest_file = self.output_dir / "training_manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Манифест обучения сохранён: {manifest_file}")
        
        return manifest_file
    
    def _save_checkpoint(self, pipeline, step: int, is_final: bool = False):
        """Сохранить чекпоинт LoRA."""
        if is_final:
            checkpoint_dir = self.output_dir / "checkpoint-final"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Сохранить LoRA веса
        if hasattr(pipeline, 'transformer') and hasattr(pipeline.transformer, 'save_pretrained'):
            pipeline.transformer.save_pretrained(checkpoint_dir / "transformer")
        
        if hasattr(pipeline, 'text_encoder') and hasattr(pipeline.text_encoder, 'save_pretrained'):
            # Сохранить только LoRA веса text_encoder если они есть
            try:
                pipeline.text_encoder.save_pretrained(checkpoint_dir / "text_encoder")
            except Exception as e:
                logger.warning(f"Не удалось сохранить text_encoder: {e}")
        
        logger.info(f"✓ Чекпоинт сохранён: {checkpoint_dir}")
