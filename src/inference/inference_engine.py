"""Движок инференса для генерации изображений."""

import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import json

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceEngine:
    """Движок для генерации изображений."""
    
    def __init__(self, config: dict, base_model_path: str, lora_path: Optional[Path] = None):
        """
        Инициализировать движок инференса.
        
        Args:
            config: Конфигурация инференса
            base_model_path: Путь к базовой модели
            lora_path: Путь к LoRA весам
        """
        self.config = config
        self.base_model_path = base_model_path
        self.lora_path = Path(lora_path) if lora_path else None
        
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Инициализирован движок инференса (device: {self.device})")
    
    def load_model(self):
        """Загрузить модель и LoRA."""
        logger.info("Загрузка модели...")
        
        # Загрузить Qwen-Image модель через diffusers
        import torch
        from diffusers import DiffusionPipeline
        
        dtype = torch.float16 if self.config.get('dtype') == 'fp16' else torch.bfloat16
        
        logger.info(f"Загрузка модели из: {self.base_model_path}")
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=dtype,
            trust_remote_code=True
        )
        
        logger.info("✓ Модель загружена")
        
        # Загрузить LoRA если указана
        if self.lora_path:
            logger.info(f"Загрузка LoRA из: {self.lora_path}")
            self.pipeline.load_lora_weights(str(self.lora_path))
            logger.info("✓ LoRA загружена")
    
    def generate_fast(self, prompt: str, num_images: int = 4, seed: Optional[int] = None) -> List[Image.Image]:
        """
        Генерация в быстром режиме.
        
        Args:
            prompt: Промт с триггер-токеном
            num_images: Количество изображений
            seed: Seed для воспроизводимости
        
        Returns:
            Список сгенерированных изображений
        """
        fast_config = self.config.get('fast_mode', {})
        
        if self.pipeline is None:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")
        
        # Настроить генератор
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator.manual_seed(seed)
        
        # Параметры генерации
        num_inference_steps = fast_config.get('num_steps', 40)
        guidance_scale = fast_config.get('cfg_scale', 7.0)
        width = height = fast_config.get('resolution', 768)
        
        logger.info(f"Генерация в быстром режиме: {num_images} изображений")
        logger.info(f"  Промт: {prompt}")
        logger.info(f"  Шаги: {num_inference_steps}, CFG: {guidance_scale}, Разрешение: {width}x{height}")
        
        # Генерация
        # images = self.pipeline(
        #     prompt=prompt,
        #     num_images_per_prompt=num_images,
        #     num_inference_steps=num_inference_steps,
        #     guidance_scale=guidance_scale,
        #     width=width,
        #     height=height,
        #     generator=generator
        # ).images
        
        # Заглушка
        logger.warning("⚠️  Генерация не может быть выполнена без загрузки модели")
        images = []
        
        return images
    
    def generate_reliable(self, prompt: str, reference_image: Optional[Image.Image] = None,
                         num_images: int = 2, seed: Optional[int] = None) -> List[Image.Image]:
        """
        Генерация в надёжном режиме с FaceID/IP-Adapter.
        
        Args:
            prompt: Промт с триггер-токеном
            reference_image: Референсное изображение для FaceID/IP-Adapter
            num_images: Количество изображений
            seed: Seed для воспроизводимости
        
        Returns:
            Список сгенерированных изображений
        """
        reliable_config = self.config.get('reliable_mode', {})
        
        if self.pipeline is None:
            raise RuntimeError("Модель не загружена. Вызовите load_model() сначала.")
        
        logger.info(f"Генерация в надёжном режиме: {num_images} изображений")
        logger.info(f"  Промт: {prompt}")
        if reference_image:
            logger.info(f"  Используется референсное изображение")
        
        # Параметры надёжного режима
        num_inference_steps = reliable_config.get('num_steps', 60)
        guidance_scale = reliable_config.get('cfg_scale', 7.5)
        width = height = reliable_config.get('resolution', 1024)
        face_id_weight = reliable_config.get('face_id_weight', 0.6)
        
        # TODO: Применить FaceID/IP-Adapter
        # if reference_image:
        #     # Применить FaceID/IP-Adapter
        #     pass
        
        # TODO: Применить ControlNet если нужно
        # if reliable_config.get('use_controlnet'):
        #     pass
        
        logger.warning("⚠️  Генерация в надёжном режиме не может быть выполнена без загрузки модели")
        images = []
        
        return images
    
    def save_generation_log(self, prompt: str, images: List[Image.Image], mode: str,
                           seed: Optional[int], output_dir: Path):
        """Сохранить лог генерации."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохранить изображения
        images_dir = output_dir / f"generation_{timestamp}"
        images_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(images):
            img.save(images_dir / f"image_{i:03d}.png")
        
        # Сохранить метаданные
        log_data = {
            'timestamp': timestamp,
            'prompt': prompt,
            'mode': mode,
            'seed': seed,
            'num_images': len(images),
            'images_dir': str(images_dir)
        }
        
        log_file = output_dir / f"generation_{timestamp}_log.json"
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Лог генерации сохранён: {log_file}")
        return log_file

