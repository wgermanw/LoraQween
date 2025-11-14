"""Подготовка датасета для обучения."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from PIL import Image

from .image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class DatasetPreparator:
    """Подготовка датасета для обучения LoRA."""
    
    def __init__(self, person_name: str, trigger_token: str, output_dir: Path, config: dict):
        """
        Инициализировать подготовщик датасета.
        
        Args:
            person_name: Имя персоны
            trigger_token: Триггер-токен
            output_dir: Директория для выходных данных
            config: Конфигурация
        """
        self.person_name = person_name
        self.trigger_token = trigger_token
        self.output_dir = Path(output_dir)
        self.config = config
        self.processor = ImageProcessor(target_resolution=config.get('resolution', 768))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_dataset(self, input_dir: Path, metadata: Optional[Dict] = None) -> Path:
        """
        Подготовить датасет из исходных изображений.
        
        Args:
            input_dir: Директория с исходными изображениями
            metadata: Дополнительные метаданные
        
        Returns:
            Путь к подготовленному датасету
        """
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise ValueError(f"Директория не найдена: {input_dir}")
        
        # Найти все изображения
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
        image_files = [f for f in input_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if len(image_files) < 10:
            logger.warning(f"Найдено только {len(image_files)} изображений. Рекомендуется 40-80.")
        
        logger.info(f"Найдено {len(image_files)} изображений для обработки")
        
        # Создать поддиректории
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Обработать изображения
        processed_images = []
        captions = []
        
        for img_path in image_files:
            try:
                # Загрузить и обработать
                img = self.processor.load_image(img_path)
                
                # Очистить EXIF
                output_img_path = images_dir / img_path.name
                self.processor.clear_exif(img_path, output_img_path)
                
                # Изменить размер
                img = self.processor.load_image(output_img_path)
                img_resized = self.processor.resize_to_target(img)
                img_resized.save(output_img_path, quality=95)
                
                # Создать подпись
                caption = self._generate_caption(img_path, metadata)
                captions.append({
                    'file_name': img_path.name,
                    'caption': caption
                })
                
                processed_images.append(output_img_path)
                
            except Exception as e:
                logger.error(f"Ошибка обработки {img_path}: {e}")
                continue
        
        # Сохранить метаданные
        metadata_file = self.output_dir / "metadata.jsonl"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for caption_data in captions:
                f.write(json.dumps(caption_data, ensure_ascii=False) + '\n')
        
        # Создать манифест
        manifest = {
            'person_name': self.person_name,
            'trigger_token': self.trigger_token,
            'num_images': len(processed_images),
            'resolution': self.config.get('resolution', 768),
            'metadata': metadata or {}
        }
        
        manifest_file = self.output_dir / "manifest.json"
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Датасет подготовлен: {len(processed_images)} изображений")
        logger.info(f"Манифест сохранён: {manifest_file}")
        
        return self.output_dir
    
    def _generate_caption(self, image_path: Path, metadata: Optional[Dict] = None) -> str:
        """
        Сгенерировать подпись для изображения.
        
        Args:
            image_path: Путь к изображению
            metadata: Метаданные персоны
        
        Returns:
            Подпись с триггер-токеном
        """
        # Базовая подпись с триггер-токеном
        caption_parts = [self.trigger_token]
        
        # Добавить базовые дескрипторы из метаданных
        if metadata:
            if 'gender' in metadata:
                caption_parts.append(metadata['gender'])
            if 'age' in metadata:
                caption_parts.append(f"age {metadata['age']}")
            if 'hair_color' in metadata:
                caption_parts.append(f"{metadata['hair_color']} hair")
        
        # Добавить базовые дескрипторы сцены
        caption_parts.append("portrait")
        
        return ", ".join(caption_parts)
    
    def create_buckets(self, images_dir: Path) -> Dict:
        """Создать бакеты для разных соотношений сторон."""
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        buckets = self.processor.create_buckets(image_files)
        
        # Сохранить информацию о бакетах
        buckets_info = {f"{w}x{h}": len(imgs) for (w, h), imgs in buckets.items()}
        
        buckets_file = self.output_dir / "buckets.json"
        with open(buckets_file, 'w', encoding='utf-8') as f:
            json.dump(buckets_info, f, indent=2)
        
        logger.info(f"Создано бакетов: {buckets_info}")
        return buckets

