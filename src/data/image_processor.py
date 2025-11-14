"""Обработка изображений для обучения."""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
from PIL import Image
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Процессор изображений для подготовки данных."""
    
    def __init__(self, target_resolution: int = 768):
        """
        Инициализировать процессор.
        
        Args:
            target_resolution: Целевое разрешение по короткой стороне
        """
        self.target_resolution = target_resolution
    
    def load_image(self, image_path: Path) -> Image.Image:
        """Загрузить изображение."""
        try:
            img = Image.open(image_path).convert('RGB')
            return img
        except Exception as e:
            logger.error(f"Ошибка загрузки изображения {image_path}: {e}")
            raise
    
    def resize_to_target(self, image: Image.Image, aspect_ratio: Optional[Tuple[int, int]] = None) -> Image.Image:
        """
        Изменить размер изображения до целевого разрешения.
        
        Args:
            image: Входное изображение
            aspect_ratio: Соотношение сторон (width, height), если None - сохранить пропорции
        
        Returns:
            Изменённое изображение
        """
        if aspect_ratio:
            # Изменить размер с учётом соотношения сторон
            width, height = aspect_ratio
            target_size = (width, height)
            return image.resize(target_size, Image.Resampling.LANCZOS)
        else:
            # Изменить размер по короткой стороне
            width, height = image.size
            if width < height:
                new_width = self.target_resolution
                new_height = int(height * (self.target_resolution / width))
            else:
                new_height = self.target_resolution
                new_width = int(width * (self.target_resolution / height))
            
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Обнаружить лицо на изображении.
        
        Returns:
            (x, y, width, height) или None
        """
        # Простая реализация через OpenCV Haar cascades
        # В продакшене лучше использовать более точные модели (MTCNN, RetinaFace)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Вернуть самое большое лицо
            largest_face = max(faces, key=lambda x: x[2] * x[3])
            return tuple(largest_face)
        
        return None
    
    def align_face(self, image: Image.Image) -> Image.Image:
        """
        Выровнять лицо на изображении.
        
        Args:
            image: Входное изображение
        
        Returns:
            Выровненное изображение
        """
        img_array = np.array(image)
        face_bbox = self.detect_face(img_array)
        
        if face_bbox is None:
            logger.warning("Лицо не обнаружено, возвращаю исходное изображение")
            return image
        
        x, y, w, h = face_bbox
        
        # Простое выравнивание: центрирование лица
        # В продакшене использовать более сложные алгоритмы (5-point alignment)
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Для простоты возвращаем исходное изображение
        # В реальной реализации здесь должна быть логика выравнивания
        return image
    
    def clear_exif(self, image_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Очистить EXIF данные из изображения.
        
        Args:
            image_path: Путь к исходному изображению
            output_path: Путь для сохранения (если None, перезаписать исходное)
        
        Returns:
            Путь к обработанному изображению
        """
        img = self.load_image(image_path)
        
        # Удалить EXIF
        data = list(img.getdata())
        image_without_exif = Image.new(img.mode, img.size)
        image_without_exif.putdata(data)
        
        if output_path is None:
            output_path = image_path
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image_without_exif.save(output_path, quality=95)
        
        logger.info(f"EXIF данные удалены из {image_path}")
        return output_path
    
    def create_buckets(self, images: List[Path]) -> dict:
        """
        Разложить изображения по бакетам разных соотношений сторон.
        
        Returns:
            Словарь {aspect_ratio: [image_paths]}
        """
        buckets = {
            (1, 1): [],      # Квадрат
            (3, 4): [],     # Портрет
            (4, 3): [],     # Альбом
            (2, 3): [],     # Вертикальный
            (3, 2): [],     # Горизонтальный
            (9, 16): [],    # Мобильный портрет
        }
        
        for img_path in images:
            img = self.load_image(img_path)
            width, height = img.size
            aspect = width / height
            
            # Найти ближайший бакет
            best_bucket = None
            min_diff = float('inf')
            
            for bucket_ratio, bucket_images in buckets.items():
                bucket_aspect = bucket_ratio[0] / bucket_ratio[1]
                diff = abs(aspect - bucket_aspect)
                if diff < min_diff:
                    min_diff = diff
                    best_bucket = bucket_ratio
            
            if best_bucket:
                buckets[best_bucket].append(img_path)
        
        # Удалить пустые бакеты
        return {k: v for k, v in buckets.items() if v}
    
    def apply_augmentation(self, image: Image.Image, augmentation_type: str = "light") -> Image.Image:
        """
        Применить аугментацию к изображению.
        
        Args:
            image: Входное изображение
            augmentation_type: Тип аугментации ("light", "medium", "heavy")
        
        Returns:
            Аугментированное изображение
        """
        if augmentation_type == "light":
            # Лёгкие аугментации: color jitter, небольшой шум
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            ])
            
            return transform(image)
        
        # Для других типов можно добавить больше аугментаций
        return image

