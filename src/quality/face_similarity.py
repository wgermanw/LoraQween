"""Измерение схожести лиц с помощью Face-ID."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class FaceSimilarity:
    """Измерение схожести лиц."""
    
    def __init__(self, model_name: str = "buffalo_l"):
        """
        Инициализировать Face-ID модель.
        
        Args:
            model_name: Имя модели InsightFace
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загрузить Face-ID модель."""
        try:
            import insightface
            from insightface.app import FaceAnalysis
            
            self.model = FaceAnalysis(name=self.model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.model.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info(f"Face-ID модель '{self.model_name}' загружена")
        except ImportError:
            logger.error("InsightFace не установлен. Установите: pip install insightface")
            raise
        except Exception as e:
            logger.error(f"Ошибка загрузки Face-ID модели: {e}")
            raise
    
    def extract_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Извлечь эмбеддинг лица из изображения.
        
        Args:
            image_path: Путь к изображению
        
        Returns:
            Эмбеддинг лица или None
        """
        try:
            import cv2
            from PIL import Image
            
            # Загрузить изображение
            img = cv2.imread(str(image_path))
            if img is None:
                logger.warning(f"Не удалось загрузить изображение: {image_path}")
                return None
            
            # Обнаружить лица
            faces = self.model.get(img)
            
            if len(faces) == 0:
                logger.warning(f"Лица не обнаружены на изображении: {image_path}")
                return None
            
            # Взять первое лицо
            face = faces[0]
            embedding = face.embedding
            
            return embedding
            
        except Exception as e:
            logger.error(f"Ошибка извлечения эмбеддинга из {image_path}: {e}")
            return None
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Вычислить косинусную близость между эмбеддингами.
        
        Args:
            embedding1: Первый эмбеддинг
            embedding2: Второй эмбеддинг
        
        Returns:
            Косинусная близость (0-1)
        """
        # Нормализовать эмбеддинги
        emb1_norm = embedding1 / np.linalg.norm(embedding1)
        emb2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Вычислить косинусную близость
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    def compare_images(self, image1_path: Path, image2_path: Path) -> Optional[float]:
        """
        Сравнить два изображения.
        
        Args:
            image1_path: Путь к первому изображению
            image2_path: Путь ко второму изображению
        
        Returns:
            Косинусная близость или None
        """
        emb1 = self.extract_embedding(image1_path)
        emb2 = self.extract_embedding(image2_path)
        
        if emb1 is None or emb2 is None:
            return None
        
        return self.compute_similarity(emb1, emb2)

