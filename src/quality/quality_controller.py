"""Контроллер качества генерации."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from .face_similarity import FaceSimilarity

logger = logging.getLogger(__name__)


class QualityController:
    """Контроллер качества."""
    
    def __init__(self, config: dict):
        """
        Инициализировать контроллер качества.
        
        Args:
            config: Конфигурация контроля качества
        """
        self.config = config
        self.face_similarity = FaceSimilarity(model_name=config.get('face_id_model', 'buffalo_l'))
        
        self.similarity_threshold = config.get('similarity_threshold', 0.7)
        self.fpr_target = config.get('fpr_target', 0.01)
        self.tpr_target = config.get('tpr_target', 0.90)
    
    def evaluate_batch(self, generated_images: List[Path], reference_images: List[Path]) -> Dict:
        """
        Оценить качество батча сгенерированных изображений.
        
        Args:
            generated_images: Пути к сгенерированным изображениям
            reference_images: Пути к референсным изображениям
        
        Returns:
            Словарь с метриками
        """
        similarities = []
        
        # Сравнить каждое сгенерированное с референсом
        for gen_img, ref_img in zip(generated_images, reference_images):
            similarity = self.face_similarity.compare_images(gen_img, ref_img)
            if similarity is not None:
                similarities.append(similarity)
        
        if not similarities:
            logger.warning("Не удалось вычислить схожесть ни для одной пары")
            return {
                'mean_similarity': 0.0,
                'median_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
                'num_valid': 0,
                'below_threshold': len(generated_images)
            }
        
        similarities = np.array(similarities)
        
        metrics = {
            'mean_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'std_similarity': float(np.std(similarities)),
            'num_valid': len(similarities),
            'below_threshold': int(np.sum(similarities < self.similarity_threshold))
        }
        
        logger.info(f"Оценка качества: средняя схожесть = {metrics['mean_similarity']:.3f}")
        
        return metrics
    
    def compute_roc(self, positive_pairs: List[Tuple[Path, Path]], 
                   negative_pairs: List[Tuple[Path, Path]]) -> Dict:
        """
        Вычислить ROC кривую.
        
        Args:
            positive_pairs: Пары (сгенерированное, референс) для позитивных примеров
            negative_pairs: Пары для негативных примеров
        
        Returns:
            Словарь с ROC метриками
        """
        # Вычислить схожести для позитивных пар
        positive_similarities = []
        for gen_img, ref_img in positive_pairs:
            sim = self.face_similarity.compare_images(gen_img, ref_img)
            if sim is not None:
                positive_similarities.append(sim)
        
        # Вычислить схожести для негативных пар
        negative_similarities = []
        for gen_img, ref_img in negative_pairs:
            sim = self.face_similarity.compare_images(gen_img, ref_img)
            if sim is not None:
                negative_similarities.append(sim)
        
        if not positive_similarities or not negative_similarities:
            logger.warning("Недостаточно данных для построения ROC")
            return {}
        
        # Найти порог при целевом FPR
        negative_similarities = np.array(negative_similarities)
        threshold_idx = int(len(negative_similarities) * (1 - self.fpr_target))
        threshold = np.sort(negative_similarities)[threshold_idx] if threshold_idx < len(negative_similarities) else 1.0
        
        # Вычислить TPR при этом пороге
        positive_similarities = np.array(positive_similarities)
        tpr = np.mean(positive_similarities >= threshold)
        
        # Вычислить FPR
        fpr = np.mean(negative_similarities >= threshold)
        
        metrics = {
            'threshold': float(threshold),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'target_fpr': self.fpr_target,
            'target_tpr': self.tpr_target,
            'meets_target': tpr >= self.tpr_target
        }
        
        logger.info(f"ROC метрики: порог={threshold:.3f}, TPR={tpr:.3f}, FPR={fpr:.3f}")
        
        return metrics
    
    def should_switch_to_reliable_mode(self, metrics: Dict) -> bool:
        """
        Определить, нужно ли переключиться на надёжный режим.
        
        Args:
            metrics: Метрики качества
        
        Returns:
            True если нужно переключиться
        """
        mean_similarity = metrics.get('mean_similarity', 0.0)
        return mean_similarity < self.similarity_threshold

