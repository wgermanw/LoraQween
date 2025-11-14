"""
Ноды ComfyUI для LoraQween.

ВНИМАНИЕ: Это базовая структура. Для полной работы требуется:
1. Установка ComfyUI
2. Правильная интеграция с API ComfyUI
3. Поддержка Qwen-Image модели в ComfyUI
"""

import os
import sys
import json
from pathlib import Path

# Добавить путь к проекту
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.tokenizer.token_manager import TokenManager


class LoadLoraQweenLoRA:
    """Нода для загрузки LoRA персоны."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_name": ("STRING", {"default": ""}),
                "lora_scale": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }
    
    RETURN_TYPES = ("LORA",)
    FUNCTION = "load_lora"
    CATEGORY = "LoraQween"
    
    def load_lora(self, person_name: str, lora_scale: float):
        """Загрузить LoRA для персоны."""
        config = get_config()
        lora_path = config.paths['loras_dir'] / person_name
        
        if not lora_path.exists():
            raise ValueError(f"LoRA для персоны '{person_name}' не найдена: {lora_path}")
        
        # TODO: Интеграция с ComfyUI API для загрузки LoRA
        # return (lora_object,)
        return (None,)


class LoadLoraQweenTokenizer:
    """Нода для загрузки токенайзера с триггер-токенами."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "person_name": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("TOKENIZER",)
    FUNCTION = "load_tokenizer"
    CATEGORY = "LoraQween"
    
    def load_tokenizer(self, person_name: str):
        """Загрузить токенайзер с триггер-токенами."""
        config = get_config()
        dataset_dir = config.paths['datasets_dir'] / person_name
        
        tokenizer_path = dataset_dir / "tokenizer"
        if not tokenizer_path.exists():
            raise ValueError(f"Токенайзер для персоны '{person_name}' не найден: {tokenizer_path}")
        
        # Runtime проверка токена
        token_manager = TokenManager(str(tokenizer_path), config.paths['datasets_dir'])
        manifest_file = dataset_dir / "manifest.json"
        
        if manifest_file.exists():
            import json
            with open(manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
                trigger_token = manifest.get('trigger_token')
                if trigger_token:
                    is_valid, token_ids = token_manager.validate_runtime_token(trigger_token)
                    if not is_valid:
                        raise RuntimeError(
                            f"RUNTIME CHECK FAILED: Токен '{trigger_token}' распался на части!\n"
                            f"Используйте tokenizer.json из набора обучения."
                        )
        
        # TODO: Интеграция с ComfyUI API для загрузки токенайзера
        # return (tokenizer_object,)
        return (None,)


class LoraQweenFaceID:
    """Нода для применения FaceID/IP-Adapter."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "face_id_weight": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "apply_to_face_mask": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("FACEID_ADAPTER",)
    FUNCTION = "apply_faceid"
    CATEGORY = "LoraQween"
    
    def apply_faceid(self, reference_image, face_id_weight: float, apply_to_face_mask: bool):
        """Применить FaceID/IP-Adapter."""
        # TODO: Интеграция с FaceID/IP-Adapter в ComfyUI
        return (None,)


class LoraQweenQualityCheck:
    """Нода для проверки качества генерации."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generated_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("FLOAT", "BOOLEAN")
    RETURN_NAMES = ("similarity", "meets_threshold")
    FUNCTION = "check_quality"
    CATEGORY = "LoraQween"
    
    def check_quality(self, generated_image, reference_image):
        """Проверить качество генерации."""
        from src.quality.face_similarity import FaceSimilarity
        
        # TODO: Конвертировать изображения ComfyUI в формат для FaceSimilarity
        # similarity = face_similarity.compare_images(...)
        # meets_threshold = similarity >= threshold
        
        return (0.0, False)


# Регистрация нод (для ComfyUI)
NODE_CLASS_MAPPINGS = {
    "LoadLoraQweenLoRA": LoadLoraQweenLoRA,
    "LoadLoraQweenTokenizer": LoadLoraQweenTokenizer,
    "LoraQweenFaceID": LoraQweenFaceID,
    "LoraQweenQualityCheck": LoraQweenQualityCheck,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadLoraQweenLoRA": "Load LoraQween LoRA",
    "LoadLoraQweenTokenizer": "Load LoraQween Tokenizer",
    "LoraQweenFaceID": "LoraQween FaceID",
    "LoraQweenQualityCheck": "LoraQween Quality Check",
}

