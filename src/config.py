"""Управление конфигурацией системы."""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Класс для загрузки и доступа к конфигурации."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Загрузить конфигурацию из файла."""
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузить YAML конфигурацию."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Получить значение по ключу (поддерживает вложенные ключи через точку)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def get_path(self, key: str) -> Path:
        """Получить путь из конфигурации."""
        path_str = self.get(key)
        if path_str:
            return Path(path_str)
        return None
    
    @property
    def paths(self) -> Dict[str, Path]:
        """Получить все пути из конфигурации."""
        paths_config = self._config.get('paths', {})
        return {k: Path(v) for k, v in paths_config.items()}
    
    @property
    def training(self) -> Dict[str, Any]:
        """Получить параметры обучения."""
        return self._config.get('training', {})
    
    @property
    def inference(self) -> Dict[str, Any]:
        """Получить параметры инференса."""
        return self._config.get('inference', {})
    
    @property
    def quality_control(self) -> Dict[str, Any]:
        """Получить параметры контроля качества."""
        return self._config.get('quality_control', {})

    @property
    def hardware(self) -> Dict[str, Any]:
        """Получить ограничения железа."""
        return self._config.get('hardware', {})

    @property
    def model(self) -> Dict[str, Any]:
        """Получить настройки модели/бэкенда."""
        model_cfg = self._config.get('model', {})
        if not model_cfg:
            # Фолбэк на старый формат
            legacy = self._config.get('base_model', {})
            return {
                'backend': 'qwen',
                'base_model_id': legacy.get('name', 'Qwen/Qwen-Image'),
                'dtype': legacy.get('dtype', 'bf16')
            }
        return model_cfg


# Глобальный экземпляр конфигурации
_config_instance = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Получить глобальный экземпляр конфигурации."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance

