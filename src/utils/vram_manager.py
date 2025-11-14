"""Управление VRAM и fallback при нехватке памяти."""

import logging
from typing import Dict, Optional
import torch

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml не установлен. Мониторинг VRAM будет ограничен.")

logger = logging.getLogger(__name__)


class VRAMManager:
    """Менеджер VRAM."""
    
    def __init__(self, config: dict):
        """
        Инициализировать менеджер VRAM.
        
        Args:
            config: Конфигурация VRAM
        """
        self.config = config
        self.monitor_interval = config.get('monitor_interval', 1.0)
        self.fallback_threshold = config.get('fallback_threshold', 0.95)
        self.degraded_profile = config.get('degraded_profile', {})
        
        if NVML_AVAILABLE:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.total_memory = pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
                logger.info(f"VRAM менеджер инициализирован. Всего памяти: {self.total_memory / 1024**3:.2f} GB")
            except Exception as e:
                logger.warning(f"Не удалось инициализировать NVML: {e}")
                self.handle = None
                self.total_memory = None
        else:
            self.handle = None
            self.total_memory = None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Получить текущее использование памяти.
        
        Returns:
            Словарь с информацией о памяти
        """
        if NVML_AVAILABLE and self.handle:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                return {
                    'total_gb': info.total / 1024**3,
                    'used_gb': info.used / 1024**3,
                    'free_gb': info.free / 1024**3,
                    'usage_ratio': info.used / info.total
                }
            except Exception as e:
                logger.warning(f"Ошибка получения информации о VRAM: {e}")
        
        # Fallback через PyTorch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'total_gb': total,
                'used_gb': allocated,
                'reserved_gb': reserved,
                'usage_ratio': reserved / total if total > 0 else 0.0
            }
        
        return {'total_gb': 0, 'used_gb': 0, 'free_gb': 0, 'usage_ratio': 0.0}
    
    def check_memory_available(self, required_gb: float) -> bool:
        """
        Проверить, достаточно ли свободной памяти.
        
        Args:
            required_gb: Требуемый объём памяти в GB
        
        Returns:
            True если памяти достаточно
        """
        usage = self.get_memory_usage()
        free_gb = usage.get('free_gb', 0)
        
        return free_gb >= required_gb
    
    def should_degrade(self) -> bool:
        """
        Определить, нужно ли переключиться на degraded профиль.
        
        Returns:
            True если нужно деградировать
        """
        usage = self.get_memory_usage()
        usage_ratio = usage.get('usage_ratio', 0.0)
        
        return usage_ratio >= self.fallback_threshold
    
    def get_degraded_config(self) -> Dict:
        """
        Получить конфигурацию для degraded режима.
        
        Returns:
            Конфигурация с пониженными параметрами
        """
        return self.degraded_profile.copy()
    
    def clear_cache(self):
        """Очистить кэш CUDA."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA кэш очищен")

