"""Скрипт обучения LoRA."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.training.lora_trainer import LoRATrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Обучение LoRA для персоны")
    parser.add_argument("--person", type=str, required=True, help="Имя персоны")
    parser.add_argument("--trigger_token", type=str, default=None, help="Триггер-токен")
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = get_config()
    paths = config.paths
    
    # Найти датасет персоны
    dataset_dir = paths['datasets_dir'] / args.person
    if not dataset_dir.exists():
        logger.error(f"Датасет для персоны '{args.person}' не найден: {dataset_dir}")
        logger.error("Сначала запустите prepare_data.py")
        return 1
    
    # Директория для сохранения LoRA
    lora_dir = paths['loras_dir'] / args.person
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    # Создать тренер
    base_model_path = config.get('base_model.name', 'Qwen/Qwen2-VL-7B-Instruct')
    trainer = LoRATrainer(
        config=config.training,
        base_model_path=base_model_path,
        dataset_dir=dataset_dir,
        output_dir=lora_dir
    )
    
    # Запустить обучение
    try:
        manifest_file = trainer.train()
        logger.info(f"✅ Обучение завершено")
        logger.info(f"   Результаты сохранены в: {lora_dir}")
        return 0
    except Exception as e:
        logger.error(f"Ошибка при обучении: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

