"""Скрипт для скачивания базовой модели Qwen-Image."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Скачать базовую модель Qwen-Image")
    parser.add_argument("--model_name", type=str, default=None, help="Имя модели (если не указано, берётся из config)")
    parser.add_argument("--output_dir", type=str, default=None, help="Директория для сохранения")
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = get_config()
    
    # Определить модель и директорию
    model_name = args.model_name or config.get('base_model.name', 'Qwen/Qwen2-VL-7B-Instruct')
    output_dir = Path(args.output_dir) if args.output_dir else config.paths['base_model_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Скачивание модели: {model_name}")
    logger.info(f"Директория: {output_dir}")
    
    try:
        # Попытка скачать через transformers
        from transformers import AutoTokenizer, AutoProcessor
        
        logger.info("Скачивание токенайзера...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.save_pretrained(output_dir / "tokenizer")
        
        logger.info("Скачивание процессора...")
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            processor.save_pretrained(output_dir / "processor")
        except Exception as e:
            logger.warning(f"Не удалось загрузить процессор: {e}")
        
        logger.info("✅ Токенайзер и процессор скачаны")
        logger.warning("⚠️  ВАЖНО: Модель Qwen-Image 20B должна быть скачана отдельно")
        logger.warning("⚠️  Проверьте официальный репозиторий Qwen для получения модели")
        logger.warning("⚠️  Модель должна быть в формате FP8/Q4 для экономии VRAM")
        
        # Сохранить информацию о модели
        model_info = {
            'model_name': model_name,
            'output_dir': str(output_dir),
            'status': 'tokenizer_downloaded',
            'note': 'Модель Qwen-Image 20B должна быть скачана отдельно'
        }
        
        info_file = output_dir / "model_info.json"
        import json
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Информация сохранена: {info_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка при скачивании: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

