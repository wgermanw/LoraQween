"""Скрипт подготовки данных для обучения."""

import argparse
import logging
import sys
from pathlib import Path

# Добавить корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.tokenizer.token_manager import TokenManager
from src.data.dataset_preparator import DatasetPreparator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Подготовка данных для обучения LoRA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Автоматический режим (из папки input_photos)
  python scripts/auto_prepare_data.py --person alex --gender male --age 30
  
  # Ручной режим (указать папку)
  python scripts/prepare_data.py --person alex --input_dir /path/to/photos
        """
    )
    parser.add_argument("--person", type=str, required=True, help="Имя персоны")
    parser.add_argument("--input_dir", type=str, required=True, help="Директория с исходными фото")
    parser.add_argument("--trigger_token", type=str, default=None, help="Триггер-токен (если не указан, будет создан автоматически)")
    parser.add_argument("--gender", type=str, default=None, help="Пол персоны (male/female)")
    parser.add_argument("--age", type=str, default=None, help="Возраст")
    parser.add_argument("--hair_color", type=str, default=None, help="Цвет волос")
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = get_config()
    paths = config.paths
    
    # Создать директорию для персоны
    person_dir = paths['datasets_dir'] / args.person
    person_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализировать менеджер токенов
    # Использовать токенайзер из скачанной модели Qwen/Qwen-Image
    base_model_name = config.get('base_model.name', 'Qwen/Qwen-Image')
    
    # Найти токенайзер в кэше модели
    cache_dir = paths['base_model_dir'] / "cache" / "models--Qwen--Qwen-Image" / "snapshots"
    tokenizer_path = base_model_name  # По умолчанию
    
    if cache_dir.exists():
        snapshots = list(cache_dir.iterdir())
        if snapshots:
            snapshot_dir = snapshots[0]
            tokenizer_cache = snapshot_dir / "tokenizer"
            if tokenizer_cache.exists():
                # Использовать токенайзер из кэша
                tokenizer_path = str(tokenizer_cache)
                logger.info(f"Использование токенайзера из кэша: {tokenizer_path}")
    
    token_manager = TokenManager(tokenizer_path, paths['datasets_dir'])
    
    # Создать или получить токен
    if args.trigger_token:
        trigger_token = args.trigger_token
        # Проверить валидность
        is_valid, token_ids = token_manager.validate_runtime_token(trigger_token)
        if not is_valid:
            logger.error(f"Токен '{trigger_token}' невалиден!")
            return 1
    else:
        trigger_token = token_manager.create_token(args.person)
    
    # Добавить токен в токенайзер, если нужно
    token_manager.add_token_to_tokenizer(trigger_token)
    
    # Сохранить токенайзер с новым токеном
    tokenizer_output = person_dir / "tokenizer"
    token_manager.save_tokenizer(tokenizer_output)
    
    # Подготовить метаданные
    metadata = {}
    if args.gender:
        metadata['gender'] = args.gender
    if args.age:
        metadata['age'] = args.age
    if args.hair_color:
        metadata['hair_color'] = args.hair_color
    
    # Подготовить датасет
    preparator = DatasetPreparator(
        person_name=args.person,
        trigger_token=trigger_token,
        output_dir=person_dir,
        config=config.training
    )
    
    dataset_path = preparator.prepare_dataset(
        input_dir=Path(args.input_dir),
        metadata=metadata
    )
    
    # Создать бакеты
    images_dir = dataset_path / "images"
    preparator.create_buckets(images_dir)
    
    logger.info(f"✅ Данные подготовлены для персоны '{args.person}'")
    logger.info(f"   Токен: {trigger_token}")
    logger.info(f"   Директория: {dataset_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

