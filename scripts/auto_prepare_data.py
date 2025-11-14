"""Автоматическая подготовка данных из папки input_photos."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_data import main as prepare_main

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Автоматическая подготовка данных из папки input_photos"
    )
    parser.add_argument("--person", type=str, required=True, help="Имя персоны")
    parser.add_argument("--gender", type=str, default=None, help="Пол персоны (male/female)")
    parser.add_argument("--age", type=str, default=None, help="Возраст")
    parser.add_argument("--hair_color", type=str, default=None, help="Цвет волос")
    parser.add_argument("--trigger_token", type=str, default=None, help="Триггер-токен (если не указан, будет создан автоматически)")
    
    args = parser.parse_args()
    
    # Найти папку с фото
    input_photos_dir = Path("data/input_photos") / args.person
    
    if not input_photos_dir.exists():
        logger.error(f"Папка не найдена: {input_photos_dir}")
        logger.error(f"\nСоздайте папку и положите туда фото:")
        logger.error(f"  mkdir {input_photos_dir}")
        logger.error(f"  # Затем скопируйте фото в эту папку")
        return 1
    
    # Проверить наличие фото
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.JPG', '.JPEG', '.PNG', '.WEBP'}
    image_files = [f for f in input_photos_dir.iterdir() 
                   if f.suffix in image_extensions and f.is_file()]
    
    if len(image_files) == 0:
        logger.error(f"В папке {input_photos_dir} не найдено фото!")
        logger.error("Поддерживаемые форматы: JPG, PNG, WEBP")
        return 1
    
    logger.info(f"Найдено {len(image_files)} фото в {input_photos_dir}")
    
    if len(image_files) < 10:
        logger.warning(f"⚠ Мало фото ({len(image_files)}). Рекомендуется 40-80 фото для лучшего качества")
    elif len(image_files) > 100:
        logger.warning(f"⚠ Много фото ({len(image_files)}). Рекомендуется 40-80 фото")
    
    # Подготовить аргументы для prepare_data.py
    sys.argv = [
        'prepare_data.py',
        '--person', args.person,
        '--input_dir', str(input_photos_dir)
    ]
    
    if args.gender:
        sys.argv.extend(['--gender', args.gender])
    if args.age:
        sys.argv.extend(['--age', args.age])
    if args.hair_color:
        sys.argv.extend(['--hair_color', args.hair_color])
    if args.trigger_token:
        sys.argv.extend(['--trigger_token', args.trigger_token])
    
    # Запустить prepare_data.py
    logger.info("\n" + "="*70)
    logger.info("НАЧАЛО ОБРАБОТКИ ДАННЫХ")
    logger.info("="*70)
    
    return prepare_main()


if __name__ == "__main__":
    sys.exit(main())

