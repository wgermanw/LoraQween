"""Скрипт инференса для генерации изображений."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.inference.inference_engine import InferenceEngine
from src.utils.vram_manager import VRAMManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Генерация изображений")
    parser.add_argument("--person", type=str, required=True, help="Имя персоны")
    parser.add_argument("--prompt", type=str, required=True, help="Промт с триггер-токеном")
    parser.add_argument("--mode", type=str, choices=['fast', 'reliable'], default='fast', help="Режим генерации")
    parser.add_argument("--num_images", type=int, default=4, help="Количество изображений")
    parser.add_argument("--seed", type=int, default=None, help="Seed для воспроизводимости")
    parser.add_argument("--reference", type=str, default=None, help="Путь к референсному изображению (для надёжного режима)")
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = get_config()
    paths = config.paths
    
    # Проверить VRAM
    vram_manager = VRAMManager(config.get('vram', {}))
    memory_info = vram_manager.get_memory_usage()
    logger.info(f"Использование VRAM: {memory_info.get('used_gb', 0):.2f} GB / {memory_info.get('total_gb', 0):.2f} GB")
    
    if vram_manager.should_degrade():
        logger.warning("⚠️  Высокое использование VRAM. Рекомендуется использовать degraded профиль")
        degraded_config = vram_manager.get_degraded_config()
        logger.info(f"Degraded профиль: {degraded_config}")
    
    # Найти LoRA для персоны
    lora_path = paths['loras_dir'] / args.person
    if not lora_path.exists():
        logger.warning(f"LoRA для персоны '{args.person}' не найдена: {lora_path}")
        logger.warning("Продолжение без LoRA...")
        lora_path = None
    
    # Создать движок инференса
    base_model_path = config.get('base_model.name', 'Qwen/Qwen2-VL-7B-Instruct')
    engine = InferenceEngine(
        config=config.inference,
        base_model_path=base_model_path,
        lora_path=lora_path
    )
    
    # Загрузить модель
    try:
        engine.load_model()
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        logger.error("Убедитесь, что базовая модель Qwen-Image скачана")
        return 1
    
    # Загрузить референсное изображение если нужно
    reference_image = None
    if args.reference:
        from PIL import Image
        reference_image = Image.open(args.reference).convert('RGB')
    
    # Генерация
    try:
        if args.mode == 'fast':
            images = engine.generate_fast(
                prompt=args.prompt,
                num_images=args.num_images,
                seed=args.seed
            )
        else:
            images = engine.generate_reliable(
                prompt=args.prompt,
                reference_image=reference_image,
                num_images=args.num_images,
                seed=args.seed
            )
        
        # Сохранить результаты
        output_dir = paths['outputs_dir'] / args.person
        log_file = engine.save_generation_log(
            prompt=args.prompt,
            images=images,
            mode=args.mode,
            seed=args.seed,
            output_dir=output_dir
        )
        
        logger.info(f"✅ Генерация завершена. Результаты сохранены в: {output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка при генерации: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

