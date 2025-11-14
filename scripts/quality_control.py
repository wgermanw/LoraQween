"""Скрипт контроля качества генерации."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.quality.quality_controller import QualityController

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Контроль качества генерации")
    parser.add_argument("--person", type=str, required=True, help="Имя персоны")
    parser.add_argument("--generated_dir", type=str, required=True, help="Директория со сгенерированными изображениями")
    parser.add_argument("--reference_dir", type=str, required=True, help="Директория с референсными изображениями")
    parser.add_argument("--output", type=str, default=None, help="Файл для сохранения отчёта")
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = get_config()
    
    # Создать контроллер качества
    controller = QualityController(config.quality_control)
    
    # Найти изображения
    generated_dir = Path(args.generated_dir)
    reference_dir = Path(args.reference_dir)
    
    generated_images = sorted(list(generated_dir.glob("*.png")) + list(generated_dir.glob("*.jpg")))
    reference_images = sorted(list(reference_dir.glob("*.png")) + list(reference_dir.glob("*.jpg")))
    
    if len(generated_images) != len(reference_images):
        logger.warning(f"Количество изображений не совпадает: {len(generated_images)} vs {len(reference_images)}")
        min_count = min(len(generated_images), len(reference_images))
        generated_images = generated_images[:min_count]
        reference_images = reference_images[:min_count]
    
    # Оценить качество
    logger.info(f"Оценка качества для {len(generated_images)} пар изображений...")
    metrics = controller.evaluate_batch(generated_images, reference_images)
    
    # Вывести результаты
    logger.info("=" * 50)
    logger.info("РЕЗУЛЬТАТЫ ОЦЕНКИ КАЧЕСТВА")
    logger.info("=" * 50)
    logger.info(f"Средняя схожесть: {metrics['mean_similarity']:.3f}")
    logger.info(f"Медианная схожесть: {metrics['median_similarity']:.3f}")
    logger.info(f"Минимальная схожесть: {metrics['min_similarity']:.3f}")
    logger.info(f"Максимальная схожесть: {metrics['max_similarity']:.3f}")
    logger.info(f"Стандартное отклонение: {metrics.get('std_similarity', 0):.3f}")
    logger.info(f"Ниже порога ({controller.similarity_threshold}): {metrics['below_threshold']} из {metrics['num_valid']}")
    
    # Проверить необходимость переключения режима
    if controller.should_switch_to_reliable_mode(metrics):
        logger.warning("⚠️  Рекомендуется переключиться на надёжный режим")
    else:
        logger.info("✅ Качество соответствует требованиям")
    
    # Сохранить отчёт
    if args.output:
        import json
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        report = {
            'person': args.person,
            'metrics': metrics,
            'recommendation': 'switch_to_reliable' if controller.should_switch_to_reliable_mode(metrics) else 'ok'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Отчёт сохранён: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

