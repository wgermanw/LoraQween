"""Скачивание официальной модели Qwen/Qwen-Image через diffusers."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Скачать официальную модель Qwen/Qwen-Image через diffusers")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen-Image", help="Имя модели на Hugging Face")
    parser.add_argument("--output_dir", type=str, default=None, help="Директория для сохранения (по умолчанию models/base)")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp8"], help="Тип данных для загрузки")
    parser.add_argument("--check_only", action="store_true", help="Только проверить доступность модели, не скачивать")
    
    args = parser.parse_args()
    
    # Загрузить конфигурацию
    config = get_config()
    output_dir = Path(args.output_dir) if args.output_dir else config.paths['base_model_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*70)
    logger.info("СКАЧИВАНИЕ ОФИЦИАЛЬНОЙ МОДЕЛИ QWEN/QWEN-IMAGE")
    logger.info("="*70)
    logger.info(f"Модель: {args.model_name}")
    logger.info(f"Директория: {output_dir}")
    logger.info(f"Тип данных: {args.dtype}")
    
    try:
        from diffusers import DiffusionPipeline
        from transformers import AutoTokenizer
        
        # Проверить доступность модели
        logger.info("\nПроверка доступности модели на Hugging Face...")
        try:
            # Попытка получить информацию о модели
            from huggingface_hub import model_info
            info = model_info(args.model_name)
            logger.info(f"✓ Модель найдена: {info.id}")
            logger.info(f"  Pipeline tag: {info.pipeline_tag if hasattr(info, 'pipeline_tag') else 'unknown'}")
            
            if args.check_only:
                logger.info("\n✅ Модель доступна для скачивания!")
                return 0
                
        except Exception as e:
            logger.warning(f"Не удалось проверить модель через model_info: {e}")
            logger.info("Продолжаем попытку загрузки...")
        
        # Определить dtype
        import torch
        if args.dtype == "fp16":
            torch_dtype = torch.float16
        elif args.dtype == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16  # fp8 пока не поддерживается напрямую
        
        logger.info("\n" + "="*70)
        logger.info("СКАЧИВАНИЕ МОДЕЛИ")
        logger.info("="*70)
        logger.info("Это может занять много времени и места на диске...")
        logger.info("Модель будет скачана в кэш Hugging Face и локально")
        
        # Скачать модель через DiffusionPipeline
        logger.info(f"\nЗагрузка модели {args.model_name}...")
        logger.info("(Это может занять 10-30 минут в зависимости от скорости интернета)")
        
        pipeline = DiffusionPipeline.from_pretrained(
            args.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            cache_dir=str(output_dir / "cache")
        )
        
        logger.info("✓ Модель загружена в память")
        
        # Сохранить модель локально
        logger.info(f"\nСохранение модели в {output_dir}...")
        pipeline.save_pretrained(str(output_dir))
        
        logger.info("✓ Модель сохранена локально")
        
        # Проверить структуру
        logger.info("\n" + "="*70)
        logger.info("ПРОВЕРКА СТРУКТУРЫ")
        logger.info("="*70)
        
        expected_dirs = ["unet", "text_encoder", "vae", "tokenizer"]
        for dir_name in expected_dirs:
            dir_path = output_dir / dir_name
            if dir_path.exists():
                files = list(dir_path.glob("*"))
                logger.info(f"✓ {dir_name}/: найдено {len(files)} файлов")
            else:
                logger.warning(f"⚠ {dir_name}/: не найдена")
        
        # Сохранить информацию о модели
        model_info = {
            'model_name': args.model_name,
            'output_dir': str(output_dir),
            'dtype': args.dtype,
            'status': 'downloaded',
            'format': 'diffusers',
            'note': 'Официальная модель через DiffusionPipeline'
        }
        
        info_file = output_dir / "model_info.json"
        import json
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✅ Модель успешно скачана и сохранена!")
        logger.info(f"   Информация: {info_file}")
        logger.info("\n" + "="*70)
        logger.info("СЛЕДУЮЩИЕ ШАГИ")
        logger.info("="*70)
        logger.info("1. Обновите config.yaml:")
        logger.info(f'   base_model:')
        logger.info(f'     name: "{args.model_name}"  # или "{output_dir}" для локальной модели')
        logger.info("2. Подготовьте данные: python scripts/prepare_data.py --person <name> --input_dir <dir>")
        logger.info("3. Обучите LoRA: python scripts/train_lora.py --person <name>")
        
        return 0
        
    except ImportError as e:
        logger.error(f"Не установлена необходимая библиотека: {e}")
        logger.error("Установите: pip install diffusers transformers")
        return 1
    except Exception as e:
        logger.error(f"Ошибка при скачивании модели: {e}", exc_info=True)
        logger.error("\nВозможные причины:")
        logger.error("1. Модель недоступна на Hugging Face")
        logger.error("2. Недостаточно места на диске")
        logger.error("3. Проблемы с интернет-соединением")
        logger.error("\nПроверьте доступность модели:")
        logger.error(f"  https://huggingface.co/{args.model_name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

