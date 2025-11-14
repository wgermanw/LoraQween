"""Проверка компонентов Qwen для обучения LoRA."""

import sys
import logging
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_model_components():
    """Проверить наличие всех компонентов модели для обучения LoRA."""
    logger.info("="*70)
    logger.info("ПРОВЕРКА КОМПОНЕНТОВ QWEN ДЛЯ ОБУЧЕНИЯ LORA")
    logger.info("="*70)
    
    config = get_config()
    base_dir = config.paths['base_model_dir']
    
    # Согласно ТЗ нужны:
    # 1. Веса UNet/MMDiT (основная модель)
    # 2. Текстовый энкодер
    # 3. VAE
    # 4. tokenizer.json
    
    components = {
        'unet': {'found': False, 'paths': [], 'required': True, 'description': 'Веса UNet/MMDiT (основная модель)'},
        'text_encoder': {'found': False, 'paths': [], 'required': True, 'description': 'Текстовый энкодер'},
        'vae': {'found': False, 'paths': [], 'required': True, 'description': 'VAE (вариационный автокодировщик)'},
        'tokenizer': {'found': False, 'paths': [], 'required': True, 'description': 'Tokenizer (токенайзер)'},
    }
    
    # Проверить токенайзер
    tokenizer_dir = base_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer_json = tokenizer_dir / "tokenizer.json"
        if tokenizer_json.exists():
            components['tokenizer']['found'] = True
            components['tokenizer']['paths'].append(str(tokenizer_json))
            logger.info(f"✓ Токенайзер найден: {tokenizer_json}")
        else:
            logger.warning(f"⚠ Токенайзер: tokenizer.json не найден в {tokenizer_dir}")
    else:
        logger.error(f"✗ Токенайзер: директория не найдена")
    
    # Проверить структуру diffusers (стандартный формат)
    unet_dir = base_dir / "unet"
    text_encoder_dir = base_dir / "text_encoder"
    vae_dir = base_dir / "vae"
    
    if unet_dir.exists():
        unet_files = list(unet_dir.glob("*.safetensors")) + list(unet_dir.glob("*.bin"))
        if unet_files:
            components['unet']['found'] = True
            components['unet']['paths'] = [str(f) for f in unet_files]
            logger.info(f"✓ UNet найден: {len(unet_files)} файлов")
    
    if text_encoder_dir.exists():
        te_files = list(text_encoder_dir.glob("*.safetensors")) + list(text_encoder_dir.glob("*.bin"))
        if te_files:
            components['text_encoder']['found'] = True
            components['text_encoder']['paths'] = [str(f) for f in te_files]
            logger.info(f"✓ Текстовый энкодер найден: {len(te_files)} файлов")
    
    if vae_dir.exists():
        vae_files = list(vae_dir.glob("*.safetensors")) + list(vae_dir.glob("*.bin"))
        if vae_files:
            components['vae']['found'] = True
            components['vae']['paths'] = [str(f) for f in vae_files]
            logger.info(f"✓ VAE найден: {len(vae_files)} файлов")
    
    # Проверить единый checkpoint файл (ComfyUI формат)
    checkpoint_files = list(base_dir.rglob("*.safetensors")) + list(base_dir.rglob("*.ckpt"))
    checkpoint_files = [f for f in checkpoint_files if f.is_file() and 'Qwen' in f.name]
    
    if checkpoint_files:
        logger.info(f"\nНайдены checkpoint файлы (возможно, объединённые модели):")
        for cf in checkpoint_files:
            size_gb = cf.stat().st_size / (1024**3)
            logger.info(f"  - {cf.relative_to(base_dir)} ({size_gb:.2f} GB)")
        
        # Если найден единый checkpoint, но нет отдельных компонентов
        if not components['unet']['found']:
            logger.warning("\n⚠ ВНИМАНИЕ: Найден единый checkpoint файл, но нет отдельных компонентов")
            logger.warning("  Для обучения LoRA через diffusers обычно нужны отдельные компоненты")
            logger.warning("  Или модель должна быть в формате diffusers (с поддиректориями)")
    
    # Проверить processor
    processor_dir = base_dir / "processor"
    if processor_dir.exists():
        logger.info(f"\n✓ Процессор найден: {processor_dir}")
    
    # Итоговый отчёт
    logger.info("\n" + "="*70)
    logger.info("ИТОГОВЫЙ ОТЧЁТ")
    logger.info("="*70)
    
    all_found = all(comp['found'] for comp in components.values() if comp['required'])
    
    for name, comp in components.items():
        status = "✓" if comp['found'] else "✗"
        logger.info(f"{status} {comp['description']}: {'найден' if comp['found'] else 'НЕ НАЙДЕН'}")
        if comp['paths']:
            for path in comp['paths'][:3]:  # Показать первые 3
                logger.info(f"    {Path(path).name}")
    
    # Рекомендации
    logger.info("\n" + "="*70)
    logger.info("РЕКОМЕНДАЦИИ")
    logger.info("="*70)
    
    if not all_found:
        logger.info("\nДля обучения LoRA нужны следующие компоненты:")
        logger.info("1. UNet/MMDiT веса (основная модель)")
        logger.info("2. Текстовый энкодер")
        logger.info("3. VAE")
        logger.info("4. Tokenizer (✓ уже есть)")
        
        if checkpoint_files:
            logger.info("\n⚠ Найден единый checkpoint файл:")
            logger.info(f"  {checkpoint_files[0].name}")
            logger.info("\nВарианты решения:")
            logger.info("1. Использовать модель в формате diffusers (с поддиректориями unet/, text_encoder/, vae/)")
            logger.info("2. Конвертировать checkpoint в формат diffusers")
            logger.info("3. Использовать модель напрямую через ComfyUI (если доступно)")
            logger.info("\nДля обучения через наш код нужна модель в формате diffusers или Hugging Face")
        else:
            logger.info("\n→ Скачайте модель Qwen-Image 20B в формате diffusers")
            logger.info("  Структура должна быть:")
            logger.info("  models/base/")
            logger.info("    ├── unet/")
            logger.info("    ├── text_encoder/")
            logger.info("    ├── vae/")
            logger.info("    └── tokenizer/ (✓ уже есть)")
    else:
        logger.info("\n✅ Все необходимые компоненты найдены!")
        logger.info("Модель готова для обучения LoRA")
    
    return all_found


if __name__ == "__main__":
    success = check_model_components()
    sys.exit(0 if success else 1)

