"""Скрипт для проверки наличия и правильности размещения моделей."""

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


def check_directory(path: Path, name: str, required: bool = False) -> bool:
    """Проверить существование директории."""
    exists = path.exists() and path.is_dir()
    if exists:
        logger.info(f"✓ {name}: {path}")
    elif required:
        logger.error(f"✗ {name} отсутствует (обязательно): {path}")
    else:
        logger.warning(f"⚠ {name} отсутствует (опционально): {path}")
    return exists


def check_files_in_dir(path: Path, extensions: list, name: str) -> list:
    """Найти файлы с указанными расширениями в директории."""
    if not path.exists():
        return []
    
    found_files = []
    for ext in extensions:
        found_files.extend(list(path.rglob(f"*{ext}")))
    
    if found_files:
        logger.info(f"  Найдено {len(found_files)} файлов {name} в {path}")
        for f in found_files[:5]:  # Показать первые 5
            logger.info(f"    - {f.relative_to(path)}")
        if len(found_files) > 5:
            logger.info(f"    ... и ещё {len(found_files) - 5} файлов")
    else:
        logger.warning(f"  Файлы {name} не найдены в {path}")
    
    return found_files


def check_base_model(config: dict) -> dict:
    """Проверить базовую модель."""
    logger.info("\n" + "="*60)
    logger.info("ПРОВЕРКА БАЗОВОЙ МОДЕЛИ")
    logger.info("="*60)
    
    base_dir = config.paths['base_model_dir']
    status = {
        'directory_exists': False,
        'tokenizer_exists': False,
        'model_files_exist': False,
        'model_info_exists': False,
        'issues': []
    }
    
    # Проверить директорию
    status['directory_exists'] = check_directory(base_dir, "Директория базовой модели", required=True)
    
    if not status['directory_exists']:
        status['issues'].append("Директория базовой модели не существует")
        return status
    
    # Проверить токенайзер
    tokenizer_dir = base_dir / "tokenizer"
    status['tokenizer_exists'] = check_directory(tokenizer_dir, "Директория токенайзера")
    
    if status['tokenizer_exists']:
        tokenizer_files = check_files_in_dir(tokenizer_dir, ['.json', '.txt'], "токенайзера")
        if not tokenizer_files:
            status['issues'].append("Токенайзер не содержит файлов")
    
    # Проверить файлы модели
    model_files = check_files_in_dir(base_dir, ['.safetensors', '.bin', '.pt', '.pth', '.ckpt'], "модели")
    status['model_files_exist'] = len(model_files) > 0
    
    if not status['model_files_exist']:
        status['issues'].append("Файлы весов модели не найдены")
    
    # Проверить model_info.json
    model_info_file = base_dir / "model_info.json"
    status['model_info_exists'] = model_info_file.exists()
    
    if status['model_info_exists']:
        try:
            with open(model_info_file, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                logger.info(f"  Информация о модели: {model_info.get('model_name', 'неизвестно')}")
                logger.info(f"  Статус: {model_info.get('status', 'неизвестно')}")
        except Exception as e:
            logger.warning(f"  Не удалось прочитать model_info.json: {e}")
    
    # Проверить наличие Qwen-Image-Edit-Rapid-AIO (ComfyUI workflow)
    rapid_dir = base_dir / "Qwen-Image-Edit-Rapid-AIO"
    if rapid_dir.exists():
        logger.info(f"  Найдена директория ComfyUI workflow: {rapid_dir}")
        versions = [d for d in rapid_dir.iterdir() if d.is_dir() and d.name.startswith('v')]
        if versions:
            logger.info(f"  Найдены версии: {[v.name for v in versions]}")
            # Проверить, есть ли файлы в версиях
            for v_dir in versions:
                files = list(v_dir.glob("*"))
                if files:
                    logger.info(f"    {v_dir.name}: найдено {len(files)} файлов")
                else:
                    logger.warning(f"    {v_dir.name}: пустая директория")
    
    return status


def check_loras(config: dict) -> dict:
    """Проверить LoRA модели."""
    logger.info("\n" + "="*60)
    logger.info("ПРОВЕРКА LORA МОДЕЛЕЙ")
    logger.info("="*60)
    
    loras_dir = config.paths['loras_dir']
    status = {
        'directory_exists': False,
        'loras_found': []
    }
    
    status['directory_exists'] = check_directory(loras_dir, "Директория LoRA", required=True)
    
    if status['directory_exists']:
        # Найти все поддиректории с LoRA
        lora_dirs = [d for d in loras_dir.iterdir() if d.is_dir()]
        
        for lora_dir in lora_dirs:
            lora_files = check_files_in_dir(lora_dir, ['.safetensors', '.pt', '.pth', '.ckpt'], f"LoRA ({lora_dir.name})")
            if lora_files:
                status['loras_found'].append(lora_dir.name)
        
        if not status['loras_found']:
            logger.info("  LoRA модели не найдены (это нормально, если ещё не обучали)")
    
    return status


def check_face_id(config: dict) -> dict:
    """Проверить Face-ID модели."""
    logger.info("\n" + "="*60)
    logger.info("ПРОВЕРКА FACE-ID МОДЕЛЕЙ")
    logger.info("="*60)
    
    face_id_dir = config.paths['face_id_dir']
    status = {
        'directory_exists': False,
        'models_found': False
    }
    
    status['directory_exists'] = check_directory(face_id_dir, "Директория Face-ID", required=False)
    
    if status['directory_exists']:
        # InsightFace скачивает модели автоматически при первом использовании
        # Проверим наличие стандартных моделей
        model_files = check_files_in_dir(face_id_dir, ['.onnx', '.param', '.bin'], "Face-ID")
        status['models_found'] = len(model_files) > 0
        
        if not status['models_found']:
            logger.info("  Face-ID модели будут скачаны автоматически при первом использовании")
    
    return status


def main():
    logger.info("="*60)
    logger.info("ПРОВЕРКА МОДЕЛЕЙ LORAQWEEN")
    logger.info("="*60)
    
    try:
        config = get_config()
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return 1
    
    # Проверить базовую модель
    base_status = check_base_model(config)
    
    # Проверить LoRA
    lora_status = check_loras(config)
    
    # Проверить Face-ID
    face_id_status = check_face_id(config)
    
    # Итоговый отчёт
    logger.info("\n" + "="*60)
    logger.info("ИТОГОВЫЙ ОТЧЁТ")
    logger.info("="*60)
    
    all_ok = True
    
    if base_status['directory_exists']:
        if base_status['model_files_exist']:
            logger.info("✓ Базовая модель: найдены файлы весов")
        else:
            logger.error("✗ Базовая модель: файлы весов не найдены")
            all_ok = False
            logger.info("  → Необходимо скачать модель Qwen-Image 20B")
            logger.info("  → См. docs/MODEL_DOWNLOAD.md для инструкций")
        
        if base_status['tokenizer_exists']:
            logger.info("✓ Токенайзер: найден")
        else:
            logger.warning("⚠ Токенайзер: не найден в локальной директории")
            logger.info("  → Токенайзер может быть в кэше HuggingFace")
    else:
        logger.error("✗ Директория базовой модели не существует")
        all_ok = False
    
    if lora_status['loras_found']:
        logger.info(f"✓ LoRA модели: найдено {len(lora_status['loras_found'])} персон")
        for person in lora_status['loras_found']:
            logger.info(f"    - {person}")
    else:
        logger.info("ℹ LoRA модели: не найдены (это нормально, если ещё не обучали)")
    
    if face_id_status['models_found']:
        logger.info("✓ Face-ID модели: найдены")
    else:
        logger.info("ℹ Face-ID модели: будут скачаны автоматически при первом использовании")
    
    # Рекомендации
    logger.info("\n" + "="*60)
    logger.info("РЕКОМЕНДАЦИИ")
    logger.info("="*60)
    
    if not base_status['model_files_exist']:
        logger.info("1. Скачайте модель Qwen-Image 20B:")
        logger.info("   - Проверьте официальный репозиторий Qwen")
        logger.info("   - Разместите модель в models/base/")
        logger.info("   - Модель должна быть в формате FP8/Q4 для экономии VRAM")
    
    if not base_status['tokenizer_exists']:
        logger.info("2. Токенайзер будет использоваться из кэша HuggingFace")
        logger.info("   Или запустите: python scripts/download_base_model.py")
    
    if not lora_status['loras_found']:
        logger.info("3. После подготовки данных и получения модели:")
        logger.info("   python scripts/train_lora.py --person <имя>")
    
    if all_ok and base_status['model_files_exist']:
        logger.info("\n✅ Все необходимые модели на месте!")
        logger.info("   Проект готов к использованию")
    else:
        logger.info("\n⚠️  Некоторые модели отсутствуют")
        logger.info("   Следуйте рекомендациям выше")
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

