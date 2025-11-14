"""Скрипт для первоначальной настройки проекта."""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Проверить версию Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        logger.error(f"Требуется Python 3.10+, текущая версия: {version.major}.{version.minor}")
        return False
    logger.info(f"Python версия: {version.major}.{version.minor}.{version.micro} ✓")
    return True


def check_cuda():
    """Проверить наличие CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA доступна: {torch.cuda.get_device_name(0)} ✓")
            logger.info(f"CUDA версия: {torch.version.cuda} ✓")
            return True
        else:
            logger.warning("CUDA не доступна. Убедитесь, что установлены драйверы NVIDIA и PyTorch с CUDA")
            return False
    except ImportError:
        logger.warning("PyTorch не установлен. Будет установлен в следующем шаге.")
        return None


def install_requirements():
    """Установить зависимости."""
    logger.info("Установка зависимостей из requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        logger.info("Зависимости установлены ✓")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка установки зависимостей: {e}")
        return False


def create_directories():
    """Создать необходимые директории."""
    # Добавить корневую директорию в путь
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        logger.warning("config.yaml не найден. Создайте его вручную или используйте пример.")
        return False
    
    from src.config import get_config
    config = get_config()
    paths = config.paths
    
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Директория создана/проверена: {path} ✓")
    
    return True


def main():
    logger.info("=" * 50)
    logger.info("Настройка проекта LoraQween")
    logger.info("=" * 50)
    
    # Проверить Python
    if not check_python_version():
        return 1
    
    # Проверить CUDA (опционально, если PyTorch установлен)
    cuda_status = check_cuda()
    
    # Установить зависимости
    if not install_requirements():
        return 1
    
    # Проверить CUDA после установки PyTorch
    if cuda_status is None:
        check_cuda()
    
    # Создать директории
    if not create_directories():
        logger.warning("Некоторые директории не были созданы. Проверьте config.yaml")
    
    logger.info("=" * 50)
    logger.info("Настройка завершена!")
    logger.info("=" * 50)
    logger.info("Следующие шаги:")
    logger.info("1. Скачайте модель Qwen-Image 20B в models/base/")
    logger.info("2. Подготовьте данные: python scripts/prepare_data.py --person <name> --input_dir <dir>")
    logger.info("3. Обучите LoRA: python scripts/train_lora.py --person <name>")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

