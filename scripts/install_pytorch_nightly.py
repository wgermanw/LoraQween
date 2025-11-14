"""Установка PyTorch nightly для поддержки новых архитектур GPU (sm_12x)."""

import subprocess
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_cuda_version():
    """Проверить версию CUDA."""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            logger.info(f"Текущая версия CUDA в PyTorch: {cuda_version}")
            return cuda_version
        else:
            logger.warning("CUDA не доступна в текущей установке PyTorch")
            return None
    except ImportError:
        logger.info("PyTorch не установлен")
        return None


def install_pytorch_nightly():
    """Установить PyTorch nightly с поддержкой CUDA 12.4+."""
    logger.info("="*70)
    logger.info("УСТАНОВКА PYTORCH NIGHTLY ДЛЯ RTX 5060 Ti (sm_120)")
    logger.info("="*70)
    
    # Проверить текущую версию
    current_cuda = check_cuda_version()
    
    logger.info("\nУстановка PyTorch nightly с CUDA 12.4+...")
    logger.info("(Это может занять несколько минут)")
    
    # Установить PyTorch nightly
    commands = [
        [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
        [sys.executable, "-m", "pip", "install", "--pre", "torch", "torchvision", 
         "--index-url", "https://download.pytorch.org/whl/nightly/cu124"],
    ]
    
    # Попробовать установить torchaudio отдельно (может быть недоступен)
    torchaudio_cmd = [sys.executable, "-m", "pip", "install", "--pre", "torchaudio", 
                      "--index-url", "https://download.pytorch.org/whl/nightly/cu124"]
    
    for cmd in commands:
        logger.info(f"\nВыполняется: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка: {e.stderr}")
            return False
    
    # Попробовать установить torchaudio отдельно (опционально)
    logger.info(f"\nПопытка установить torchaudio (может быть недоступен): {' '.join(torchaudio_cmd)}")
    try:
        result = subprocess.run(torchaudio_cmd, check=True, capture_output=True, text=True)
        logger.info("✓ torchaudio установлен")
    except subprocess.CalledProcessError:
        logger.warning("⚠ torchaudio недоступен в nightly сборках, пропускаем...")
        logger.info("Это не критично для обучения LoRA")
    
    logger.info("\n" + "="*70)
    logger.info("ПРОВЕРКА УСТАНОВКИ")
    logger.info("="*70)
    
    # Проверить установку
    try:
        import torch
        logger.info(f"✓ PyTorch установлен: {torch.__version__}")
        
        if torch.cuda.is_available():
            logger.info(f"✓ CUDA доступна: {torch.version.cuda}")
            logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            
            # Проверить compute capability
            compute_capability = torch.cuda.get_device_capability(0)
            compute_capability_str = f"sm_{compute_capability[0]}{compute_capability[1]}"
            logger.info(f"✓ Compute Capability: {compute_capability_str}")
            
            # Попробовать простую операцию на GPU
            try:
                test_tensor = torch.randn(10, 10).cuda()
                logger.info("✓ Тестовая операция на GPU выполнена успешно")
                logger.info("\n✅ PyTorch nightly установлен и работает корректно!")
                return True
            except RuntimeError as e:
                logger.error(f"✗ Ошибка при выполнении операции на GPU: {e}")
                logger.error("Возможно, нужна более новая версия или другой индекс")
                return False
        else:
            logger.warning("⚠ CUDA не доступна после установки")
            return False
            
    except ImportError as e:
        logger.error(f"✗ Ошибка импорта PyTorch: {e}")
        return False


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Установить PyTorch nightly для поддержки новых GPU")
    parser.add_argument("--check-only", action="store_true", help="Только проверить текущую установку")
    
    args = parser.parse_args()
    
    if args.check_only:
        check_cuda_version()
        return
    
    success = install_pytorch_nightly()
    
    if success:
        logger.info("\n" + "="*70)
        logger.info("✅ УСТАНОВКА ЗАВЕРШЕНА УСПЕШНО")
        logger.info("="*70)
        logger.info("\nТеперь можно запустить обучение:")
        logger.info("  python scripts/train_lora.py --person Mikassa")
        sys.exit(0)
    else:
        logger.error("\n" + "="*70)
        logger.error("✗ УСТАНОВКА ЗАВЕРШИЛАСЬ С ОШИБКАМИ")
        logger.error("="*70)
        logger.error("\nПопробуйте:")
        logger.error("  1. Обновить pip: pip install --upgrade pip")
        logger.error("  2. Установить вручную:")
        logger.error("     pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124")
        sys.exit(1)


if __name__ == "__main__":
    main()

