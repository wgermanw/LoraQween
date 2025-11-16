"""Проверка установки PyTorch."""

import sys

try:
    import torch
    import torchvision
    
    print("="*70)
    print("ПРОВЕРКА УСТАНОВКИ PYTORCH")
    print("="*70)
    print(f"PyTorch: {torch.__version__}")
    print(f"torchvision: {torchvision.__version__}")
    print(f"CUDA доступна: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA версия: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Проверить compute capability
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"Compute Capability: sm_{compute_cap[0]}{compute_cap[1]}")
        
        # Тестовая операция на GPU
        try:
            x = torch.randn(10, 10).cuda()
            y = x @ x.T
            print("✅ GPU работает корректно!")
            print("✅ Все готово для обучения!")
            sys.exit(0)
        except RuntimeError as e:
            print(f"❌ Ошибка при работе с GPU: {e}")
            sys.exit(1)
    else:
        print("⚠ CUDA не доступна")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)




