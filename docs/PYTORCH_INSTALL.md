# Установка PyTorch для RTX 5060 Ti

## Проблема

RTX 5060 Ti имеет архитектуру **sm_120** (compute capability 12.0), которая не поддерживается стандартными стабильными версиями PyTorch. Требуется установка **nightly builds** PyTorch.

## ✅ Решение 1: Автоматическая установка (РЕКОМЕНДУЕТСЯ)

### Windows:
```bash
# Просто запустите bat-файл
scripts\install_pytorch_for_rtx5060.bat
```

### Linux/WSL/Windows (Python скрипт):
```bash
python scripts/install_pytorch_nightly.py
```

## ✅ Решение 2: Ручная установка

### Шаг 1: Удалить старую версию PyTorch
```bash
pip uninstall -y torch torchvision torchaudio
```

### Шаг 2: Установить PyTorch nightly
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### Шаг 3: Проверить установку
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Проверка совместимости

После установки проверьте, что всё работает:

```python
import torch

# Проверить версию
print(f"PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Проверить compute capability
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: sm_{compute_cap[0]}{compute_cap[1]}")
    
    # Попробовать простую операцию
    try:
        x = torch.randn(10, 10).cuda()
        print("✅ GPU работает корректно!")
    except RuntimeError as e:
        print(f"❌ Ошибка GPU: {e}")
```

## Альтернативные варианты

### Если nightly не работает:

1. **Попробовать CUDA 12.1 nightly:**
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```

2. **Использовать CPU fallback:**
   - Код автоматически переключится на CPU при ошибках CUDA
   - Будет медленнее, но будет работать

3. **Дождаться стабильного релиза:**
   - Следите за обновлениями PyTorch
   - Новые архитектуры обычно поддерживаются в следующих стабильных релизах

## Известные проблемы

- **Предупреждение о sm_120**: Это нормально, если обучение работает
- **CUDA kernel errors**: Установите nightly версию PyTorch
- **Медленная работа на CPU**: Это ожидаемо, используйте GPU когда возможно

## Дополнительная информация

- [PyTorch Nightly Builds](https://pytorch.org/get-started/locally/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)



