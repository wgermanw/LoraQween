# Решение проблемы "No space left on device" при установке PyTorch

## Проблема

При установке PyTorch nightly возникает ошибка:
```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

## Решения

### ✅ Решение 1: Освободить место на диске C

1. **Очистить временные файлы:**
   ```powershell
   # Очистить кэш pip
   pip cache purge
   
   # Очистить временные файлы Windows
   # Win+R -> %temp% -> удалить старые файлы
   ```

2. **Очистить кэш Python:**
   ```powershell
   # Найти кэш
   python -m pip cache dir
   
   # Очистить
   python -m pip cache purge
   ```

3. **Удалить старые версии PyTorch:**
   ```powershell
   pip uninstall -y torch torchvision torchaudio
   ```

### ✅ Решение 2: Установить в другое место

1. **Использовать виртуальное окружение на диске D:**
   ```powershell
   # Создать venv на диске D
   python -m venv D:\venv_loraqween
   
   # Активировать
   D:\venv_loraqween\Scripts\activate
   
   # Установить PyTorch
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
   ```

2. **Изменить TMPDIR для установки:**
   ```powershell
   # Установить временную директорию на диск D
   $env:TMPDIR = "D:\temp"
   $env:TMP = "D:\temp"
   $env:TEMP = "D:\temp"
   
   # Создать директорию
   New-Item -ItemType Directory -Force -Path "D:\temp"
   
   # Установить PyTorch
   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
   ```

### ✅ Решение 3: Использовать CPU версию (временно)

Если нужно срочно продолжить работу:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Внимание:** Это будет работать только на CPU, обучение будет очень медленным.

### ✅ Решение 4: Проверить, установился ли PyTorch частично

Возможно, PyTorch уже установлен, но установка прервалась:

```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

Если PyTorch импортируется, можно попробовать доустановить только torchvision:

```powershell
pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu124 --no-deps
```

## Рекомендация

**Лучше всего:** Освободить место на диске C (нужно минимум 5-10 GB свободного места) и повторить установку.

## Проверка после установки

```python
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA версия: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Проверить compute capability
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Compute Capability: sm_{compute_cap[0]}{compute_cap[1]}")
    
    # Тестовая операция
    try:
        x = torch.randn(10, 10).cuda()
        print("✅ GPU работает корректно!")
    except RuntimeError as e:
        print(f"❌ Ошибка GPU: {e}")
```




