# Руководство по настройке LoraQween

## Быстрая установка

1. **Клонируйте или скачайте проект**

2. **Запустите скрипт настройки:**
```bash
python scripts/setup.py
```

Этот скрипт:
- Проверит версию Python
- Установит зависимости
- Создаст необходимые директории
- Проверит наличие CUDA

3. **Скачайте базовую модель:**
```bash
python scripts/download_base_model.py
```

⚠️ **ВАЖНО**: Модель Qwen-Image 20B должна быть скачана отдельно (см. `docs/MODEL_DOWNLOAD.md`)

## Ручная установка

Если автоматический скрипт не работает:

### 1. Создайте виртуальное окружение

```bash
python -m venv venv
source venv/bin/activate  # Linux/WSL
venv\Scripts\activate     # Windows
```

### 2. Установите PyTorch с CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

### 4. Установите InsightFace (для Face-ID)

```bash
pip install insightface
```

### 5. Создайте директории

Скрипт setup.py создаст их автоматически, или создайте вручную согласно структуре в README.md

## Проверка установки

```bash
# Проверить Python
python --version  # Должно быть 3.10+

# Проверить CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Проверить конфигурацию
python -c "from src.config import get_config; print('OK')"
```

## Следующие шаги

1. Подготовьте данные для обучения (см. `docs/QUICKSTART.md`)
2. Обучите LoRA
3. Запустите генерацию

## Решение проблем

### CUDA не обнаружена
- Убедитесь, что установлены драйверы NVIDIA
- Проверьте совместимость версии CUDA
- Переустановите PyTorch с правильной версией CUDA

### Ошибки импорта
- Убедитесь, что виртуальное окружение активировано
- Переустановите зависимости: `pip install -r requirements.txt --force-reinstall`

### Недостаточно VRAM
- Используйте degraded профиль в config.yaml
- Уменьшите batch_size и resolution
- Используйте более агрессивную квантизацию модели

## Дополнительная документация

- `docs/INSTALLATION.md` - Подробная инструкция по установке
- `docs/QUICKSTART.md` - Быстрый старт
- `docs/MODEL_DOWNLOAD.md` - Инструкция по скачиванию модели
- `README.md` - Общая информация о проекте

