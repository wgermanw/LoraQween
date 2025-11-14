# Инструкция по установке

## Требования

- Windows 11 / WSL2 с CUDA 12.x
- Python 3.10+
- NVIDIA RTX 5060 Ti 16 GB (или совместимая видеокарта)
- Локальный SSD ≥ 200 ГБ свободного места

## Установка

### 1. Клонирование и подготовка окружения

```bash
# Создать виртуальное окружение
python -m venv venv

# Активировать окружение
# Windows:
venv\Scripts\activate
# Linux/WSL:
source venv/bin/activate

# Обновить pip
pip install --upgrade pip
```

### 2. Установка PyTorch с CUDA

```bash
# Для CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Установка зависимостей проекта

```bash
pip install -r requirements.txt
```

### 4. Установка InsightFace (для Face-ID)

```bash
pip install insightface
# Скачать модели (выполнится автоматически при первом использовании)
```

### 5. Скачивание базовой модели

```bash
python scripts/download_base_model.py
```

⚠️ **ВАЖНО**: Модель Qwen-Image 20B должна быть скачана отдельно из официального репозитория Qwen. Скрипт скачает только токенайзер и процессор.

### 6. Настройка конфигурации

Отредактируйте `config.yaml` при необходимости, указав правильные пути к моделям.

## Проверка установки

```bash
# Проверить GPU
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')"

# Проверить установку
python -c "from src.config import get_config; print('Конфигурация загружена успешно')"
```

## Следующие шаги

1. Подготовьте данные для обучения (см. `docs/DATA_PREPARATION.md`)
2. Обучите LoRA (см. `docs/TRAINING.md`)
3. Запустите генерацию (см. `docs/INFERENCE.md`)

