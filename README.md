# LoraQween - Система генерации фотореалистичных портретов

Локальная система генерации фотореалистичных портретов на основе Qwen-Image 20B с поддержкой LoRA и FaceID/IP-Adapter.

## Требования

- Windows 11 / WSL2 с CUDA 12.x
- Python 3.10+
- NVIDIA RTX 5060 Ti 16 GB (или совместимая видеокарта)
- Локальный SSD ≥ 200 ГБ

## Установка

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # Linux/WSL
# или
venv\Scripts\activate  # Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Установите PyTorch с поддержкой CUDA:

**Для RTX 5060 Ti (sm_120) и других новых GPU:**
```bash
# Используйте скрипт для автоматической установки nightly версии
python scripts/install_pytorch_nightly.py

# Или вручную:
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

**Для старых GPU (RTX 30xx, 40xx и т.д.):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Структура проекта

```
LoraQween/
├── data/                    # Данные для обучения
│   ├── datasets/            # Датасеты персон
│   ├── references/          # Эталонные фото
│   └── regularization/      # Регуляризационные данные
├── models/                  # Модели и веса
│   ├── base/                # Базовая модель Qwen-Image
│   ├── loras/               # Обученные LoRA
│   └── face_id/             # Face-ID модели
├── scripts/                 # Скрипты
│   ├── train_lora.py        # Обучение LoRA
│   ├── inference.py         # Инференс
│   ├── prepare_data.py      # Подготовка данных
│   └── quality_control.py   # Контроль качества
├── src/                     # Исходный код
│   ├── tokenizer/           # Управление токенами
│   ├── data/                # Обработка данных
│   ├── training/            # Обучение
│   ├── inference/            # Инференс
│   └── quality/             # Контроль качества
├── logs/                    # Логи
├── outputs/                 # Результаты генерации
└── comfyui/                 # Интеграция с ComfyUI
```

## Быстрый старт

1. Подготовка данных:
```bash
python scripts/prepare_data.py --person alex --input_dir /path/to/photos
```

2. Обучение LoRA:
```bash
python scripts/train_lora.py --person alex --trigger_token "<qwn_alex>"
```

3. Генерация:
```bash
python scripts/inference.py --person alex --prompt "<qwn_alex> portrait, smiling" --mode fast
```

## Запуск на vast.ai / Linux

1. Клонируйте репозиторий и создайте окружение (Python 3.10+).
2. `pip install -r requirements.txt` (при необходимости выполните `python scripts/install_pytorch_nightly.py`).
3. Скачайте базовую модель Qwen-Image:  
   `python scripts/download_qwen_image.py --dtype bf16 --output_dir models/base`
4. Скопируйте сырые фото в `data/input_photos/<person>/` и запустите подготовку:  
   `python scripts/auto_prepare_data.py --person <person> [--gender ...]`
5. Запустите обучение через кэшируемый скрипт:  
   `bash train_with_cache.sh <person> [trigger_token]`

Скрипт `train_with_cache.sh` создаёт локальный Hugging Face cache в `.cache/huggingface` (можно переопределить переменными `HF_HOME`/`HF_HUB_CACHE`) и прокидывает их в `python scripts/train_lora.py`. Подробнее о структуре папок — в `data/README.md` и `models/README.md`.

## Настройка железа и low-memory загрузка

- В `config.yaml` появилась секция `hardware`. По умолчанию она рассчитана на машину с RTX 4090 (24 ГБ VRAM) и ~40 ГБ системной RAM.  
- Параметры `max_cpu_ram_gb` и `max_vram_gb` пробрасываются в загрузчик `load_qwen_components`, который собирает только нужные блоки Qwen-Image (transformer, text encoder, VAE, tokenizer) через `low_cpu_mem_usage` и `device_map`. Остальные части пайплайна остаются на CPU или выгружаются обратно после использования.  
- `dataloader_workers`, `prefetch_factor` и `max_batch_size` позволяют ограничить число воркеров и автоматически уменьшать batch size, если видеопамяти недостаточно.  
- Если железо слабее, скорректируйте эти значения и перезапустите обучение. Для более мощных машин достаточно увеличить `max_batch_size` и, при необходимости, снять ограничение на workers.

## Документация

Подробная документация находится в директории `docs/`.
