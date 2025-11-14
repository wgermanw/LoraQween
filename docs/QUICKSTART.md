# Быстрый старт

## 1. Подготовка данных

Соберите 40-80 фотографий персоны и поместите их в одну директорию.

```bash
python scripts/prepare_data.py \
    --person alex \
    --input_dir /path/to/photos \
    --gender male \
    --age 30 \
    --hair_color brown
```

Это создаст:
- Обработанные изображения в `data/datasets/alex/images/`
- Метаданные в `data/datasets/alex/metadata.jsonl`
- Токенайзер с триггер-токеном в `data/datasets/alex/tokenizer/`

## 2. Обучение LoRA

```bash
python scripts/train_lora.py --person alex
```

⚠️ **ВАЖНО**: Обучение требует загруженной модели Qwen-Image 20B. Убедитесь, что модель находится в `models/base/`.

## 3. Генерация изображений

### Быстрый режим

```bash
python scripts/inference.py \
    --person alex \
    --prompt "<qwn_alex> portrait, smiling, outdoor lighting" \
    --mode fast \
    --num_images 4
```

### Надёжный режим

```bash
python scripts/inference.py \
    --person alex \
    --prompt "<qwn_alex> portrait, professional photo" \
    --mode reliable \
    --num_images 2 \
    --reference /path/to/reference.jpg
```

## 4. Контроль качества

```bash
python scripts/quality_control.py \
    --person alex \
    --generated_dir outputs/alex/generation_20240101_120000 \
    --reference_dir data/references/alex \
    --output reports/quality_report.json
```

## Структура проекта после подготовки

```
LoraQween/
├── data/
│   └── datasets/
│       └── alex/
│           ├── images/          # Обработанные изображения
│           ├── metadata.jsonl  # Метаданные с подписями
│           ├── manifest.json   # Манифест персоны
│           └── tokenizer/       # Токенайзер с триггер-токеном
├── models/
│   ├── base/                   # Базовая модель Qwen-Image
│   └── loras/
│       └── alex/               # Обученная LoRA
└── outputs/
    └── alex/                   # Результаты генерации
```

## Следующие шаги

- Изучите полную документацию в `docs/`
- Настройте параметры в `config.yaml`
- Интегрируйте с ComfyUI (см. `comfyui/README.md`)

