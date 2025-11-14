# Настройка модели Qwen-Image

## ✅ Рекомендуемый подход (совет друга)

**Используйте официальные diffusers-веса `Qwen/Qwen-Image`** - это самый простой и надёжный способ!

### Преимущества:
1. ✅ **Одна строка загрузки**: `DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")`
2. ✅ **Автоматическая структура**: все компоненты (UNet, text_encoder, VAE, tokenizer) уже разделены
3. ✅ **Совместимость**: работает напрямую с нашим тренером без конвертации
4. ✅ **Без потери метаданных**: все метаданные сохраняются
5. ✅ **Обновления**: автоматический доступ к обновлениям модели

### Скачивание модели

```bash
# Проверить доступность
python scripts/download_qwen_image.py --check_only

# Скачать модель (займёт время и место ~20-30 GB)
python scripts/download_qwen_image.py

# Или указать другую директорию
python scripts/download_qwen_image.py --output_dir /path/to/models
```

### Использование

После скачивания модель будет доступна через:
- **Hugging Face ID**: `Qwen/Qwen-Image` (скачивается автоматически при первом использовании)
- **Локальный путь**: `models/base` (если скачали локально)

Обновите `config.yaml`:
```yaml
base_model:
  name: "Qwen/Qwen-Image"  # или "models/base" для локальной модели
  dtype: "fp16"
```

## Альтернативный подход (ComfyUI checkpoint)

Если у вас уже есть ComfyUI checkpoint (`Qwen-Rapid-AIO-NSFW-v10.4.safetensors`):

1. **Оставьте его для инференса** через ComfyUI
2. **Для обучения LoRA** используйте официальные diffusers-веса

Это разделение:
- ✅ Обучение: надёжный diffusers формат
- ✅ Инференс: оптимизированный ComfyUI checkpoint

## Структура после скачивания

```
models/base/
├── unet/              # Веса UNet/MMDiT
│   ├── diffusion_pytorch_model.safetensors
│   └── config.json
├── text_encoder/      # Текстовый энкодер
│   ├── model.safetensors
│   └── config.json
├── vae/               # VAE
│   ├── diffusion_pytorch_model.safetensors
│   └── config.json
├── tokenizer/         # Токенайзер
│   ├── tokenizer.json
│   └── ...
└── model_index.json   # Индекс модели
```

## Проверка установки

```bash
# Проверить компоненты
python scripts/check_qwen_components.py

# Проверить все модели
python scripts/check_models.py
```

## Экономия VRAM

Для RTX 5060 Ti 16 GB рекомендуется:
- Использовать `fp16` или `bf16` dtype
- Включить gradient checkpointing
- Использовать batch_size=1 с gradient_accumulation_steps=4

Для более агрессивной экономии VRAM можно использовать квантизацию (fp8/q4), но это требует дополнительных библиотек.

