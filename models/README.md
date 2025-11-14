# Models Directory

Каталог `models/` содержит крупные артефакты, которые не коммитятся в репозиторий. Ожидаемая структура:

```
models/
├── base/             # Скачанная модель Qwen/Qwen-Image (Diffusers)
├── loras/<person>/   # Результаты обучения LoRA
└── face_id/          # Модели Face-ID / IP-Adapter (опционально)
```

## Как заполнить

1. Скачайте базовую модель:
   ```bash
   python scripts/download_qwen_image.py --dtype bf16 --output_dir models/base
   ```
2. (Опционально) Скачайте модели Face-ID или IP-Adapter и положите их в `models/face_id/`.
3. После обучения LoRA чекпоинты появятся в `models/loras/<person>/`.

Вы можете настроить другие пути через `config.yaml` → секция `paths`.
