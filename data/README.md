# Data Directory

Данные не хранятся в репозитории по умолчанию (см. `.gitignore`). Перед запуском обучения создайте следующую структуру:

```
data/
├── input_photos/<person>/        # Сырые фото, которые вы собираетесь обучать
├── datasets/<person>/            # Готовый датасет (генерируется prepare_data.py)
│   ├── images/*.jpg
│   ├── metadata.jsonl
│   ├── manifest.json
│   └── tokenizer/                # Токенайзер с добавленным триггер-токеном
├── references/                   # Опциональные эталонные фото
└── regularization/               # Опциональные регуляризационные выборки
```

Быстрый сценарий:
1. Скопируйте исходные фото в `data/input_photos/<person>/`.
2. Запустите `python scripts/auto_prepare_data.py --person <person> [...]`.
3. Датасет появится в `data/datasets/<person>/` и будет готов для `scripts/train_lora.py`.
