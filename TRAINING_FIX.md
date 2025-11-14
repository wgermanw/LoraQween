# Решение проблемы с местом на диске при обучении

## Проблема
На диске C осталось только 0.42 GB, а для модели Qwen-Image нужно ~20-30 GB.

## ✅ Решение 1: Изменить путь кэша на диск D (РЕКОМЕНДУЕТСЯ)

### В PowerShell (перед запуском обучения):

```powershell
# Установить кэш на диск D
$env:HF_HOME = "D:\huggingface_cache"
$env:HF_HUB_CACHE = "D:\huggingface_cache\hub"

# Затем запустить обучение
python scripts/train_lora.py --person Mikassa
```

### Или создать файл `set_cache.bat`:

```batch
@echo off
set HF_HOME=D:\huggingface_cache
set HF_HUB_CACHE=D:\huggingface_cache\hub
python scripts/train_lora.py --person Mikassa
```

## ✅ Решение 2: Использовать локальный кэш

Модель уже частично скачана в `models/base/cache/`. Можно использовать её:

```bash
# Обновить config.yaml чтобы использовать локальный путь
# base_model:
#   name: "models/base/cache/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"
```

Но это сложнее, лучше использовать Решение 1.

## ✅ Решение 3: Очистить место на диске C

1. Очистить корзину
2. Удалить временные файлы: `Win+R` -> `%temp%` -> удалить старые файлы
3. Очистить загрузки
4. Удалить старые модели из кэша Hugging Face (если не нужны)

## Рекомендация

**Используйте Решение 1** - это самое простое и надёжное. Просто выполните команды в PowerShell перед обучением.

