# Инструкция по запуску обучения

## ⚠️ ВАЖНО: Проблема с местом на диске

На диске C недостаточно места. Нужно использовать диск D для кэша.

## ✅ Способ 1: Использовать bat-файл (РЕКОМЕНДУЕТСЯ)

Просто запустите:
```bash
train_with_cache.bat
```

Этот файл автоматически:
1. Установит кэш на диск D
2. Создаст необходимые папки
3. Запустит обучение

## ✅ Способ 2: В PowerShell вручную

```powershell
# Установить переменные окружения
$env:HF_HOME = "D:\huggingface_cache"
$env:HF_HUB_CACHE = "D:\huggingface_cache\hub"

# Создать папки
New-Item -ItemType Directory -Force -Path "D:\huggingface_cache\hub"

# Запустить обучение
python scripts/train_lora.py --person Mikassa
```

## ✅ Способ 3: Использовать уже скачанные файлы

Модель уже частично скачана в `models/base/cache/`. Можно использовать её напрямую, но это сложнее.

## Что происходит при обучении

1. Загружается модель Qwen-Image (~20-30 GB)
2. Применяется LoRA к transformer (в Qwen-Image используется transformer вместо unet)
3. Загружается датасет из `data/datasets/Mikassa/`
4. Запускается обучение (1-2 часа)

## После обучения

Обученная LoRA будет сохранена в:
```
models/loras/Mikassa/
```

Затем можно использовать для генерации:
```bash
python scripts/inference.py --person Mikassa --prompt "<qwn_mikassa> portrait" --mode fast
```




