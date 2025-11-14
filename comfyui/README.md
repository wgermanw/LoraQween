# Интеграция с ComfyUI

Этот модуль содержит ноды и пресеты для интеграции LoraQween с ComfyUI.

## Установка

1. Скопируйте содержимое директории `nodes/` в `ComfyUI/custom_nodes/loraqween/`
2. Установите зависимости через менеджер пакетов ComfyUI или вручную

## Использование

### Пресеты

Пресеты находятся в директории `presets/`:
- `fast_mode.json` - Быстрый режим генерации
- `reliable_mode.json` - Надёжный режим с FaceID/IP-Adapter

### Ноды

- `LoadLoraQweenLoRA` - Загрузка LoRA для персоны
- `LoadLoraQweenTokenizer` - Загрузка токенайзера с триггер-токенами
- `LoraQweenFaceID` - Применение FaceID/IP-Adapter
- `LoraQweenQualityCheck` - Проверка качества генерации

## Примечания

⚠️ Полная интеграция требует установки ComfyUI и соответствующих зависимостей.

