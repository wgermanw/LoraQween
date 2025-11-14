"""Скрипт для решения проблемы с местом на диске."""

import os
import shutil
from pathlib import Path

print("="*70)
print("РЕШЕНИЕ ПРОБЛЕМЫ С МЕСТОМ НА ДИСКЕ")
print("="*70)

# Проверить место на диске C
cache_dir = Path("C:/Users/gerab/.cache/huggingface/hub")
disk = cache_dir.drive if cache_dir.exists() else "C:"
total, used, free = shutil.disk_usage(disk)

print(f"\nДиск {disk}:")
print(f"  Всего: {total / (1024**3):.2f} GB")
print(f"  Использовано: {used / (1024**3):.2f} GB")
print(f"  Свободно: {free / (1024**3):.2f} GB")
print(f"\nНужно для модели Qwen-Image: ~20-30 GB")

if free < 30 * (1024**3):
    print("\n⚠ НЕДОСТАТОЧНО МЕСТА!")
    print("\nВарианты решения:")
    print("\n1. ОЧИСТИТЬ КЭШ HUGGING FACE")
    print("   Удалить старые модели из кэша:")
    cache_hub = Path("C:/Users/gerab/.cache/huggingface/hub")
    if cache_hub.exists():
        models = [d for d in cache_hub.iterdir() if d.is_dir() and d.name.startswith("models--")]
        print(f"   Найдено моделей в кэше: {len(models)}")
        for model in models[:5]:
            size = sum(f.stat().st_size for f in model.rglob('*') if f.is_file())
            print(f"   - {model.name}: {size / (1024**3):.2f} GB")
    
    print("\n2. ИЗМЕНИТЬ ПУТЬ КЭША НА ДРУГОЙ ДИСК")
    print("   Установите переменную окружения:")
    print("   set HF_HOME=D:\\huggingface_cache")
    print("   или")
    print("   set HF_HOME=E:\\huggingface_cache  # если есть диск E с местом")
    
    print("\n3. ИСПОЛЬЗОВАТЬ УЖЕ СКАЧАННЫЕ ФАЙЛЫ")
    print("   Модель частично скачана в:")
    print("   models/base/cache/models--Qwen--Qwen-Image/")
    print("   Можно попробовать использовать локальный кэш")
    
    print("\n4. ОЧИСТИТЬ ВРЕМЕННЫЕ ФАЙЛЫ WINDOWS")
    print("   - Очистить корзину")
    print("   - Удалить временные файлы (Win+R -> %temp%)")
    print("   - Очистить загрузки")
else:
    print("\n✓ Места достаточно!")

