"""Установить путь кэша Hugging Face на диск D."""

import os
from pathlib import Path

# Установить переменную окружения для текущей сессии
cache_path = Path("D:/huggingface_cache")
cache_path.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(cache_path)
os.environ['HF_HUB_CACHE'] = str(cache_path / "hub")

print("="*70)
print("НАСТРОЙКА ПУТИ КЭША HUGGING FACE")
print("="*70)
print(f"\n✓ Кэш установлен на: {cache_path}")
print(f"✓ HF_HOME = {os.environ['HF_HOME']}")
print(f"✓ HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")

print("\n" + "="*70)
print("ВАЖНО: Для постоянной установки выполните в PowerShell:")
print("="*70)
print(f'$env:HF_HOME = "{cache_path}"')
print(f'$env:HF_HUB_CACHE = "{cache_path / "hub"}"')
print("\nИли добавьте в системные переменные окружения Windows")

