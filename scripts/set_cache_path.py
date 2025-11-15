
"""Configure the Hugging Face cache path inside the repository."""

import os
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
cache_path = (repo_root / "models" / "base" / "cache").resolve()
cache_path.mkdir(parents=True, exist_ok=True)
hub_cache = cache_path / "hub"
hub_cache.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(cache_path)
os.environ["HF_HUB_CACHE"] = str(hub_cache)

print("=" * 70)
print("НАСТРОЙКА ПУТИ КЭША HUGGING FACE")
print("=" * 70)
print(f"\n✓ Кэш установлен на: {cache_path}")
print(f"✓ HF_HOME = {os.environ['HF_HOME']}")
print(f"✓ HF_HUB_CACHE = {os.environ['HF_HUB_CACHE']}")

print("\n" + "=" * 70)
print("ВАЖНО: Для постоянной установки выполните в PowerShell:")
print("=" * 70)
print(f':HF_HOME = "{cache_path}"')
print(f':HF_HUB_CACHE = "{hub_cache}"')
print("\nЛибо настройте стандартный ~/.cache/huggingface, если предпочитаете системный путь.")
