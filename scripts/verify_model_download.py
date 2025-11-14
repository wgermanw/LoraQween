"""Проверка скачанной модели Qwen-Image."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config

def check_model():
    """Проверить наличие модели."""
    print("="*70)
    print("ПРОВЕРКА СКАЧАННОЙ МОДЕЛИ QWEN-IMAGE")
    print("="*70)
    
    config = get_config()
    base_dir = config.paths['base_model_dir']
    
    # Проверить кэш Hugging Face
    cache_dir = base_dir / "cache" / "models--Qwen--Qwen-Image" / "snapshots"
    
    if cache_dir.exists():
        snapshots = list(cache_dir.iterdir())
        if snapshots:
            snapshot_dir = snapshots[0]
            print(f"\n✓ Модель найдена в кэше: {snapshot_dir.name}")
            
            # Проверить model_index.json
            model_index = snapshot_dir / "model_index.json"
            if model_index.exists():
                print("✓ model_index.json найден")
                with open(model_index, 'r') as f:
                    index_data = json.load(f)
                    print(f"  Компоненты: {list(index_data.keys())}")
            else:
                print("✗ model_index.json не найден")
            
            # Проверить компоненты
            components = ['unet', 'text_encoder', 'vae', 'tokenizer']
            print("\nПроверка компонентов:")
            for comp in components:
                comp_dir = snapshot_dir / comp
                if comp_dir.exists():
                    files = list(comp_dir.glob("*"))
                    print(f"  ✓ {comp}/: {len(files)} файлов")
                else:
                    print(f"  ✗ {comp}/: не найдена")
        else:
            print("✗ Снимки модели не найдены в кэше")
    else:
        print("✗ Кэш модели не найден")
    
    # Проверить локальную копию
    print("\n" + "="*70)
    print("ПРОВЕРКА ЛОКАЛЬНОЙ КОПИИ")
    print("="*70)
    
    components = ['unet', 'text_encoder', 'vae', 'tokenizer']
    all_found = True
    
    for comp in components:
        comp_dir = base_dir / comp
        if comp_dir.exists():
            files = list(comp_dir.glob("*"))
            print(f"✓ {comp}/: {len(files)} файлов")
        else:
            print(f"✗ {comp}/: не найдена")
            all_found = False
    
    model_index = base_dir / "model_index.json"
    if model_index.exists():
        print("✓ model_index.json найден локально")
    else:
        print("⚠ model_index.json не найден локально (нормально, если модель в кэше)")
    
    # Итог
    print("\n" + "="*70)
    print("ИТОГ")
    print("="*70)
    
    if all_found:
        print("✅ Все компоненты найдены локально!")
        print("   Модель готова к использованию")
    else:
        print("⚠ Модель найдена в кэше Hugging Face, но не сохранена локально")
        print("\nРекомендации:")
        print("1. Модель будет использоваться из кэша автоматически")
        print("2. Или сохраните модель локально:")
        print("   python scripts/download_qwen_image.py")
        print("   (без флага --check_only)")
    
    return all_found

if __name__ == "__main__":
    success = check_model()
    sys.exit(0 if success else 1)

