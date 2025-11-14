"""Финальная проверка скачанной модели."""

from pathlib import Path

print("="*70)
print("ФИНАЛЬНАЯ ПРОВЕРКА МОДЕЛИ QWEN-IMAGE")
print("="*70)

base_dir = Path("models/base")
cache_dir = base_dir / "cache" / "models--Qwen--Qwen-Image" / "snapshots"
snapshots = list(cache_dir.iterdir()) if cache_dir.exists() else []

if snapshots:
    snapshot = snapshots[0]
    print(f"\n✓ Модель найдена в кэше: {snapshot.name}\n")
    
    components = {
        'transformer': 'UNet/MMDiT',
        'text_encoder': 'Text Encoder',
        'vae': 'VAE',
        'tokenizer': 'Tokenizer',
        'scheduler': 'Scheduler'
    }
    
    all_found = True
    for comp_dir, comp_name in components.items():
        path = snapshot / comp_dir
        if path.exists():
            files = list(path.glob("*.safetensors")) + list(path.glob("*.json")) + list(path.glob("*.txt")) + list(path.glob("*.jinja"))
            print(f"✓ {comp_name} ({comp_dir}/): {len(files)} файлов")
            if comp_dir == 'transformer':
                safetensors = [f for f in files if f.suffix == '.safetensors']
                print(f"    → {len(safetensors)} файлов весов")
            elif comp_dir == 'text_encoder':
                safetensors = [f for f in files if f.suffix == '.safetensors']
                print(f"    → {len(safetensors)} файлов весов")
        else:
            print(f"✗ {comp_name} ({comp_dir}/): не найден")
            all_found = False
    
    print("\n" + "="*70)
    if all_found:
        print("✅ ВСЕ КОМПОНЕНТЫ НАЙДЕНЫ!")
        print("\nМодель готова к использованию!")
        print("\nПримечание: Модель находится в кэше Hugging Face.")
        print("Diffusers автоматически использует её оттуда при загрузке.")
        print("\nДля использования укажите в config.yaml:")
        print('  base_model:')
        print('    name: "Qwen/Qwen-Image"')
    else:
        print("⚠ Некоторые компоненты отсутствуют")
else:
    print("✗ Модель не найдена в кэше")

