"""Проверка содержимого checkpoint файла."""

import sys
from pathlib import Path

try:
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("safetensors не установлен. Установите: pip install safetensors")

def inspect_checkpoint(checkpoint_path: Path):
    """Проверить содержимое checkpoint файла."""
    if not SAFETENSORS_AVAILABLE:
        print("safetensors не доступен")
        return
    
    if not checkpoint_path.exists():
        print(f"Файл не найден: {checkpoint_path}")
        return
    
    print(f"Проверка checkpoint: {checkpoint_path.name}")
    print(f"Размер: {checkpoint_path.stat().st_size / (1024**3):.2f} GB\n")
    
    try:
        with safe_open(str(checkpoint_path), framework='pt') as f:
            keys = list(f.keys())
            
            print(f"Всего ключей в checkpoint: {len(keys)}\n")
            print("Первые 30 ключей:")
            for i, key in enumerate(keys[:30]):
                print(f"  {i+1}. {key}")
            
            if len(keys) > 30:
                print(f"  ... и ещё {len(keys) - 30} ключей\n")
            
            # Поиск компонентов
            unet_keys = [k for k in keys if any(x in k.lower() for x in ['model.diffusion_model', 'unet', 'mmdit'])]
            te_keys = [k for k in keys if any(x in k.lower() for x in ['text_encoder', 'transformer.text_model', 'clip'])]
            vae_keys = [k for k in keys if any(x in k.lower() for x in ['vae', 'first_stage_model'])]
            
            print("\n" + "="*60)
            print("АНАЛИЗ КОМПОНЕНТОВ")
            print("="*60)
            print(f"\nUNet/MMDiT ключей: {len(unet_keys)}")
            if unet_keys:
                print("  Примеры:")
                for k in unet_keys[:5]:
                    print(f"    - {k}")
            
            print(f"\nText Encoder ключей: {len(te_keys)}")
            if te_keys:
                print("  Примеры:")
                for k in te_keys[:5]:
                    print(f"    - {k}")
            
            print(f"\nVAE ключей: {len(vae_keys)}")
            if vae_keys:
                print("  Примеры:")
                for k in vae_keys[:5]:
                    print(f"    - {k}")
            
            # Вывод
            print("\n" + "="*60)
            print("ВЫВОД")
            print("="*60)
            
            if unet_keys and te_keys and vae_keys:
                print("✓ Checkpoint содержит все необходимые компоненты!")
                print("  - UNet/MMDiT")
                print("  - Text Encoder")
                print("  - VAE")
                print("\n⚠ Но для обучения LoRA через diffusers нужна модель")
                print("  в формате diffusers (с отдельными директориями)")
                print("\nВарианты:")
                print("1. Конвертировать checkpoint в формат diffusers")
                print("2. Использовать модель через ComfyUI")
                print("3. Скачать оригинальную модель Qwen-Image в формате diffusers")
            else:
                missing = []
                if not unet_keys:
                    missing.append("UNet/MMDiT")
                if not te_keys:
                    missing.append("Text Encoder")
                if not vae_keys:
                    missing.append("VAE")
                
                print(f"✗ В checkpoint отсутствуют компоненты: {', '.join(missing)}")
                
    except Exception as e:
        print(f"Ошибка при чтении checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    checkpoint_path = Path("models/base/Qwen-Image-Edit-Rapid-AIO/v10/Qwen-Rapid-AIO-NSFW-v10.4.safetensors")
    
    if len(sys.argv) > 1:
        checkpoint_path = Path(sys.argv[1])
    
    inspect_checkpoint(checkpoint_path)

