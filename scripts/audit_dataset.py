"""Аудит датасета персоны: структура, метаданные, размеры и бакеты."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple, Set

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config


def reduce_ratio(width: int, height: int) -> str:
    """Свести соотношение сторон к простому виду, например 3000x2000 -> 3x2."""
    from math import gcd
    if width <= 0 or height <= 0:
        return f"{width}x{height}"
    g = gcd(width, height)
    return f"{width // g}x{height // g}"


def read_metadata(metadata_file: Path) -> List[Dict]:
    records: List[Dict] = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON ошибка в строке {idx}: {e}") from e
            # Базовая валидация полей
            if "file_name" not in obj or "caption" not in obj:
                raise ValueError(f"Отсутствуют обязательные поля в строке {idx}: {line}")
            records.append(obj)
    return records


def collect_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    files: List[Path] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def get_image_info(image_path: Path) -> Tuple[int, int, str]:
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height, reduce_ratio(width, height)


def audit_dataset(person: str) -> None:
    config = get_config()
    paths = config.paths

    dataset_dir = paths["datasets_dir"] / person
    images_dir = dataset_dir / "images"
    metadata_file = dataset_dir / "metadata.jsonl"
    buckets_file = dataset_dir / "buckets.json"
    tokenizer_dir = dataset_dir / "tokenizer"

    print("=== Структура датасета ===")
    print(f"dataset_dir: {dataset_dir}")
    print(f"images/: {'OK' if images_dir.exists() else 'MISSING'}")
    print(f"metadata.jsonl: {'OK' if metadata_file.exists() else 'MISSING'}")
    print(f"tokenizer/: {'OK' if tokenizer_dir.exists() else 'MISSING'}")
    print(f"buckets.json: {'OK' if buckets_file.exists() else 'MISSING'}")

    if not (images_dir.exists() and metadata_file.exists() and tokenizer_dir.exists()):
        print("❌ Критические элементы структуры отсутствуют. Прервёмся.")
        return

    # Собрать файлы изображений
    image_files = collect_images(images_dir)
    ext_counter = Counter(p.suffix.lower() for p in image_files)
    total_images = len(image_files)

    print("\n=== Картинки в images/ ===")
    print(f"Всего файлов: {total_images}")
    print("Расширения: " + ", ".join(f"{ext}: {cnt}" for ext, cnt in sorted(ext_counter.items())))

    # Прочитать метаданные
    print("\n=== metadata.jsonl ===")
    try:
        records = read_metadata(metadata_file)
    except ValueError as e:
        print(f"❌ Ошибка в metadata.jsonl: {e}")
        return

    print(f"Количество записей (валидных JSON): {len(records)}")

    metadata_names: List[str] = [r["file_name"] for r in records]
    caption_texts: List[str] = [r["caption"] for r in records]

    # Дубликаты записей
    name_counter = Counter(metadata_names)
    duplicates = sorted([name for name, cnt in name_counter.items() if cnt > 1])

    # Сопоставление с физическими файлами
    image_name_set: Set[str] = {p.name for p in image_files}
    metadata_name_set: Set[str] = set(metadata_names)

    missing_files = sorted([name for name in metadata_name_set if name not in image_name_set])
    extra_files = sorted([name for name in image_name_set if name not in metadata_name_set])

    if duplicates:
        print("Дубликаты в metadata.jsonl:", ", ".join(duplicates))
    else:
        print("Дубликаты в metadata.jsonl: нет")

    if missing_files:
        print("Записи без файла на диске:", ", ".join(missing_files))
    else:
        print("Записей без файла на диске: нет")

    if extra_files:
        print("Файлы без записи в metadata.jsonl:", ", ".join(extra_files))
    else:
        print("Файлов без записи в metadata.jsonl: нет")

    # Единообразие caption и наличие триггер-токена
    token = f"<qwn_{person.lower()}>"
    with_token = sum(1 for c in caption_texts if token in c)
    print("\n=== Caption / триггер-токен ===")
    print(f"С токеном '{token}': {with_token}/{len(caption_texts)}")
    sample_captions = caption_texts[:2] if caption_texts else []
    for idx, cap in enumerate(sample_captions, start=1):
        print(f"Пример {idx}: {cap}")

    # Оценка разнообразия по ключевым словам
    keywords = ["portrait", "full body", "full-body", "close-up", "profile", "side view", "upper body"]
    kw_counts = {kw: sum(1 for c in caption_texts if kw in c.lower()) for kw in keywords}
    print("\nКлючевые слова в caption: " + ", ".join(f"{k}: {v}" for k, v in kw_counts.items()))

    # Чтение изображений, размеры и пропорции
    print("\n=== Размеры изображений и пропорции ===")
    sizes: List[Tuple[int, int]] = []
    ratio_counter: Counter = Counter()
    broken: List[str] = []

    for p in image_files:
        try:
            w, h, ratio = get_image_info(p)
            sizes.append((w, h))
            ratio_counter[ratio] += 1
        except Exception:
            broken.append(p.name)

    if broken:
        print("Проблемные/нечитаемые файлы:", ", ".join(broken))
    else:
        print("Все изображения открываются без ошибок")

    if sizes:
        widths = [w for w, _ in sizes]
        heights = [h for _, h in sizes]
        min_w, min_h = min(widths), min(heights)
        max_w, max_h = max(widths), max(heights)
        avg_w, avg_h = int(mean(widths)), int(mean(heights))
        print(f"Мин. разрешение: {min_w}x{min_h}")
        print(f"Макс. разрешение: {max_w}x{max_h}")
        print(f"Среднее разрешение: {avg_w}x{avg_h}")
        print("Соотношения сторон: " + ", ".join(f"{r}:{cnt}" for r, cnt in sorted(ratio_counter.items())))
    else:
        print("Не удалось определить размеры изображений")

    # Проверка buckets.json
    if buckets_file.exists():
        try:
            buckets = json.loads(buckets_file.read_text(encoding="utf-8"))
            print("\n=== buckets.json ===")
            # buckets может быть либо dict ratio->count, либо подробная структура; обработаем типичный случай
            if isinstance(buckets, dict) and all(isinstance(v, int) for v in buckets.values()):
                total_from_buckets = sum(buckets.values())
                print("Бакеты:", ", ".join(f"{k}: {v}" for k, v in sorted(buckets.items())))
                print(f"Сумма по бакетам: {total_from_buckets} (ожидалось {total_images})")
                if total_from_buckets != total_images:
                    print("❌ Несоответствие: сумма бакетов не равна количеству изображений")
                # Сравнить с фактическими пропорциями, если они посчитаны
                if ratio_counter:
                    # Нормализуем ключи вида "3x4"
                    mismatches: List[str] = []
                    for k, v in buckets.items():
                        count_real = ratio_counter.get(k, 0)
                        if count_real != v:
                            mismatches.append(f"{k}: buckets={v}, real={count_real}")
                    if mismatches:
                        print("Несоответствия бакетов фактическим пропорциям: " + "; ".join(mismatches))
                    else:
                        print("Бакеты соответствуют фактическим пропорциям изображений (по агрегатам)")
            else:
                print("buckets.json имеет нестандартный формат — пропускаю подробную проверку")
        except Exception as e:
            print(f"Не удалось прочитать buckets.json: {e}")

    # Проверка наличия токена в токенайзере
    print("\n=== Tokenizer ===")
    added_tokens_path = tokenizer_dir / "added_tokens.json"
    tokenizer_cfg_path = tokenizer_dir / "tokenizer_config.json"

    token_found = False
    if added_tokens_path.exists():
        try:
            added_tokens = json.loads(added_tokens_path.read_text(encoding="utf-8"))
            token_found = token in added_tokens
        except Exception:
            pass

    if not token_found and tokenizer_cfg_path.exists():
        try:
            tok_cfg = json.loads(tokenizer_cfg_path.read_text(encoding="utf-8"))
            # Поиск в added_tokens_decoder
            added_decoder = tok_cfg.get("added_tokens_decoder", {})
            for entry in added_decoder.values():
                if isinstance(entry, dict) and entry.get("content") == token:
                    token_found = True
                    break
        except Exception:
            pass

    print(f"Токен '{token}': {'найден' if token_found else 'не найден'} в токенайзере")


def main() -> int:
    parser = argparse.ArgumentParser(description="Аудит датасета для LoRA")
    parser.add_argument("--person", required=True, type=str, help="Имя персоны (папка в data/datasets)")
    args = parser.parse_args()

    audit_dataset(args.person)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

