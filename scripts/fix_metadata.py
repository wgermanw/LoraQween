"""Фикс metadata.jsonl: объединяет многострочные записи в одну строку и нормализует имена файлов под фактические из images/."""

import json
import re
from pathlib import Path
from typing import List, Dict


def normalize_name(name: str) -> str:
    # image (N).jpg -> _image_ (N).jpg
    m = re.match(r'^image \((\d+)\)\.jpg$', name, flags=re.IGNORECASE)
    if m:
        return f"_image_ ({m.group(1)}).jpg"
    # image.jpg -> _image_.jpg
    if name.lower() == "image.jpg":
        return "_image_.jpg"
    # image_1 (N).png -> _image_1_ (N).png
    m = re.match(r'^image_1 \((\d+)\)\.png$', name, flags=re.IGNORECASE)
    if m:
        return f"_image_1_ ({m.group(1)}).png"
    # image_3.png -> _image_3_.png
    if name.lower() == "image_3.png":
        return "_image_3_.png"
    return name


def read_metadata_loose(path: Path) -> List[Dict]:
    """Считать metadata.jsonl, поддерживая записи, разбитые на 2 строки."""
    rows: List[Dict] = []
    buf = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            # Если строка содержит и file_name, и caption и обе скобки - пытаемся сразу парсить
            if '"file_name"' in line and '"caption"' in line and line.strip().startswith("{") and line.strip().endswith("}"):
                rows.append(json.loads(line))
                continue
            # Иначе накапливаем до закрывающей скобки
            buf.append(line)
            if line.strip().endswith("}"):
                merged = " ".join(buf)
                buf = []
                # Удаляем лишние перевод строки; пробуем парсить
                # Приводим к валидному JSON (заменим возможные хвостовые запятые перед } не требуется тут)
                try:
                    rows.append(json.loads(merged))
                except json.JSONDecodeError:
                    # Пробуем привести к формату {"file_name": "...", "caption": "..."}
                    fn = re.search(r'"file_name"\s*:\s*"([^"]+)"', merged)
                    cp = re.search(r'"caption"\s*:\s*"(.+)"\s*}', merged)
                    if not (fn and cp):
                        raise
                    file_name = fn.group(1)
                    caption = cp.group(1)
                    rows.append({"file_name": file_name, "caption": caption})
    return rows


def main() -> int:
    root = Path(__file__).parent.parent
    dataset_dir = root / "data" / "datasets" / "Mikassa"
    images_dir = dataset_dir / "images"
    meta_path = dataset_dir / "metadata.jsonl"

    assert images_dir.exists(), f"images dir not found: {images_dir}"
    assert meta_path.exists(), f"metadata.jsonl not found: {meta_path}"

    # Считать метаданные "вольно"
    records = read_metadata_loose(meta_path)

    # Нормализовать имена
    image_names = {p.name for p in images_dir.iterdir() if p.is_file()}
    fixed: List[Dict] = []
    missing: List[str] = []

    for rec in records:
        file_name = normalize_name(rec["file_name"])
        if file_name not in image_names:
            # Если исходное (ненормализованное) имя существует, используем его
            if rec["file_name"] in image_names:
                file_name = rec["file_name"]
            else:
                missing.append(f"{rec['file_name']} -> {file_name}")
        fixed.append({"file_name": file_name, "caption": rec["caption"]})

    # Перезаписать metadata.jsonl в одном-строчном формате
    with meta_path.open("w", encoding="utf-8", newline="\n") as f:
        for rec in fixed:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if missing:
        print("ВНИМАНИЕ: найдены записи без соответствующего файла (после нормализации):")
        for m in missing:
            print(" -", m)
    else:
        print("metadata.jsonl успешно нормализован. Все имена соответствуют файлам или сохранены исходными.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

