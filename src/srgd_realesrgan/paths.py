from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Iterable

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def iter_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    return sorted(
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sanitize_id(text: str, max_len: int = 80) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return clean[:max_len] or "image"


def read_pairs_csv(
    csv_path: str | Path,
    *,
    split: str | None = None,
    limit: int | None = None,
    shuffle: bool = False,
    seed: int = 0,
) -> list[dict[str, str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if split:
        split_lower = split.lower()
        rows = [row for row in rows if row.get("split", "").lower() == split_lower]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)

    if limit is not None:
        rows = rows[:limit]

    return rows


def write_csv(path: str | Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    output = Path(path)
    ensure_dir(output.parent)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
