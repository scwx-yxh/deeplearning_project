from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from srgd_realesrgan.paths import IMAGE_EXTENSIONS, ensure_dir, read_pairs_csv


def parse_method(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Methods must use NAME=PATH format.")
    name, path = value.split("=", 1)
    return name, Path(path)


def find_image(folder: Path, pair_id: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = folder / f"{pair_id}{ext}"
        if candidate.exists():
            return candidate
    matches = []
    for ext in IMAGE_EXTENSIONS:
        matches.extend(folder.glob(f"{pair_id}*{ext}"))
        matches.extend(folder.glob(f"{pair_id}*{ext.upper()}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No image found for pair_id={pair_id} in {folder}")


def resize_for_cell(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def labeled_cell(image: Image.Image, label: str, width: int, label_h: int) -> Image.Image:
    resized = resize_for_cell(image.convert("RGB"), width)
    canvas = Image.new("RGB", (width, resized.height + label_h), "white")
    canvas.paste(resized, (0, label_h))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width, label_h), fill=(24, 24, 24))
    draw.text((8, 7), label, fill=(255, 255, 255), font=font)
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create LR / method / HR visual comparison grids.")
    parser.add_argument("--pairs", type=Path, required=True, help="Pairs CSV from prepare_pairs.py.")
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_method,
        default=[],
        help="Method output folders in NAME=PATH format.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder for comparison grids.")
    parser.add_argument("--split", default=None, help="Optional split filter: train, val, or test.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of grids.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle.")
    parser.add_argument("--cell-width", type=int, default=320, help="Display width per image.")
    parser.add_argument("--label-height", type=int, default=28, help="Top label band height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_pairs_csv(args.pairs, split=args.split, limit=args.limit, shuffle=args.shuffle, seed=args.seed)
    out_dir = ensure_dir(args.out_dir)

    for row in tqdm(rows, desc="Make grids"):
        cells = [labeled_cell(Image.open(row["lr_path"]), "LR", args.cell_width, args.label_height)]
        for method_name, method_dir in args.methods:
            method_image = Image.open(find_image(method_dir, row["pair_id"]))
            cells.append(labeled_cell(method_image, method_name, args.cell_width, args.label_height))
        cells.append(labeled_cell(Image.open(row["hr_path"]), "HR", args.cell_width, args.label_height))

        max_height = max(cell.height for cell in cells)
        grid = Image.new("RGB", (args.cell_width * len(cells), max_height), "white")
        for index, cell in enumerate(cells):
            grid.paste(cell, (index * args.cell_width, 0))
        grid.save(out_dir / f"{row['pair_id']}_grid.png")

    print(f"Wrote visual grids to {out_dir}")


if __name__ == "__main__":
    main()
