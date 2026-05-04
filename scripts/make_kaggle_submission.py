from __future__ import annotations

import argparse
import base64
import csv
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]


def encode_image_bgr(image: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(image, dtype=np.uint8).ravel()
    if flat.size == 0:
        return base64.b64encode(zlib.compress(b"", zlib.Z_BEST_COMPRESSION))

    change_positions = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], change_positions))
    ends = np.concatenate((change_positions, [flat.size]))
    values = flat[starts]
    lengths = ends - starts

    pieces_per_run = (lengths + 254) // 255
    if int(pieces_per_run.max()) == 1:
        rle = np.empty(values.size * 2, dtype=np.uint8)
        rle[0::2] = values
        rle[1::2] = lengths.astype(np.uint8)
    else:
        split_values = np.repeat(values, pieces_per_run)
        split_counts = np.full(int(pieces_per_run.sum()), 255, dtype=np.uint8)
        last_piece_indices = np.cumsum(pieces_per_run) - 1
        split_counts[last_piece_indices] = (lengths - 255 * (pieces_per_run - 1)).astype(np.uint8)
        rle = np.empty(split_values.size * 2, dtype=np.uint8)
        rle[0::2] = split_values
        rle[1::2] = split_counts

    compressed = zlib.compress(rle.tobytes(), zlib.Z_BEST_COMPRESSION)
    return base64.b64encode(compressed)


def find_image(images_dir: Path, filename: str) -> Path:
    exact = images_dir / filename
    if exact.exists():
        return exact

    stem = Path(filename).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find an SR image for {filename} in {images_dir}")


def read_image_bgr(path: str | Path) -> np.ndarray:
    if cv2 is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return image

    image_rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return image_rgb[..., ::-1]


def rows_from_sample(sample_submission: Path, images_dir: Path, id_column: str, filename_column: str) -> list[dict[str, str]]:
    with sample_submission.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            filename = row[filename_column]
            rows.append(
                {
                    "id": row.get(id_column, str(len(rows))),
                    "filename": filename,
                    "image_path": find_image(images_dir, filename).as_posix(),
                }
            )
    return rows


def rows_from_images(images_dir: Path) -> list[dict[str, str]]:
    images = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return [
        {
            "id": str(index),
            "filename": path.name,
            "image_path": path.as_posix(),
        }
        for index, path in enumerate(images)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode super-resolution outputs for the SRGD Kaggle submission format.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Folder containing generated HR/SR images.")
    parser.add_argument("--out", type=Path, default=Path("submission.csv"), help="Output submission CSV.")
    parser.add_argument("--sample-submission", type=Path, default=None, help="Optional Kaggle sample_submission.csv.")
    parser.add_argument("--id-column", default="id", help="ID column name in sample_submission.csv.")
    parser.add_argument("--filename-column", default="filename", help="Filename column name in sample_submission.csv.")
    parser.add_argument("--rle-column", default="rle", help="Encoded image column name for the output CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    rows = (
        rows_from_sample(args.sample_submission, images_dir, args.id_column, args.filename_column)
        if args.sample_submission
        else rows_from_images(images_dir)
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[args.id_column, args.filename_column, args.rle_column])
        writer.writeheader()
        for row in rows:
            image = read_image_bgr(row["image_path"])
            writer.writerow(
                {
                    args.id_column: row["id"],
                    args.filename_column: row["filename"],
                    args.rle_column: str(encode_image_bgr(image)),
                }
            )

    print(f"Wrote {len(rows)} encoded predictions to {args.out}")


if __name__ == "__main__":
    main()
