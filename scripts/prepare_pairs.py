from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from srgd_realesrgan.paths import iter_images, relative_or_absolute, sanitize_id, write_csv


SPLIT_NAMES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}


def replace_token_case_insensitive(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


def detect_split(path: Path) -> str:
    for part in path.parts:
        split = SPLIT_NAMES.get(part.lower())
        if split:
            return split
    return "train"


def collect_explicit_pairs(lr_dir: Path, hr_dir: Path, lr_token: str, hr_token: str) -> list[tuple[Path, Path]]:
    hr_by_rel = {path.relative_to(hr_dir).as_posix().lower(): path for path in iter_images(hr_dir)}
    hr_by_name = {path.name.lower(): path for path in iter_images(hr_dir)}
    pairs: list[tuple[Path, Path]] = []

    for lr_path in iter_images(lr_dir):
        rel = lr_path.relative_to(lr_dir).as_posix()
        candidates = [
            rel,
            replace_token_case_insensitive(rel, lr_token, hr_token),
            lr_path.name,
            replace_token_case_insensitive(lr_path.name, lr_token, hr_token),
        ]
        hr_path = None
        for candidate in candidates:
            hr_path = hr_by_rel.get(candidate.lower()) or hr_by_name.get(candidate.lower())
            if hr_path:
                break
        if hr_path:
            pairs.append((lr_path, hr_path))
    return pairs


def collect_auto_pairs(data_root: Path, lr_token: str, hr_token: str) -> list[tuple[Path, Path]]:
    images = iter_images(data_root)
    image_lookup = {path.resolve().as_posix().lower(): path for path in images}
    pairs: list[tuple[Path, Path]] = []

    for lr_path in images:
        path_text = lr_path.resolve().as_posix()
        if lr_token.lower() not in path_text.lower():
            continue
        hr_text = replace_token_case_insensitive(path_text, lr_token, hr_token)
        hr_path = image_lookup.get(Path(hr_text).resolve().as_posix().lower())
        if hr_path:
            pairs.append((lr_path, hr_path))
    return pairs


def build_rows(pairs: list[tuple[Path, Path]], data_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, (lr_path, hr_path) in enumerate(sorted(pairs, key=lambda item: item[0].as_posix())):
        key = (lr_path.resolve().as_posix(), hr_path.resolve().as_posix())
        if key in seen:
            continue
        seen.add(key)
        pair_id = f"{len(rows):06d}_{sanitize_id(hr_path.stem)}"
        rows.append(
            {
                "pair_id": pair_id,
                "split": detect_split(lr_path),
                "lr_path": lr_path.resolve().as_posix(),
                "hr_path": hr_path.resolve().as_posix(),
                "lr_rel": relative_or_absolute(lr_path, data_root),
                "hr_rel": relative_or_absolute(hr_path, data_root),
            }
        )
    return rows


def write_meta_info(rows: list[dict[str, str]], meta_info: Path) -> None:
    meta_info.parent.mkdir(parents=True, exist_ok=True)
    with meta_info.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"{row['hr_rel']}, {row['lr_rel']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LR/HR pairs for the SRGD Real-ESRGAN project.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder containing the dataset.")
    parser.add_argument("--lr-token", default="270p", help="Token identifying low-resolution files/folders.")
    parser.add_argument("--hr-token", default="1080p", help="Token identifying high-resolution files/folders.")
    parser.add_argument("--lr-dir", type=Path, default=None, help="Optional explicit LR folder.")
    parser.add_argument("--hr-dir", type=Path, default=None, help="Optional explicit HR folder.")
    parser.add_argument("--out", type=Path, default=Path("data/pairs.csv"), help="Output pairs CSV.")
    parser.add_argument("--meta-info", type=Path, default=None, help="Optional Real-ESRGAN pair metadata txt.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of pairs.")
    parser.add_argument("--test-ratio", type=float, default=0.0,
                        help="Fraction of pairs to hold out as test split (0 = no test split).")
    parser.add_argument("--test-seed", type=int, default=42,
                        help="Random seed for the reproducible train/test split.")
    return parser.parse_args()


def main() -> None:
    import random

    args = parse_args()
    data_root = args.data_root.resolve()

    if args.lr_dir and args.hr_dir:
        pairs = collect_explicit_pairs(args.lr_dir.resolve(), args.hr_dir.resolve(), args.lr_token, args.hr_token)
    else:
        pairs = collect_auto_pairs(data_root, args.lr_token, args.hr_token)

    rows = build_rows(pairs, data_root)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.test_ratio > 0.0:
        rng = random.Random(args.test_seed)
        pair_ids = [r["pair_id"] for r in rows]
        rng.shuffle(pair_ids)
        n_test = max(1, round(len(pair_ids) * args.test_ratio))
        test_ids = set(pair_ids[:n_test])
        for row in rows:
            row["split"] = "test" if row["pair_id"] in test_ids else "train"

    train_rows = [row for row in rows if row["split"] != "test"]

    fieldnames = ["pair_id", "split", "lr_path", "hr_path", "lr_rel", "hr_rel"]
    write_csv(args.out, rows, fieldnames)
    if args.meta_info:
        write_meta_info(train_rows, args.meta_info)

    split_counts: dict[str, int] = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    print(f"Wrote {len(rows)} pairs to {args.out}  (train={len(train_rows)}, test={len(rows)-len(train_rows)})")
    if args.meta_info:
        print(f"Wrote Real-ESRGAN metadata ({len(train_rows)} train rows) to {args.meta_info}")
    print(f"Split counts: {split_counts}")


if __name__ == "__main__":
    main()
