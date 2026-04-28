from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from srgd_realesrgan.paths import ensure_dir, read_pairs_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bicubic x4-style baseline matched to HR size.")
    parser.add_argument("--pairs", type=Path, required=True, help="Pairs CSV from prepare_pairs.py.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder for upscaled PNG outputs.")
    parser.add_argument("--split", default=None, help="Optional split filter: train, val, or test.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_pairs_csv(args.pairs, split=args.split, limit=args.limit, shuffle=args.shuffle, seed=args.seed)
    out_dir = ensure_dir(args.out_dir)

    for row in tqdm(rows, desc="Bicubic"):
        out_path = out_dir / f"{row['pair_id']}.png"
        if out_path.exists() and not args.overwrite:
            continue

        lr = Image.open(row["lr_path"]).convert("RGB")
        hr = Image.open(row["hr_path"]).convert("RGB")
        sr = lr.resize(hr.size, Image.Resampling.BICUBIC)
        sr.save(out_path)

    print(f"Wrote bicubic outputs to {out_dir}")


if __name__ == "__main__":
    main()
