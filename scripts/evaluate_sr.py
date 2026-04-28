from __future__ import annotations

import argparse
import json
import statistics
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

from srgd_realesrgan.metrics import LPIPSEvaluator, calculate_metrics
from srgd_realesrgan.paths import IMAGE_EXTENSIONS, ensure_dir, read_pairs_csv, write_csv


def find_sr_path(sr_dir: Path, pair_id: str) -> Path:
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"]:
        candidate = sr_dir / f"{pair_id}{ext}"
        if candidate.exists():
            return candidate
    matches = []
    for ext in IMAGE_EXTENSIONS:
        matches.extend(sr_dir.glob(f"{pair_id}*{ext}"))
        matches.extend(sr_dir.glob(f"{pair_id}*{ext.upper()}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No SR output found for pair_id={pair_id} in {sr_dir}")


def resize_to_hr_if_needed(sr_path: Path, hr_path: Path, temp_dir: Path) -> Path:
    sr = Image.open(sr_path).convert("RGB")
    hr = Image.open(hr_path).convert("RGB")
    if sr.size == hr.size:
        return sr_path
    temp_dir.mkdir(parents=True, exist_ok=True)
    resized = sr.resize(hr.size, Image.Resampling.BICUBIC)
    resized_path = temp_dir / sr_path.name
    resized.save(resized_path)
    return resized_path


def mean(values: list[float]) -> float | None:
    finite_values = [value for value in values if value is not None]
    if not finite_values:
        return None
    return float(statistics.fmean(finite_values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SR outputs against HR images.")
    parser.add_argument("--pairs", type=Path, required=True, help="Pairs CSV from prepare_pairs.py.")
    parser.add_argument("--sr-dir", type=Path, required=True, help="Folder containing pair_id.png SR outputs.")
    parser.add_argument("--out", type=Path, required=True, help="Per-image metrics CSV.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional summary JSON path.")
    parser.add_argument("--split", default=None, help="Optional split filter: train, val, or test.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle.")
    parser.add_argument("--crop", type=int, default=4, help="Border crop for PSNR/SSIM.")
    parser.add_argument("--y-channel", action="store_true", help="Evaluate PSNR/SSIM on luma channel.")
    parser.add_argument("--resize-sr", action="store_true", help="Resize SR outputs to HR size before metrics.")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS. Requires lpips + torch.")
    parser.add_argument("--lpips-net", default="alex", help="LPIPS backbone: alex, vgg, or squeeze.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_pairs_csv(args.pairs, split=args.split, limit=args.limit, shuffle=args.shuffle, seed=args.seed)
    lpips_evaluator = LPIPSEvaluator(net=args.lpips_net) if args.lpips else None
    temp_dir = ensure_dir(args.sr_dir / "_resized_for_metrics")

    metric_rows: list[dict[str, object]] = []
    for row in tqdm(rows, desc="Evaluate"):
        sr_path = find_sr_path(args.sr_dir, row["pair_id"])
        metric_sr_path = sr_path
        if args.resize_sr:
            metric_sr_path = resize_to_hr_if_needed(sr_path, Path(row["hr_path"]), temp_dir)

        result = calculate_metrics(
            metric_sr_path,
            row["hr_path"],
            crop=args.crop,
            y_channel=args.y_channel,
            lpips_evaluator=lpips_evaluator,
        )
        metric_rows.append(
            {
                "pair_id": row["pair_id"],
                "split": row.get("split", ""),
                "sr_path": sr_path.as_posix(),
                "hr_path": row["hr_path"],
                "psnr": result.psnr,
                "ssim": result.ssim,
                "lpips": result.lpips,
            }
        )

    fieldnames = ["pair_id", "split", "sr_path", "hr_path", "psnr", "ssim", "lpips"]
    write_csv(args.out, metric_rows, fieldnames)

    summary = {
        "count": len(metric_rows),
        "psnr_mean": mean([float(row["psnr"]) for row in metric_rows]),
        "ssim_mean": mean([float(row["ssim"]) for row in metric_rows]),
        "lpips_mean": mean([row["lpips"] for row in metric_rows if row["lpips"] is not None]),
    }
    print(json.dumps(summary, indent=2))

    if args.summary:
        ensure_dir(args.summary.parent)
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
