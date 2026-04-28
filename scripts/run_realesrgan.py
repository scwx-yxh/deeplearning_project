from __future__ import annotations

import argparse
import shutil
import subprocess
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

from srgd_realesrgan.paths import IMAGE_EXTENSIONS, ensure_dir, read_pairs_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official Real-ESRGAN inference on paired LR images.")
    parser.add_argument("--realesrgan-dir", type=Path, required=True, help="Path to cloned xinntao/Real-ESRGAN repo.")
    parser.add_argument("--pairs", type=Path, required=True, help="Pairs CSV from prepare_pairs.py.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Folder for normalized SR PNG outputs.")
    parser.add_argument("--model-name", default="RealESRGAN_x4plus", help="Real-ESRGAN model name.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional custom Real-ESRGAN checkpoint.")
    parser.add_argument("--outscale", type=float, default=4.0, help="Output scale passed to inference_realesrgan.py.")
    parser.add_argument("--tile", type=int, default=0, help="Tile size. Use 128/256 on small GPUs.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Optional GPU id passed to inference_realesrgan.py.")
    parser.add_argument("--ext", default="png", help="Output image extension passed to inference_realesrgan.py.")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 inference instead of half precision.")
    parser.add_argument("--face-enhance", action="store_true", help="Enable GFPGAN face enhancement.")
    parser.add_argument("--split", default=None, help="Optional split filter: train, val, or test.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of images.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before applying --limit.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --shuffle.")
    return parser.parse_args()


def normalize_outputs(raw_dir: Path, out_dir: Path, rows: list[dict[str, str]], suffix: str) -> None:
    for row in tqdm(rows, desc="Normalize outputs"):
        pair_id = row["pair_id"]
        matches = []
        for ext in IMAGE_EXTENSIONS:
            matches.extend(raw_dir.glob(f"{pair_id}_{suffix}{ext}"))
            matches.extend(raw_dir.glob(f"{pair_id}_{suffix}{ext.upper()}"))
        if not matches:
            matches = list(raw_dir.glob(f"{pair_id}_*"))
        if not matches:
            raise FileNotFoundError(f"Real-ESRGAN output not found for pair_id={pair_id} in {raw_dir}")

        image = Image.open(matches[0]).convert("RGB")
        image.save(out_dir / f"{pair_id}.png")


def main() -> None:
    args = parse_args()
    realesrgan_dir = args.realesrgan_dir.resolve()
    inference_script = realesrgan_dir / "inference_realesrgan.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Cannot find {inference_script}. Did you clone the official Real-ESRGAN repo?")

    rows = read_pairs_csv(args.pairs, split=args.split, limit=args.limit, shuffle=args.shuffle, seed=args.seed)
    out_dir = ensure_dir(args.out_dir)
    input_dir = ensure_dir(out_dir / "_lr_inputs")
    raw_dir = ensure_dir(out_dir / "_raw_realesrgan")
    suffix = "sr"

    for row in tqdm(rows, desc="Stage LR inputs"):
        src = Path(row["lr_path"])
        dst = input_dir / f"{row['pair_id']}{src.suffix.lower()}"
        shutil.copy2(src, dst)

    command = [
        sys.executable,
        str(inference_script),
        "-n",
        args.model_name,
        "-i",
        str(input_dir),
        "-o",
        str(raw_dir),
        "--outscale",
        str(args.outscale),
        "--suffix",
        suffix,
        "--tile",
        str(args.tile),
        "--ext",
        args.ext,
    ]
    if args.model_path:
        command.extend(["--model_path", str(args.model_path.resolve())])
    if args.gpu_id is not None:
        command.extend(["--gpu-id", str(args.gpu_id)])
    if args.fp32:
        command.append("--fp32")
    if args.face_enhance:
        command.append("--face_enhance")

    print("Running:", " ".join(command))
    subprocess.run(command, cwd=realesrgan_dir, check=True)
    normalize_outputs(raw_dir, out_dir, rows, suffix)
    print(f"Wrote normalized Real-ESRGAN outputs to {out_dir}")


if __name__ == "__main__":
    main()
