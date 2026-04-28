# Game Image Super-Resolution with Real-ESRGAN

This project implements a practical deep-learning course pipeline for the Kaggle/SRGD **Super Resolution in Video Games** dataset and the official **Real-ESRGAN** model.

The project does three things:

1. Builds paired low-resolution/high-resolution image metadata from the dataset.
2. Runs a bicubic baseline and Real-ESRGAN inference.
3. Evaluates PSNR, SSIM, optional LPIPS, and creates visual comparison grids.

## Project Layout

```text
.
├── configs/                  # generated Real-ESRGAN fine-tuning configs
├── data/                     # put the Kaggle/SRGD data here
├── outputs/                  # predictions, metrics, visual grids
├── scripts/                  # command-line project workflow
└── src/srgd_realesrgan/      # reusable helper code
```

## Setup

Create an environment and install the lightweight project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Real-ESRGAN inference/fine-tuning, also clone and install the official repo:

```bash
git clone https://github.com/xinntao/Real-ESRGAN.git external/Real-ESRGAN
cd external/Real-ESRGAN
pip install basicsr facexlib gfpgan
pip install -r requirements.txt
python setup.py develop
```

Download pretrained weights inside `external/Real-ESRGAN/weights` or `external/Real-ESRGAN/experiments/pretrained_models` following the official README/training guide.

## 1. Prepare Dataset Pairs

Put the Kaggle/SRGD data under `data/raw`. The script can usually discover pairs automatically when paths contain resolution tokens such as `270p` and `1080p`.

```bash
python scripts/prepare_pairs.py \
  --data-root data/raw \
  --lr-token 270p \
  --hr-token 1080p \
  --out data/pairs.csv \
  --meta-info data/meta_info_srgd_pair.txt
```

If auto-discovery is not enough, provide folders explicitly:

```bash
python scripts/prepare_pairs.py \
  --data-root data/raw \
  --lr-dir data/raw/train/270p \
  --hr-dir data/raw/train/1080p \
  --out data/pairs.csv \
  --meta-info data/meta_info_srgd_pair.txt
```

The generated `meta_info_srgd_pair.txt` uses the Real-ESRGAN paired-data format:

```text
relative_hr_path, relative_lr_path
```

## 2. Run Bicubic Baseline

```bash
python scripts/run_bicubic.py \
  --pairs data/pairs.csv \
  --out-dir outputs/bicubic \
  --limit 100
```

## 3. Run Real-ESRGAN Inference

```bash
python scripts/run_realesrgan.py \
  --realesrgan-dir external/Real-ESRGAN \
  --pairs data/pairs.csv \
  --out-dir outputs/realesrgan_x4plus \
  --model-name RealESRGAN_x4plus \
  --outscale 4 \
  --tile 256 \
  --limit 100
```

Use `--tile 256` or `--tile 128` if GPU memory is limited.

## 4. Evaluate Metrics

```bash
python scripts/evaluate_sr.py \
  --pairs data/pairs.csv \
  --sr-dir outputs/bicubic \
  --out outputs/bicubic_metrics.csv \
  --summary outputs/bicubic_summary.json
```

For LPIPS, install `lpips` and add `--lpips`:

```bash
python scripts/evaluate_sr.py \
  --pairs data/pairs.csv \
  --sr-dir outputs/realesrgan_x4plus \
  --out outputs/realesrgan_metrics.csv \
  --summary outputs/realesrgan_summary.json \
  --lpips
```

## 5. Make Visual Comparison Grids

```bash
python scripts/make_visual_grid.py \
  --pairs data/pairs.csv \
  --methods Bicubic=outputs/bicubic RealESRGAN=outputs/realesrgan_x4plus \
  --out-dir outputs/grids \
  --limit 12
```

Each grid shows:

```text
LR | Bicubic | RealESRGAN | HR
```

## 6. Generate a Fine-Tuning Config

This creates a Real-ESRGAN paired-data config using our dataset metadata:

```bash
python scripts/write_realesrgan_config.py \
  --data-root data/raw \
  --meta-info data/meta_info_srgd_pair.txt \
  --out configs/finetune_realesrgan_x4plus_pairdata_srgd.yml \
  --batch-size 4 \
  --gt-size 256 \
  --total-iter 20000
```

Copy or reference that config from inside the Real-ESRGAN repo, then run:

```bash
cd external/Real-ESRGAN
python realesrgan/train.py -opt ../../configs/finetune_realesrgan_x4plus_pairdata_srgd.yml --auto_resume
```

For a course project, start with inference plus bicubic comparison, then fine-tune on a small subset once the metrics pipeline is working.
