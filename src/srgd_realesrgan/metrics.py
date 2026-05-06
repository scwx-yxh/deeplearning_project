from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def load_rgb01(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def crop_border(image: np.ndarray, border: int) -> np.ndarray:
    if border <= 0:
        return image
    if image.shape[0] <= border * 2 or image.shape[1] <= border * 2:
        raise ValueError(f"Cannot crop border={border} from image with shape {image.shape}")
    return image[border:-border, border:-border, ...]


def rgb_to_luma(image: np.ndarray) -> np.ndarray:
    return image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114


def calculate_psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    mse = float(np.mean((sr - hr) ** 2))
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(1.0 / mse)


def calculate_ssim(sr: np.ndarray, hr: np.ndarray, *, y_channel: bool = False) -> float:
    if y_channel:
        hr_y = rgb_to_luma(hr)
        sr_y = rgb_to_luma(sr)
        if structural_similarity is not None:
            return float(structural_similarity(hr_y, sr_y, data_range=1.0))
        return _global_ssim(hr_y, sr_y)
    if structural_similarity is not None:
        return float(structural_similarity(hr, sr, channel_axis=2, data_range=1.0))
    return float(np.mean([_global_ssim(hr[..., channel], sr[..., channel]) for channel in range(3)]))


def _global_ssim(first: np.ndarray, second: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    mu_first = float(first.mean())
    mu_second = float(second.mean())
    var_first = float(((first - mu_first) ** 2).mean())
    var_second = float(((second - mu_second) ** 2).mean())
    covariance = float(((first - mu_first) * (second - mu_second)).mean())
    numerator = (2 * mu_first * mu_second + c1) * (2 * covariance + c2)
    denominator = (mu_first**2 + mu_second**2 + c1) * (var_first + var_second + c2)
    return numerator / denominator


@dataclass
class MetricResult:
    psnr: float
    ssim: float
    lpips: float | None = None
    dists: float | None = None


def calculate_metrics(
    sr_path: str | Path,
    hr_path: str | Path,
    *,
    crop: int = 4,
    y_channel: bool = False,
    lpips_evaluator: Any | None = None,
    dists_evaluator: Any | None = None,
) -> MetricResult:
    sr = load_rgb01(sr_path)
    hr = load_rgb01(hr_path)
    if sr.shape != hr.shape:
        raise ValueError(f"Shape mismatch: SR {sr.shape} vs HR {hr.shape} for {sr_path}")

    sr_eval = crop_border(sr, crop)
    hr_eval = crop_border(hr, crop)
    lpips_score = None
    if lpips_evaluator is not None:
        lpips_score = float(lpips_evaluator(sr_eval, hr_eval))
    dists_score = None
    if dists_evaluator is not None:
        dists_score = float(dists_evaluator(sr_eval, hr_eval))

    return MetricResult(
        psnr=calculate_psnr(sr_eval, hr_eval),
        ssim=calculate_ssim(sr_eval, hr_eval, y_channel=y_channel),
        lpips=lpips_score,
        dists=dists_score,
    )


class LPIPSEvaluator:
    def __init__(self, net: str = "alex", device: str | None = None) -> None:
        import torch
        import lpips

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = lpips.LPIPS(net=net).to(self.device).eval()

    def __call__(self, sr: np.ndarray, hr: np.ndarray) -> float:
        torch = self.torch
        sr_tensor = self._to_tensor(sr)
        hr_tensor = self._to_tensor(hr)
        with torch.no_grad():
            value = self.loss_fn(sr_tensor, hr_tensor)
        return float(value.item())

    def _to_tensor(self, image: np.ndarray):
        tensor = self.torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor * 2.0 - 1.0


class DISTSEvaluator:
    """DISTS: perceptual + texture similarity, invariant to texture shifts.

    Lower is better (like LPIPS). Requires: pip install DISTS-pytorch
    """

    def __init__(self, device: str | None = None) -> None:
        import os
        import shutil
        import sys
        import torch
        import DISTS_pytorch as _dists_pkg
        from DISTS_pytorch import DISTS

        # DISTS_pytorch loads weights from sys.prefix/weights.pt, but pip
        # installs the file inside the package directory. Copy it if missing.
        weights_target = os.path.join(sys.prefix, "weights.pt")
        if not os.path.exists(weights_target):
            weights_src = os.path.join(os.path.dirname(_dists_pkg.__file__), "weights.pt")
            if os.path.exists(weights_src):
                shutil.copy2(weights_src, weights_target)

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = DISTS().to(self.device).eval()

    def __call__(self, sr: np.ndarray, hr: np.ndarray) -> float:
        sr_tensor = self._to_tensor(sr)
        hr_tensor = self._to_tensor(hr)
        with self.torch.no_grad():
            value = self.loss_fn(sr_tensor, hr_tensor)
        return float(value.item())

    def _to_tensor(self, image: np.ndarray):
        # DISTS expects [0, 1] float tensors, shape (1, 3, H, W)
        tensor = self.torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
        return tensor.to(self.device)
