"""
Evaluation metrics for generated texture quality.

Metrics:
  SSIM  — Structural Similarity Index (skimage): perceptual structural quality
  LPIPS — Learned Perceptual Image Patch Similarity (lpips): deep feature distance
  FID   — Fréchet Inception Distance (torch-fidelity): distribution-level quality

All functions accept either file paths or PIL Images / numpy arrays for flexibility.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def _load_image(img: Union[str, Image.Image, np.ndarray], size: Optional[int] = None) -> np.ndarray:
    if isinstance(img, str):
        img = Image.open(img).convert("RGB")
    if isinstance(img, Image.Image):
        if size:
            img = img.resize((size, size), Image.LANCZOS)
        img = np.array(img)
    return img.astype(np.float32) / 255.0


def compute_ssim(
    img1: Union[str, Image.Image, np.ndarray],
    img2: Union[str, Image.Image, np.ndarray],
    size: int = 256,
) -> float:
    """Structural Similarity Index (range: [-1, 1], higher is better)."""
    from skimage.metrics import structural_similarity as ssim

    a = _load_image(img1, size)
    b = _load_image(img2, size)
    score = ssim(a, b, data_range=1.0, channel_axis=-1)
    return float(score)


def compute_lpips(
    img1: Union[str, Image.Image, np.ndarray],
    img2: Union[str, Image.Image, np.ndarray],
    net: str = "alex",
    size: int = 256,
    device: Optional[str] = None,
) -> float:
    """LPIPS perceptual distance (lower is better, range: [0, ∞))."""
    try:
        import torch
        import lpips
    except ImportError:
        raise ImportError("Install lpips: pip install lpips")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loss_fn = lpips.LPIPS(net=net).to(dev)
    loss_fn.eval()

    def to_tensor(img):
        arr = _load_image(img, size)
        t = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0).float()
        return t * 2 - 1  # normalize to [-1, 1]

    with torch.no_grad():
        dist = loss_fn(to_tensor(img1).to(dev), to_tensor(img2).to(dev))
    return float(dist.item())


def compute_fid(
    real_dir: str,
    gen_dir: str,
    batch_size: int = 64,
    device: Optional[str] = None,
) -> float:
    """Fréchet Inception Distance (lower is better).

    Both directories must contain image files.
    Requires torch-fidelity: pip install torch-fidelity
    """
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        raise ImportError("Install torch-fidelity: pip install torch-fidelity")

    metrics = calculate_metrics(
        input1=real_dir,
        input2=gen_dir,
        cuda=True if (device != "cpu") else False,
        isc=False,
        fid=True,
        kid=False,
        batch_size=batch_size,
        verbose=False,
    )
    return float(metrics["frechet_inception_distance"])


def evaluate_batch(
    real_dir: str,
    gen_dir: str,
    size: int = 256,
    max_pairs: int = 100,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Compute SSIM + LPIPS on matched file pairs, FID on directory-level distribution.

    Expects gen_dir to have files with the same stems as real_dir.
    """
    real_paths = sorted(Path(real_dir).glob("*.jpg")) + sorted(Path(real_dir).glob("*.png"))
    results: Dict[str, List[float]] = {"ssim": [], "lpips": []}

    pairs = 0
    for real_path in real_paths:
        gen_path = Path(gen_dir) / real_path.name
        if not gen_path.exists():
            gen_path = Path(gen_dir) / (real_path.stem + ".png")
        if not gen_path.exists():
            continue

        results["ssim"].append(compute_ssim(str(real_path), str(gen_path), size=size))
        results["lpips"].append(compute_lpips(str(real_path), str(gen_path), size=size, device=device))
        pairs += 1
        if pairs >= max_pairs:
            break

    summary: Dict[str, float] = {}
    for metric, values in results.items():
        if values:
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values))

    print(f"Evaluated {pairs} pairs:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")

    try:
        fid = compute_fid(real_dir, gen_dir, device=device)
        summary["fid"] = fid
        print(f"  fid: {fid:.2f}")
    except Exception as e:
        print(f"  FID skipped: {e}")

    return summary


def print_results_table(results: Dict[str, float], title: str = "Evaluation Results"):
    print(f"\n{'='*40}")
    print(f"  {title}")
    print(f"{'='*40}")
    for k, v in results.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"{'='*40}\n")
