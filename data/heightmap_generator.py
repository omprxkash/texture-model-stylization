"""
Synthetic heightmap generation from texture images.

Pipeline:
  1. Grayscale conversion
  2. Sobel edge filter  →  captures surface microstructure variation
  3. Gaussian blur (σ=2)  →  smooth continuous displacement field
  4. Normalize to [0, 255] uint8  →  grayscale PNG heightmap

The resulting heightmap encodes perceived surface relief:
bright pixels = raised areas, dark pixels = recessed areas.
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


def generate_heightmap(
    texture: np.ndarray,
    sigma: float = 2.0,
    invert: bool = False,
    height_scale: float = 1.0,
) -> np.ndarray:
    """Convert a texture image array (H, W, 3) or (H, W) to a heightmap (H, W) uint8."""
    if texture.ndim == 3:
        gray = 0.299 * texture[..., 0] + 0.587 * texture[..., 1] + 0.114 * texture[..., 2]
    else:
        gray = texture.astype(np.float32)

    # Sobel gradient magnitude → encodes edge density / surface roughness
    sx = ndimage.sobel(gray, axis=1)
    sy = ndimage.sobel(gray, axis=0)
    gradient_mag = np.hypot(sx, sy)

    # Smooth to get a continuous displacement field
    blurred = ndimage.gaussian_filter(gradient_mag, sigma=sigma)

    # Blend gradient with luminance to preserve large-scale shape
    luminance_norm = gray / 255.0
    combined = 0.6 * blurred + 0.4 * luminance_norm * blurred.max()

    combined = combined * height_scale
    if invert:
        combined = combined.max() - combined

    # Normalize to uint8
    lo, hi = combined.min(), combined.max()
    if hi - lo < 1e-6:
        return np.zeros_like(combined, dtype=np.uint8)
    normalized = (combined - lo) / (hi - lo) * 255.0
    return normalized.astype(np.uint8)


def heightmap_from_path(
    image_path: str,
    sigma: float = 2.0,
    invert: bool = False,
    height_scale: float = 1.0,
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    hm = generate_heightmap(arr, sigma=sigma, invert=invert, height_scale=height_scale)
    return Image.fromarray(hm, mode="L")


def batch_generate_heightmaps(
    input_dir: str,
    output_dir: str,
    sigma: float = 2.0,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    invert: bool = False,
) -> List[Tuple[str, str]]:
    """Process all texture images in input_dir and save heightmaps to output_dir."""
    inp = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[str, str]] = []
    image_files = [f for f in inp.rglob("*") if f.suffix.lower() in extensions]

    for img_path in image_files:
        rel = img_path.relative_to(inp)
        hm_path = out / rel.with_suffix(".png")
        hm_path.parent.mkdir(parents=True, exist_ok=True)

        hm_img = heightmap_from_path(str(img_path), sigma=sigma, invert=invert)
        hm_img.save(str(hm_path))
        pairs.append((str(img_path), str(hm_path)))

    return pairs


def generate_dtd_heightmaps(dtd_root: str, output_root: str, sigma: float = 2.0):
    """Generate paired heightmaps for every image in the DTD dataset."""
    dtd_images = os.path.join(dtd_root, "images")
    out_root = Path(output_root)

    total, done = 0, 0
    for category in sorted(os.listdir(dtd_images)):
        cat_in = os.path.join(dtd_images, category)
        cat_out = out_root / category
        cat_out.mkdir(parents=True, exist_ok=True)
        imgs = list(Path(cat_in).glob("*.jpg"))
        total += len(imgs)
        for img_path in imgs:
            hm_path = cat_out / (img_path.stem + ".png")
            if not hm_path.exists():
                hm = heightmap_from_path(str(img_path), sigma=sigma)
                hm.save(str(hm_path))
            done += 1
        print(f"  {category}: {len(imgs)} heightmaps generated", flush=True)

    print(f"\nTotal: {done}/{total} heightmaps written to {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heightmaps from texture images")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument("--output", default=None, help="Output path (image or directory)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur sigma")
    parser.add_argument("--invert", action="store_true", help="Invert heightmap")
    parser.add_argument("--dtd-root", default=None, help="Process full DTD dataset")
    args = parser.parse_args()

    if args.dtd_root:
        out = args.output or "data/dtd_heightmaps"
        generate_dtd_heightmaps(args.dtd_root, out, sigma=args.sigma)
    elif os.path.isdir(args.input):
        out = args.output or args.input + "_heightmaps"
        pairs = batch_generate_heightmaps(args.input, out, sigma=args.sigma, invert=args.invert)
        print(f"Generated {len(pairs)} heightmaps in {out}")
    else:
        hm = heightmap_from_path(args.input, sigma=args.sigma, invert=args.invert)
        out = args.output or Path(args.input).stem + "_heightmap.png"
        hm.save(out)
        print(f"Saved heightmap: {out}")
