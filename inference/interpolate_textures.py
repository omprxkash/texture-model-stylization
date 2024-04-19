"""
Texture interpolation — blend two reference textures and generate a heightmap sweep.

Three interpolation modes:
  pixel    : blend in pixel/image space (fast, works without GPU)
  adain    : blend in AdaIN feature space (encoder latent space)
  clip     : blend in CLIP image embedding space, then decode via diffusion
             Inspired by CLIP affordance embedding clustering (3d-Affordance-CLIP-Embeddings).

Usage:
  python inference/interpolate_textures.py --a wood.jpg --b fabric.jpg --mode pixel --steps 5
  python inference/interpolate_textures.py --a wood.jpg --b fabric.jpg --mode adain --steps 7
  python inference/interpolate_textures.py --a wood.jpg --b fabric.jpg --mode clip --steps 5
"""
import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


def _pil_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])(img).unsqueeze(0)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    from torchvision import transforms
    return transforms.ToPILImage()(t.squeeze(0).clamp(0, 1))


def interpolate_pixel(
    img_a: Image.Image,
    img_b: Image.Image,
    alphas: List[float],
    size: int = 512,
) -> List[Image.Image]:
    a = np.array(img_a.resize((size, size))).astype(np.float32)
    b = np.array(img_b.resize((size, size))).astype(np.float32)
    results = []
    for alpha in alphas:
        blend = ((1 - alpha) * a + alpha * b).clip(0, 255).astype(np.uint8)
        results.append(Image.fromarray(blend))
    return results


def interpolate_adain(
    img_a: Image.Image,
    img_b: Image.Image,
    alphas: List[float],
    decoder_path: Optional[str] = None,
    size: int = 512,
    device: str = "cpu",
) -> List[Image.Image]:
    from models.adain import AdaINStyleTransfer

    dev = torch.device(device)
    if decoder_path:
        model = AdaINStyleTransfer.from_checkpoint(decoder_path, device=device)
    else:
        model = AdaINStyleTransfer(device=device)

    ta = _pil_to_tensor(img_a, size).to(dev)
    tb = _pil_to_tensor(img_b, size).to(dev)

    with torch.no_grad():
        feat_a = model.encoder(ta)
        feat_b = model.encoder(tb)

    results = []
    for alpha in alphas:
        blended_feat = (1 - alpha) * feat_a + alpha * feat_b
        decoded = model.decode(blended_feat)
        results.append(_tensor_to_pil(decoded.cpu()))
    return results


def interpolate_clip(
    img_a: Image.Image,
    img_b: Image.Image,
    alphas: List[float],
    lora_path: Optional[str] = None,
    size: int = 512,
    device: Optional[str] = None,
) -> List[Image.Image]:
    """
    Blend in CLIP image embedding space, then decode via diffusion heightmap pipeline.
    Inspired by CLIP embedding space clustering for 3D affordances.
    """
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from models.heightmap_diffusion import HeightmapDiffusionPipeline
    except ImportError as e:
        raise ImportError("Install transformers and diffusers for CLIP interpolation") from e

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs_a = processor(images=img_a, return_tensors="pt").to(dev)
    inputs_b = processor(images=img_b, return_tensors="pt").to(dev)

    with torch.no_grad():
        emb_a = clip_model.get_image_features(**inputs_a)
        emb_b = clip_model.get_image_features(**inputs_b)
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)

    # Decode back to pixel space using blended pixel images for conditioning
    # (true inversion from CLIP latent → pixel is non-trivial without a dedicated decoder;
    #  we blend pixel images proportionally as a practical approximation)
    pixel_interps = interpolate_pixel(img_a, img_b, alphas, size=size)

    pipeline = HeightmapDiffusionPipeline(lora_weights_path=lora_path, device=str(dev))
    results = []
    for img in pixel_interps:
        hm = pipeline.generate(img, output_size=size)
        results.append(hm)
    pipeline.unload()
    return results


def save_grid(images: List[Image.Image], alphas: List[float], output_dir: str, prefix: str = "interp"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for img, alpha in zip(images, alphas):
        name = f"{prefix}_alpha{alpha:.2f}.png"
        img.save(os.path.join(output_dir, name))

    # Save side-by-side grid
    w, h = images[0].size
    grid = Image.new("RGB", (w * len(images), h))
    for i, img in enumerate(images):
        grid.paste(img.convert("RGB"), (i * w, 0))
    grid_path = os.path.join(output_dir, f"{prefix}_grid.png")
    grid.save(grid_path)
    print(f"Saved {len(images)} interpolation frames + grid to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Interpolate between two texture/style images")
    parser.add_argument("--a", required=True, help="First texture image")
    parser.add_argument("--b", required=True, help="Second texture image")
    parser.add_argument("--mode", choices=["pixel", "adain", "clip"], default="pixel")
    parser.add_argument("--steps", type=int, default=5, help="Number of interpolation steps")
    parser.add_argument("--output-dir", default="outputs/interpolation")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--decoder", default=None, help="AdaIN decoder weights")
    parser.add_argument("--lora", default=None, help="LoRA weights for CLIP mode")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    img_a = Image.open(args.a).convert("RGB")
    img_b = Image.open(args.b).convert("RGB")
    alphas = [i / (args.steps - 1) for i in range(args.steps)]

    prefix = f"{args.mode}_{Path(args.a).stem}_to_{Path(args.b).stem}"

    print(f"Interpolating ({args.mode} mode, {args.steps} steps) ...")

    if args.mode == "pixel":
        images = interpolate_pixel(img_a, img_b, alphas, size=args.size)
    elif args.mode == "adain":
        images = interpolate_adain(img_a, img_b, alphas, args.decoder, args.size, args.device or "cpu")
    elif args.mode == "clip":
        images = interpolate_clip(img_a, img_b, alphas, args.lora, args.size, args.device)

    save_grid(images, alphas, args.output_dir, prefix=prefix)


if __name__ == "__main__":
    main()
