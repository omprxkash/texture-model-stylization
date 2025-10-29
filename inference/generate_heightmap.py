"""
Generate a tactile heightmap from a texture image using the fine-tuned diffusion model.

Falls back to the gradient-based synthetic generator if LoRA weights are not available,
making this script runnable without GPU or pre-trained weights.

Usage:
  python inference/generate_heightmap.py --input texture.jpg
  python inference/generate_heightmap.py --input texture.jpg --lora checkpoints/heightmap_lora/final
  python inference/generate_heightmap.py --input texture.jpg --method synthetic --sigma 3.0
"""
import argparse
from pathlib import Path
from typing import Optional

from PIL import Image


def generate_with_diffusion(
    input_path: str,
    lora_path: Optional[str],
    output_path: str,
    strength: float = 0.8,
    guidance_scale: float = 7.5,
    steps: int = 30,
    seed: int = 42,
    output_size: int = 512,
    device: Optional[str] = None,
):
    from models.heightmap_diffusion import HeightmapDiffusionPipeline

    pipeline = HeightmapDiffusionPipeline(
        lora_weights_path=lora_path,
        device=device,
    )
    print(f"Generating heightmap from {input_path} (diffusion method) ...")
    hm = pipeline.generate(
        texture_image=input_path,
        strength=strength,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        seed=seed,
        output_size=output_size,
    )
    pipeline.unload()
    hm.save(output_path)
    print(f"Saved: {output_path}")


def generate_with_synthetic(
    input_path: str,
    output_path: str,
    sigma: float = 2.0,
    invert: bool = False,
):
    from data.heightmap_generator import heightmap_from_path

    print(f"Generating heightmap from {input_path} (synthetic/gradient method) ...")
    hm = heightmap_from_path(input_path, sigma=sigma, invert=invert)
    hm.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate heightmap from texture image")
    parser.add_argument("--input", required=True, help="Input texture image")
    parser.add_argument("--output", default=None, help="Output heightmap path")
    parser.add_argument("--method", choices=["diffusion", "synthetic", "auto"], default="auto",
                        help="Generation method (auto: diffusion if weights available, else synthetic)")
    parser.add_argument("--lora", default=None, help="Path to LoRA weights directory")
    parser.add_argument("--strength", type=float, default=0.8, help="Diffusion strength (0–1)")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian blur sigma for synthetic method")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    stem = Path(args.input).stem
    output = args.output or f"outputs/{stem}_heightmap.png"
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    use_diffusion = args.method == "diffusion"
    if args.method == "auto":
        try:
            import diffusers  # noqa: F401
            import torch
            use_diffusion = True
        except ImportError:
            use_diffusion = False

    if use_diffusion:
        generate_with_diffusion(
            args.input, args.lora, output,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            steps=args.steps,
            seed=args.seed,
            output_size=args.size,
            device=args.device,
        )
    else:
        generate_with_synthetic(args.input, output, sigma=args.sigma, invert=args.invert)


if __name__ == "__main__":
    main()
