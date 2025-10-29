"""
CLI for neural style transfer — supports Gatys (VGG-19), Johnson (fast), and AdaIN.

Usage:
  python inference/run_style_transfer.py --method gatys  --content c.jpg --style s.jpg --output out.jpg
  python inference/run_style_transfer.py --method johnson --content c.jpg --weights checkpoints/fast_style/wood.pth
  python inference/run_style_transfer.py --method adain  --content c.jpg --style s.jpg --alpha 0.8
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms


def load_image(path: str, size: Optional[int] = None, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    ops = []
    if size:
        ops.append(transforms.Resize(size))
    ops.append(transforms.ToTensor())
    t = transforms.Compose(ops)(img).unsqueeze(0)
    return t.to(device)


def save_image(tensor: torch.Tensor, path: str):
    img = tensor.squeeze(0).clamp(0, 1)
    img = transforms.ToPILImage()(img)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"Saved: {path}")


def run_gatys(
    content_path: str,
    style_path: str,
    output_path: str,
    num_steps: int = 500,
    content_weight: float = 1.0,
    style_weight: float = 1e6,
    tv_weight: float = 1e-4,
    image_size: int = 512,
    device: Optional[str] = None,
):
    from models.vgg_style_transfer import VGGStyleTransfer

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    nst = VGGStyleTransfer(content_weight=content_weight, style_weight=style_weight, tv_weight=tv_weight, device=str(dev))

    content = load_image(content_path, size=image_size, device=dev)
    style = load_image(style_path, size=image_size, device=dev)

    print(f"Running Gatys NST for {num_steps} steps on {dev} ...")

    def cb(step, loss, img):
        print(f"  step {step} | loss={loss:.2f}")

    result = nst.transfer(content, style, num_steps=num_steps, callback=cb)
    save_image(result, output_path)


def run_johnson(
    content_path: str,
    weights_path: str,
    output_path: str,
    image_size: int = 512,
    device: Optional[str] = None,
):
    from models.fast_style_network import FastStyleNetwork

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FastStyleNetwork.from_checkpoint(weights_path, device=str(dev))

    content = load_image(content_path, size=image_size, device=dev)
    print(f"Running Johnson fast NST on {dev} ...")
    with torch.no_grad():
        result = model(content)
    save_image(result, output_path)


def run_adain(
    content_path: str,
    style_path: str,
    output_path: str,
    decoder_path: Optional[str] = None,
    alpha: float = 1.0,
    image_size: int = 512,
    device: Optional[str] = None,
):
    from models.adain import AdaINStyleTransfer

    dev = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if decoder_path:
        model = AdaINStyleTransfer.from_checkpoint(decoder_path, device=dev)
    else:
        model = AdaINStyleTransfer(device=dev)

    content = load_image(content_path, size=image_size, device=torch.device(dev))
    style = load_image(style_path, size=image_size, device=torch.device(dev))
    print(f"Running AdaIN style transfer (alpha={alpha}) on {dev} ...")
    result = model.transfer(content, style, alpha=alpha)
    save_image(result, output_path)


def main():
    parser = argparse.ArgumentParser(description="Neural style transfer inference")
    parser.add_argument("--method", required=True, choices=["gatys", "johnson", "adain"])
    parser.add_argument("--content", required=True, help="Content image path")
    parser.add_argument("--style", default=None, help="Style image path (gatys / adain)")
    parser.add_argument("--weights", default=None, help="Model weights path (johnson)")
    parser.add_argument("--decoder", default=None, help="Decoder weights path (adain)")
    parser.add_argument("--output", default="outputs/result.jpg")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=500, help="Gatys optimization steps")
    parser.add_argument("--alpha", type=float, default=1.0, help="AdaIN style strength [0–1]")
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=1e6)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.method == "gatys":
        if not args.style:
            parser.error("--style required for gatys")
        run_gatys(args.content, args.style, args.output, args.num_steps,
                  args.content_weight, args.style_weight, image_size=args.image_size, device=args.device)

    elif args.method == "johnson":
        if not args.weights:
            parser.error("--weights required for johnson")
        run_johnson(args.content, args.weights, args.output, image_size=args.image_size, device=args.device)

    elif args.method == "adain":
        if not args.style:
            parser.error("--style required for adain")
        run_adain(args.content, args.style, args.output, args.decoder,
                  args.alpha, args.image_size, args.device)


if __name__ == "__main__":
    main()
