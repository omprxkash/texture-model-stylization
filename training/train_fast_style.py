"""
Train the Johnson fast style transfer network.

One network is trained per style image, saved as checkpoints/fast_style/<style_name>.pth.
Content images are drawn from DTD (or COCO if available).

Usage:
  python training/train_fast_style.py --style assets/style_wood.jpg --content-dir data/dtd/images
  python training/train_fast_style.py --config configs/style_transfer.yaml --style-name mosaic
"""
import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from models.fast_style_network import FastStyleNetwork, PerceptualLoss


class ContentDataset(Dataset):
    def __init__(self, root: str, image_size: int = 256, limit: Optional[int] = None):
        exts = {".jpg", ".jpeg", ".png"}
        paths = [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]
        if limit:
            paths = paths[:limit]
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size, pad_if_needed=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)


def load_style_image(path: str, size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    t = transforms.ToTensor()(img).unsqueeze(0)
    return t.to(device)


def train(
    style_image_path: str,
    content_dir: str,
    output_dir: str = "checkpoints/fast_style",
    style_name: Optional[str] = None,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-3,
    content_weight: float = 1.0,
    style_weight: float = 1e5,
    tv_weight: float = 1e-6,
    image_size: int = 256,
    style_size: int = 512,
    num_workers: int = 4,
    log_interval: int = 100,
    device: Optional[str] = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    name = style_name or Path(style_image_path).stem
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, f"{name}.pth")

    style_img = load_style_image(style_image_path, style_size, device)

    dataset = ContentDataset(content_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    print(f"Content dataset: {len(dataset)} images, style: {name}")

    model = FastStyleNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = PerceptualLoss().to(device)

    style_grams = loss_fn.precompute_style_grams(style_img)
    print(f"Precomputed style Gram matrices for {len(style_grams)} layers")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in loader:
            content = batch.to(device)
            generated = model(content)

            losses = loss_fn(
                generated, content, style_grams,
                content_w=content_weight, style_w=style_weight, tv_w=tv_weight,
            )
            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            global_step += 1
            if global_step % log_interval == 0:
                print(
                    f"Epoch {epoch} step {global_step} | "
                    f"total={losses['total'].item():.2f} "
                    f"content={losses['content'].item():.4f} "
                    f"style={losses['style'].item():.2f} "
                    f"tv={losses['tv'].item():.4f}"
                )

    torch.save({"model_state_dict": model.state_dict(), "style_name": name}, out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Johnson fast style transfer network")
    parser.add_argument("--config", default=None)
    parser.add_argument("--style", required=True, help="Path to style image")
    parser.add_argument("--content-dir", default="data/dtd/images")
    parser.add_argument("--output-dir", default="checkpoints/fast_style")
    parser.add_argument("--style-name", default=None)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument("--style-weight", type=float, default=1e5)
    parser.add_argument("--tv-weight", type=float, default=1e-6)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train(
        style_image_path=args.style,
        content_dir=cfg.get("content_dir", args.content_dir),
        output_dir=cfg.get("output_dir", args.output_dir),
        style_name=args.style_name,
        epochs=cfg.get("epochs", args.epochs),
        batch_size=cfg.get("batch_size", args.batch_size),
        lr=cfg.get("lr", args.lr),
        content_weight=cfg.get("content_weight", args.content_weight),
        style_weight=cfg.get("style_weight", args.style_weight),
        tv_weight=cfg.get("tv_weight", args.tv_weight),
        image_size=cfg.get("image_size", args.image_size),
        device=args.device,
    )


if __name__ == "__main__":
    main()
