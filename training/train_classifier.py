"""
Train ResNet-50 texture classifier on DTD (47 classes).

Usage:
  python training/train_classifier.py --config configs/classifier.yaml
  python training/train_classifier.py --dtd-root data/dtd --epochs 30 --lr 1e-4
"""
import argparse
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.texture_dataset import build_dtd_loaders
from models.classifier import TextureClassifier


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train(
    dtd_root: str,
    output_dir: str = "checkpoints/classifier",
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
    dropout: float = 0.4,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dtd_loaders(
        dtd_root, batch_size=batch_size, num_workers=num_workers, image_size=image_size
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = TextureClassifier(dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = GradScaler()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }
            torch.save(ckpt, os.path.join(output_dir, "best_classifier.pth"))
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest accuracy: {test_acc:.3f} | loss: {test_loss:.4f}")
    print(f"Best val accuracy: {best_val_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-50 texture classifier on DTD")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--dtd-root", default="data/dtd")
    parser.add_argument("--output-dir", default="checkpoints/classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train(
        dtd_root=cfg.get("dtd_root", args.dtd_root),
        output_dir=cfg.get("output_dir", args.output_dir),
        epochs=cfg.get("epochs", args.epochs),
        lr=cfg.get("lr", args.lr),
        batch_size=cfg.get("batch_size", args.batch_size),
        image_size=cfg.get("image_size", args.image_size),
        num_workers=cfg.get("num_workers", args.num_workers),
        device=args.device,
    )


if __name__ == "__main__":
    main()
