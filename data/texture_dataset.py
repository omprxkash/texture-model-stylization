"""
PyTorch Datasets for DTD texture classification and (texture, heightmap) diffusion training.
"""
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# 47 DTD categories in alphabetical order
DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed",
    "cracked", "crosshatched", "crystalline", "dotted", "fibrous", "flecked",
    "freckled", "frilly", "gauzy", "grid", "grooved", "honeycombed", "interlaced",
    "knitted", "lacelike", "lined", "marbled", "matted", "meshed", "paisley",
    "perforated", "pitted", "pleated", "polka-dotted", "porous", "potholed",
    "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified",
    "striped", "studded", "swirly", "veined", "waffled", "woven", "wrinkled", "zigzagged",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(DTD_CLASSES)}


def default_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def default_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DTDDataset(Dataset):
    """DTD dataset for 47-class texture classification.

    Expects DTD extracted at `root/images/<category>/*.jpg`.
    Splits are controlled by `split` in ['train', 'val', 'test', 'all'].
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform or (
            default_train_transform(image_size)
            if split == "train"
            else default_val_transform(image_size)
        )
        self.samples: List[Tuple[Path, int]] = []
        self._load_split()

    def _load_split(self):
        images_dir = self.root / "images"
        labels_dir = self.root / "labels"

        if self.split == "all" or not labels_dir.exists():
            for category in DTD_CLASSES:
                cat_dir = images_dir / category
                if cat_dir.exists():
                    for img_path in sorted(cat_dir.glob("*.jpg")):
                        self.samples.append((img_path, CLASS_TO_IDX[category]))
            return

        split_file = labels_dir / f"{self.split}1.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file) as f:
            for line in f:
                rel = line.strip()
                if not rel:
                    continue
                category = rel.split("/")[0]
                img_path = images_dir / rel
                if img_path.exists() and category in CLASS_TO_IDX:
                    self.samples.append((img_path, CLASS_TO_IDX[category]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label

    @staticmethod
    def get_class_names() -> List[str]:
        return DTD_CLASSES


class HeightmapDataset(Dataset):
    """Paired (texture image, heightmap) dataset for diffusion model fine-tuning.

    Expects a directory structure:
        root/textures/<name>.jpg
        root/heightmaps/<name>.png
    """

    def __init__(
        self,
        root: str,
        image_size: int = 512,
        augment: bool = True,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.pairs: List[Tuple[Path, Path]] = self._discover_pairs()

        shared_ops = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        if augment:
            shared_ops = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ] + shared_ops

        self.texture_transform = transforms.Compose(
            shared_ops + [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.heightmap_transform = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        tex_dir = self.root / "textures"
        hm_dir = self.root / "heightmaps"
        pairs = []
        if not tex_dir.exists():
            return pairs
        for tex_path in sorted(tex_dir.glob("*.jpg")) + sorted(tex_dir.glob("*.png")):
            hm_path = hm_dir / (tex_path.stem + ".png")
            if hm_path.exists():
                pairs.append((tex_path, hm_path))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        tex_path, hm_path = self.pairs[idx]
        texture = Image.open(tex_path).convert("RGB")
        heightmap = Image.open(hm_path).convert("L")
        return {
            "texture": self.texture_transform(texture),
            "heightmap": self.heightmap_transform(heightmap),
            "texture_path": str(tex_path),
        }


def build_dtd_loaders(
    dtd_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = DTDDataset(dtd_root, split="train", image_size=image_size)
    val_ds = DTDDataset(dtd_root, split="val", image_size=image_size)
    test_ds = DTDDataset(dtd_root, split="test", image_size=image_size)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader
