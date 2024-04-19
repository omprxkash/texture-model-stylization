"""
Fast Neural Style Transfer — Johnson et al. (2016).

Architecture: Conv → [ResBlock × 9] → Upsample → Conv
Uses Instance Normalization (not Batch Norm) for stable single-image inference.
Trained once per style image; inference is a single forward pass (~10ms on GPU).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG16_Weights
from typing import Dict, List, Optional


class ConvINReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_ch, out_ch, kernel, stride),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, upsample: int = 2):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel // 2),
            nn.Conv2d(in_ch, out_ch, kernel, stride),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.block(x)


class FastStyleNetwork(nn.Module):
    """Encoder–residual–decoder network for real-time style transfer."""

    def __init__(self, num_res_blocks: int = 9):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvINReLU(3, 32, 9, stride=1, padding=4),
            ConvINReLU(32, 64, 3, stride=2, padding=1),
            ConvINReLU(64, 128, 3, stride=2, padding=1),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(128) for _ in range(num_res_blocks)])
        self.decoder = nn.Sequential(
            UpsampleConv(128, 64, 3, upsample=2),
            UpsampleConv(64, 32, 3, upsample=2),
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, 9),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        feat = self.residuals(feat)
        out = self.decoder(feat)
        return (out + 1.0) / 2.0  # map [-1,1] → [0,1]

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "FastStyleNetwork":
        model = cls()
        state = torch.load(path, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        return model


class PerceptualLoss(nn.Module):
    """VGG-16 perceptual + Gram-matrix style loss for training FastStyleNetwork."""

    CONTENT_LAYER = "relu3_3"
    STYLE_LAYERS = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    VGG16_IDX = {"relu1_2": 3, "relu2_2": 8, "relu3_3": 15, "relu4_3": 22}

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        max_idx = max(self.VGG16_IDX.values())
        self.slices = nn.Sequential(*list(vgg.children())[: max_idx + 1])
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats: Dict[str, torch.Tensor] = {}
        target_map = {v: k for k, v in self.VGG16_IDX.items()}
        for i, layer in enumerate(self.slices):
            x = layer(x)
            if i in target_map:
                feats[target_map[i]] = x
        return feats

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        f = feat.view(b, c, h * w)
        return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)

    def forward(
        self,
        generated: torch.Tensor,
        content: torch.Tensor,
        style_grams: Dict[str, torch.Tensor],
        content_w: float = 1.0,
        style_w: float = 1e5,
        tv_w: float = 1e-6,
    ) -> Dict[str, torch.Tensor]:
        g_feats = self._extract(self._normalize(generated))
        c_feats = self._extract(self._normalize(content))

        c_loss = F.mse_loss(g_feats[self.CONTENT_LAYER], c_feats[self.CONTENT_LAYER].detach())
        s_loss = sum(
            F.mse_loss(self._gram(g_feats[k]), style_grams[k])
            for k in self.STYLE_LAYERS
        ) / len(self.STYLE_LAYERS)

        tv_loss = (
            torch.mean(torch.abs(generated[:, :, :, :-1] - generated[:, :, :, 1:]))
            + torch.mean(torch.abs(generated[:, :, :-1, :] - generated[:, :, 1:, :]))
        )

        total = content_w * c_loss + style_w * s_loss + tv_w * tv_loss
        return {"total": total, "content": c_loss, "style": s_loss, "tv": tv_loss}

    def precompute_style_grams(self, style_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            feats = self._extract(self._normalize(style_img))
            return {k: self._gram(feats[k]).detach() for k in self.STYLE_LAYERS}
