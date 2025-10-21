"""
Adaptive Instance Normalization (AdaIN) style transfer — Huang & Belongie (2017).

AdaIN(x, y) = σ(y) * (x - μ(x)) / σ(x) + μ(y)

Encoder: VGG-19 up to relu4_1 (frozen)
Decoder: Symmetric inverse of encoder (learned)
No spatial or temporal constraints — style is transferred in a single forward pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights
from typing import Optional, Tuple


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    """Align content feature statistics to style feature statistics."""
    assert content_feat.dim() == 4
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    normalized = (content_feat - c_mean) / c_std
    return s_std * normalized + s_mean


class VGGEncoder(nn.Module):
    """VGG-19 feature encoder — fixed, not trained."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features

        # Split into 4 slices for multi-scale feature extraction
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])   # relu1_1
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])  # relu2_1
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12]) # relu3_1
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])# relu4_1

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

    def forward(self, x: torch.Tensor, return_all: bool = False):
        x = self._normalize(x)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if return_all:
            return h1, h2, h3, h4
        return h4


class ReflectConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1, pad: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_ch, out_ch, kernel, stride),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdaINDecoder(nn.Module):
    """Learned decoder — mirrors the VGG-19 encoder structure."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ReflectConv(512, 256, 3, pad=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ReflectConv(256, 256, 3, pad=1),
            ReflectConv(256, 256, 3, pad=1),
            ReflectConv(256, 256, 3, pad=1),
            ReflectConv(256, 128, 3, pad=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ReflectConv(128, 128, 3, pad=1),
            ReflectConv(128, 64, 3, pad=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ReflectConv(64, 64, 3, pad=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AdaINStyleTransfer(nn.Module):
    """Complete AdaIN style transfer model (encoder + transfer + decoder)."""

    def __init__(self, device: Optional[str] = None):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = VGGEncoder().to(self.device)
        self.decoder = AdaINDecoder().to(self.device)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.encoder(x.to(self.device), return_all=True)

    def decode(self, feat: torch.Tensor) -> torch.Tensor:
        return self.decoder(feat).clamp(0, 1)

    def transfer(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """
        alpha: interpolation strength [0=content only, 1=full style transfer]
        """
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)

        with torch.no_grad():
            content_feat = self.encoder(content_img)
            style_feat = self.encoder(style_img)

        transferred = adain(content_feat, style_feat)
        # Interpolate between content and transferred feature
        feat = alpha * transferred + (1 - alpha) * content_feat
        return self.decode(feat)

    def forward(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        return self.transfer(content_img, style_img, alpha)

    @classmethod
    def from_checkpoint(cls, decoder_path: str, device: str = "cpu") -> "AdaINStyleTransfer":
        model = cls(device=device)
        state = torch.load(decoder_path, map_location=device)
        if "decoder_state_dict" in state:
            state = state["decoder_state_dict"]
        model.decoder.load_state_dict(state)
        model.eval()
        return model
