"""
VGG-19 Gram matrix neural style transfer — Gatys et al. (2016).

Optimizes the input image directly using LBFGS to minimize:
  L_total = α * L_content + β * L_style + γ * L_tv

Content layers: relu4_2
Style layers:   relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights
from typing import Dict, List, Optional


CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]

# Map readable names to VGG-19 Sequential layer indices
VGG19_LAYER_MAP = {
    "relu1_1": 1,  "relu1_2": 3,
    "relu2_1": 6,  "relu2_2": 8,
    "relu3_1": 11, "relu3_2": 13, "relu3_3": 15, "relu3_4": 17,
    "relu4_1": 20, "relu4_2": 22, "relu4_3": 24, "relu4_4": 26,
    "relu5_1": 29, "relu5_2": 31, "relu5_3": 33, "relu5_4": 35,
}


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.size()
    f = feat.view(b, c, h * w)
    g = torch.bmm(f, f.transpose(1, 2))
    return g / (c * h * w)


class VGGFeatureExtractor(nn.Module):
    def __init__(self, layer_names: List[str]):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.layer_names = layer_names
        max_idx = max(VGG19_LAYER_MAP[n] for n in layer_names)
        self.slices = nn.Sequential(*list(vgg.children())[: max_idx + 1])

        # Replace MaxPool with AvgPool for smoother gradients
        for i, layer in enumerate(self.slices):
            if isinstance(layer, nn.MaxPool2d):
                self.slices[i] = nn.AvgPool2d(2, 2)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs: Dict[str, torch.Tensor] = {}
        target_indices = {VGG19_LAYER_MAP[n]: n for n in self.layer_names}
        for i, layer in enumerate(self.slices):
            x = layer(x)
            if i in target_indices:
                outputs[target_indices[i]] = x
        return outputs


class VGGStyleTransfer:
    """Gatys-style optimization-based neural style transfer."""

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        tv_weight: float = 1e-4,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        all_layers = list(set(CONTENT_LAYERS + STYLE_LAYERS))
        self.extractor = VGGFeatureExtractor(all_layers).to(self.device)

    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        return (img - mean) / std

    @staticmethod
    def _tv_loss(img: torch.Tensor) -> torch.Tensor:
        return (
            torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
            + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        )

    def transfer(
        self,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        num_steps: int = 500,
        init: str = "content",
        style_layer_weights: Optional[Dict[str, float]] = None,
        callback=None,
    ) -> torch.Tensor:
        content_img = content_img.to(self.device)
        style_img = style_img.to(self.device)

        # Precompute targets
        with torch.no_grad():
            content_feats = self.extractor(self._normalize(content_img))
            style_feats = self.extractor(self._normalize(style_img))
            content_targets = {k: content_feats[k].detach() for k in CONTENT_LAYERS}
            style_targets = {k: gram_matrix(style_feats[k]).detach() for k in STYLE_LAYERS}

        w = style_layer_weights or {k: 1.0 / len(STYLE_LAYERS) for k in STYLE_LAYERS}

        if init == "content":
            canvas = content_img.clone().requires_grad_(True)
        elif init == "style":
            canvas = style_img.clone().requires_grad_(True)
        else:
            canvas = torch.rand_like(content_img, requires_grad=True)

        optimizer = torch.optim.LBFGS([canvas], max_iter=20, lr=1.0)

        step = [0]

        def closure():
            canvas.data.clamp_(0, 1)
            optimizer.zero_grad()
            feats = self.extractor(self._normalize(canvas))

            c_loss = sum(
                F.mse_loss(feats[k], content_targets[k]) for k in CONTENT_LAYERS
            )
            s_loss = sum(
                w[k] * F.mse_loss(gram_matrix(feats[k]), style_targets[k])
                for k in STYLE_LAYERS
            )
            tv_loss = self._tv_loss(canvas)

            loss = self.content_weight * c_loss + self.style_weight * s_loss + self.tv_weight * tv_loss
            loss.backward()

            step[0] += 1
            if callback and step[0] % 50 == 0:
                callback(step[0], loss.item(), canvas.detach().clamp(0, 1))
            return loss

        for _ in range(num_steps // 20):
            optimizer.step(closure)

        canvas.data.clamp_(0, 1)
        return canvas.detach()
