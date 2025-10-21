"""
ResNet-50 texture classifier fine-tuned on DTD (47 classes).

Strategy:
  - Freeze conv1 through layer2 (low-level features remain pretrained)
  - Fine-tune layer3, layer4, and the new 47-class head
  - Use dropout(0.4) before the final FC to reduce overfitting on DTD's small per-class count
"""
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


NUM_CLASSES = 47


class TextureClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Freeze early layers
        for name, param in backbone.named_parameters():
            if any(name.startswith(p) for p in ("conv1", "bn1", "layer1", "layer2")):
                param.requires_grad = False

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        return self.classifier(feat)

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return torch.flatten(feat, 1)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "TextureClassifier":
        model = cls(pretrained=False)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model
