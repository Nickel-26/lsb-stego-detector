"""Optional ImageNet transfer-learning baselines (3-channel input)."""

from __future__ import annotations

import torch.nn as nn

TRANSFER_ARCHS = {"efficientnet_v2_s", "resnet18"}


def is_transfer(arch: str) -> bool:
    return arch in TRANSFER_ARCHS


def build_transfer(arch: str, pretrained: bool = True) -> nn.Module:
    import torchvision.models as tvm

    if arch == "resnet18":
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "efficientnet_v2_s":
        weights = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"unknown transfer arch: {arch}")
