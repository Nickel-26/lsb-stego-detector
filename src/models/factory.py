"""Model factory shared by training and inference."""

from __future__ import annotations

import torch.nn as nn

from .cnn import SRNetLite
from .transfer import TRANSFER_ARCHS, build_transfer, is_transfer


def build_cnn(arch: str = "srnet_lite", pretrained: bool = True,
              in_channels: int = 1) -> nn.Module:
    if arch == "srnet_lite":
        return SRNetLite(in_channels=in_channels)
    if arch in TRANSFER_ARCHS:
        return build_transfer(arch, pretrained=pretrained)
    raise ValueError(f"unknown arch: {arch}")


__all__ = ["build_cnn", "is_transfer", "TRANSFER_ARCHS"]
