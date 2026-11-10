"""Compact SRNet-style CNN for spatial LSB steganalysis.

Key design choices:
* fixed high-pass (SRM-like) first layer, frozen, so the network sees residuals;
* no aggressive early pooling, to preserve the weak LSB signal;
* global average pooling + single logit.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..features.srm import SRM_FILTERS


def _highpass_kernels() -> list[np.ndarray]:
    kernels = [k for k in SRM_FILTERS if k.shape == (3, 3)]
    # Pad any 5-wide 1-D kernel into a 3x3? Not needed; ensure a minimum bank.
    if len(kernels) < 4:
        kernels += [np.array([[-1, 1, 0], [0, 0, 0], [0, 0, 0]], np.float32)]
    return kernels


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRNetLite(nn.Module):
    def __init__(self, dropout: float = 0.2, in_channels: int = 1) -> None:
        super().__init__()
        kernels = _highpass_kernels()
        n = len(kernels)
        self.in_channels = in_channels
        # One independent residual filter bank per input channel.
        self.highpass = nn.Conv2d(in_channels, n * in_channels, 3, padding=1,
                                  groups=in_channels, bias=False)
        with torch.no_grad():
            w = torch.from_numpy(np.stack(kernels)).float()  # (n, 3, 3)
            w = w / (w.abs().sum(dim=(1, 2), keepdim=True) + 1e-8)
            w = w.repeat(in_channels, 1, 1)                  # (n*C, 3, 3)
            self.highpass.weight.copy_(w.unsqueeze(1))       # (n*C, 1, 3, 3)
        for p in self.highpass.parameters():
            p.requires_grad = False

        n_in = n * in_channels
        self.features = nn.Sequential(
            ConvBlock(n_in, 16),
            ConvBlock(16, 16),
            ConvBlock(16, 32, stride=2),
            ConvBlock(32, 32),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 64),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.highpass(x)
        x = self.features(x)
        return self.head(x).squeeze(1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
