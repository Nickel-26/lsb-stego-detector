from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.cnn import SRNetLite, n_params  # noqa: E402
from src.models.factory import build_cnn  # noqa: E402


def test_srnetlite_forward_shape():
    model = SRNetLite()
    model.eval()
    x = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2,)
    assert n_params(model) > 0


def test_highpass_layer_is_frozen():
    model = SRNetLite()
    assert not model.highpass.weight.requires_grad


def test_factory_builds_transfer_without_pretrained_weights():
    model = build_cnn("resnet18", pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape[0] == 1


def test_srnetlite_accepts_rgb():
    model = SRNetLite(in_channels=3)
    model.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2,)


def test_dataloader_provides_rgb(synthetic_dataset):
    from src.data.dataset import build_dataloaders

    manifest, _, _ = synthetic_dataset
    loaders = build_dataloaders(manifest, size=64, batch_size=2, num_workers=0,
                                in_channels=3)
    x, y = next(iter(loaders["train"]))
    assert x.shape[1] == 3
    assert y.shape[0] == x.shape[0]
