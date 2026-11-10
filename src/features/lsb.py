"""Least-significant-bit plane statistics.

These capture direct regularities of the LSB plane that LSB replacement tends
to randomise: run lengths, entropy, even/odd histogram asymmetry and spatial
autocorrelation.
"""

from __future__ import annotations

import numpy as np


def lsb_plane(gray: np.ndarray) -> np.ndarray:
    """Return the LSB plane as uint8 values in {0, 1}."""
    return (gray & 1).astype(np.uint8)


def _entropy(plane: np.ndarray) -> float:
    p1 = float(plane.mean())
    p0 = 1.0 - p1
    ent = 0.0
    for p in (p0, p1):
        if p > 0:
            ent -= p * np.log2(p)
    return ent


def _even_odd_asymmetry(gray: np.ndarray) -> float:
    """Sum of |hist[2k] - hist[2k+1]| normalised by pixel count.

    LSB *replacement* equalises each even/odd pair -> this drops toward 0.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    even = hist[0::2]
    odd = hist[1::2]
    return float(np.abs(even - odd).sum() / gray.size)


def _autocorr(plane: np.ndarray, axis: int) -> float:
    a = plane.astype(np.float64)
    if axis == 1:  # horizontal neighbour correlation
        x, y = a[:, :-1].ravel(), a[:, 1:].ravel()
    else:          # vertical
        x, y = a[:-1, :].ravel(), a[1:, :].ravel()
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / denom) if denom > 0 else 0.0


def _run_length(plane: np.ndarray) -> float:
    """Mean number of same-value runs (highly compressible planes have fewer runs)."""
    flat = plane.ravel()
    if flat.size < 2:
        return 0.0
    runs = int(1 + np.count_nonzero(flat[1:] != flat[:-1]))
    return runs / flat.size


def lsb_features(gray: np.ndarray) -> dict[str, float]:
    plane = lsb_plane(gray)
    return {
        "lsb_mean": float(plane.mean()),
        "lsb_entropy": _entropy(plane),
        "lsb_even_odd_asymmetry": _even_odd_asymmetry(gray),
        "lsb_run_length": _run_length(plane),
        "lsb_autocorr_h": _autocorr(plane, 1),
        "lsb_autocorr_v": _autocorr(plane, 0),
    }


_LSB_KEYS = (
    "lsb_mean",
    "lsb_entropy",
    "lsb_even_odd_asymmetry",
    "lsb_run_length",
    "lsb_autocorr_h",
    "lsb_autocorr_v",
)


def lsb_feature_vector(gray: np.ndarray) -> np.ndarray:
    feats = lsb_features(gray)
    return np.asarray([feats[k] for k in _LSB_KEYS], dtype=np.float32)


lsb_feature_names = list(_LSB_KEYS)
