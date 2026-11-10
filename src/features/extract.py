"""Combine all feature groups into one flat vector + interpretable extras."""

from __future__ import annotations

from typing import Any

import numpy as np

from .lsb import lsb_feature_names, lsb_feature_vector, lsb_features
from .spam import spam_feature_length, spam_features
from .srm import srm_feature_length, srm_features
from .statistical import statistical_features


def combined_feature_length(spam_t: int = 3, srm_t: int = 2) -> int:
    return spam_feature_length(spam_t) + srm_feature_length(srm_t) + len(lsb_feature_names)


def feature_names(spam_t: int = 3, srm_t: int = 2) -> list[str]:
    names = [f"spam_{i}" for i in range(spam_feature_length(spam_t))]
    names += [f"srm_{i}" for i in range(srm_feature_length(srm_t))]
    names += list(lsb_feature_names)
    return names


def combined_feature_length_color(spam_t: int = 3, srm_t: int = 2,
                                  channels: int = 3) -> int:
    return channels * combined_feature_length(spam_t, srm_t)


def feature_names_color(spam_t: int = 3, srm_t: int = 2,
                        channels: str = "RGB") -> list[str]:
    base = feature_names(spam_t, srm_t)
    names: list[str] = []
    for ch in channels:
        names += [f"{ch}_{n}" for n in base]
    return names


def extract_features(gray: np.ndarray, spam_t: int = 3, srm_t: int = 2) -> dict[str, Any]:
    """Return {'vector': np.ndarray, 'spam','srm','lsb','stat'} for one image."""
    spam = spam_features(gray, spam_t)
    srm = srm_features(gray, srm_t)
    lsb_vec = lsb_feature_vector(gray)
    lsb = lsb_features(gray)
    stat = statistical_features(gray)
    vector = np.concatenate([spam, srm, lsb_vec]).astype(np.float32)
    return {"vector": vector, "spam": spam, "srm": srm, "lsb": lsb,
            "lsb_vector": lsb_vec, "stat": stat}


def _merge_channel_dicts(dicts: list[dict[str, float]], channels: str) -> dict[str, float]:
    merged: dict[str, float] = {}
    for key in dicts[0]:
        merged[key] = float(np.mean([d[key] for d in dicts]))
        for ch, d in zip(channels, dicts):
            merged[f"{key}_{ch}"] = float(d[key])
    return merged


def extract_features_color(rgb: np.ndarray, spam_t: int = 3,
                           srm_t: int = 2) -> dict[str, Any]:
    """Per-channel SPAM/SRM/LSB features for RGB LSB steganography.

    LSB replacement in colour images is spread over R/G/B; converting to
    grayscale mixes the bit-planes and destroys most of the signal, so colour
    models must see each channel separately.
    """
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    channels = "RGB"[: rgb.shape[2]]
    parts = [extract_features(rgb[:, :, i], spam_t, srm_t) for i in range(rgb.shape[2])]
    vector = np.concatenate([p["vector"] for p in parts]).astype(np.float32)
    return {
        "vector": vector,
        "parts": parts,
        "stat": _merge_channel_dicts([p["stat"] for p in parts], channels),
        "lsb": _merge_channel_dicts([p["lsb"] for p in parts], channels),
    }
