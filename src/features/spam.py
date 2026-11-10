"""SPAM: Subtractive Pixel Adjacency Model features.

Second-order Markov transition probabilities of clipped pixel differences,
computed along rows and columns. With the default T=3 this yields
2 * (2T+1)^3 = 686 features, matching the classic SPAM dimensionality.
"""

from __future__ import annotations

import numpy as np


def spam_feature_length(threshold: int = 3) -> int:
    s = 2 * threshold + 1
    return 2 * s ** 3


def _markov2(diff: np.ndarray, t: int) -> np.ndarray:
    s = 2 * t + 1
    n_states = s ** 3
    if diff.shape[1] < 3:
        return np.zeros(n_states, dtype=np.float32)
    d = np.clip(np.rint(diff), -t, t).astype(np.int32) + t
    a, b, c = d[:, :-2], d[:, 1:-1], d[:, 2:]
    idx = (a * s + b) * s + c
    counts = np.bincount(idx.ravel(), minlength=n_states).astype(np.float64)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts.astype(np.float32)


def spam_features(gray: np.ndarray, threshold: int = 3) -> np.ndarray:
    g = gray.astype(np.float32)
    h = np.diff(g, axis=1)   # horizontal neighbour differences
    v = np.diff(g, axis=0)   # vertical neighbour differences
    return np.concatenate([_markov2(h, threshold), _markov2(v, threshold)], axis=0)
