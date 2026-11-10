"""A compact SRM-style residual feature set.

The full Spatial Rich Model (Fridrich & Kodovsky) has ~34k features. Here we use
a small, fast subset: a bank of high-pass residuals, quantised to [-T, T], with
a second-order co-occurrence histogram per residual. This is enough to form a
strong CPU baseline while keeping extraction practical for tens of thousands of
images.
"""

from __future__ import annotations

import cv2
import numpy as np


def _filters() -> list[np.ndarray]:
    f: list[np.ndarray] = []

    def add(kernel: list[list[float]]) -> None:
        f.append(np.asarray(kernel, dtype=np.float32))

    # 1st order derivatives
    add([[-1, 1]])
    add([[-1], [1]])
    # 2nd order derivatives
    add([[1, -2, 1]])
    add([[1], [-2], [1]])
    # 3rd order derivatives
    add([[-1, 3, -3, 1]])
    add([[-1], [3], [-3], [1]])
    # 3x3 residuals
    add([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])          # square
    add([[-1, 2, -1], [0, 0, 0], [1, -2, 1]])            # horizontal edge
    add([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]])            # vertical edge
    add([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])           # laplacian
    add([[-2, 3, -2], [3, -6, 3], [-2, 3, -2]])          # KB-like
    add([[-1, 2, -2, 2, -1]])
    return f


SRM_FILTERS = _filters()


def srm_feature_length(threshold: int = 2) -> int:
    s = 2 * threshold + 1
    return len(SRM_FILTERS) * s * s


def _cooccurrence(q: np.ndarray, s: int) -> np.ndarray:
    counts = np.zeros(s * s, dtype=np.float64)
    if q.shape[1] >= 2:
        a = q[:, :-1].ravel()
        b = q[:, 1:].ravel()
        counts += np.bincount(a * s + b, minlength=s * s)
    if q.shape[0] >= 2:
        a = q[:-1, :].ravel()
        b = q[1:, :].ravel()
        counts += np.bincount(a * s + b, minlength=s * s)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts.astype(np.float32)


def srm_features(gray: np.ndarray, threshold: int = 2) -> np.ndarray:
    s = 2 * threshold + 1
    g = gray.astype(np.float32)
    feats: list[np.ndarray] = []
    for kernel in SRM_FILTERS:
        residual = cv2.filter2D(g, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        q = np.clip(np.rint(residual), -threshold, threshold).astype(np.int32) + threshold
        feats.append(_cooccurrence(q, s))
    return np.concatenate(feats, axis=0)
