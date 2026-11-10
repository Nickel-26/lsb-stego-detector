"""Classic statistical steganalysis attacks (no machine learning).

* chi-square attack          - detects LSB replacement via even/odd histogram pairs
* RS analysis                - regular/singular group analysis
* sample-pair heuristic      - closeness statistics of neighbouring pixels

These double as a dataset sanity check: if they cannot separate cover from
stego, the labels or the dataset are wrong.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.stats import chi2 as _chi2_dist
except Exception:  # pragma: no cover
    _chi2_dist = None


# --------------------------------------------------------------------------- #
# Chi-square attack (Westfeld & Pfitzmann)
# --------------------------------------------------------------------------- #
def chi_square_attack(gray: np.ndarray) -> dict[str, float]:
    """Test whether even/odd histogram pairs are equalised (=> LSB replacement).

    Returns the chi-square statistic, degrees of freedom and the p-value.
    Small p-value  -> pairs differ  -> cover-like.
    Large p-value  -> pairs equal   -> stego-like.
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    even = hist[0::2]
    odd = hist[1::2]
    expected = (even + odd) / 2.0
    mask = expected > 0
    statistic = float((((even - expected)[mask] ** 2 + (odd - expected)[mask] ** 2) / expected[mask]).sum())
    df = int(mask.sum())
    if _chi2_dist is not None and df > 0:
        p_value = float(1.0 - _chi2_dist.cdf(statistic, df))
    else:  # normal approximation fallback
        from scipy.stats import norm  # local import; scipy is a core dep

        z = (statistic - df) / np.sqrt(2 * df) if df else 0.0
        p_value = float(1.0 - norm.cdf(z))
    return {"chi2_statistic": statistic, "chi2_df": df, "chi2_p_value": p_value}


def chi_square_blocks(gray: np.ndarray, block: int = 32) -> float:
    """Mean chi-square p-value over non-overlapping blocks (more stable)."""
    h, w = gray.shape
    ps: list[float] = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            ps.append(chi_square_attack(gray[y:y + block, x:x + block])["chi2_p_value"])
    return float(np.mean(ps)) if ps else 0.0


# --------------------------------------------------------------------------- #
# RS analysis (Fridrich & Goljan)
# --------------------------------------------------------------------------- #
def _flip_f1(x: np.ndarray) -> np.ndarray:
    """LSB flip: even -> +1, odd -> -1  (x XOR 1)."""
    return x ^ 1


def _flip_fneg(x: np.ndarray) -> np.ndarray:
    """Dual flip: even -> -1, odd -> +1, clamped to [0, 255]."""
    out = np.where(x % 2 == 0, x - 1, x + 1)
    return np.clip(out, 0, 255).astype(x.dtype)


def _discriminant(groups: np.ndarray) -> np.ndarray:
    """f(g) = sum of absolute neighbouring differences within each group of 4."""
    return np.abs(np.diff(groups, axis=1)).sum(axis=1)


def rs_analysis(gray: np.ndarray, mask: tuple[int, ...] = (1, 0, 1, 0)) -> dict[str, float]:
    """Regular/Singular group analysis for LSB replacement / matching.

    Returns fractions and a heuristic score. ``rs_score`` grows with embedding;
    ``rs_estimated_rate`` is a rough (clipped) payload estimate.
    """
    h, w = gray.shape
    gray = gray.astype(np.int32)
    mask = np.asarray(mask, dtype=bool)
    n = mask.size

    # Build non-overlapping groups of 4 along rows and columns.
    cols = w - (w % n)
    rows = h - (h % n)
    row_groups = gray[:rows, :cols].reshape(rows, cols // n, n)
    col_groups = gray.T[:cols, :rows].reshape(cols, rows // n, n).transpose(1, 0, 2)

    def stats(groups: np.ndarray) -> tuple[float, float, float, float]:
        f0 = _discriminant(groups)
        m = mask.reshape(1, -1)
        g1 = groups.copy()
        g1[:, mask] = _flip_f1(groups[:, mask])
        f1 = _discriminant(g1)
        gm = groups.copy()
        gm[:, mask] = _flip_fneg(groups[:, mask])
        fm = _discriminant(gm)
        total = groups.shape[0]
        reg_p = float(np.count_nonzero(f1 > f0) / total)
        sing_p = float(np.count_nonzero(f1 < f0) / total)
        reg_m = float(np.count_nonzero(fm > f0) / total)
        sing_m = float(np.count_nonzero(fm < f0) / total)
        return reg_p, sing_p, reg_m, sing_m

    rp1, sp1, rm1, sm1 = stats(row_groups.reshape(-1, n))
    rp2, sp2, rm2, sm2 = stats(col_groups.reshape(-1, n))
    rp = (rp1 + rp2) / 2
    sp = (sp1 + sp2) / 2
    rm = (rm1 + rm2) / 2
    sm = (sm1 + sm2) / 2

    score = (rm - sm) - (rp - sp)
    return {
        "rs_Rp": rp, "rs_Sp": sp, "rs_Rm": rm, "rs_Sm": sm,
        "rs_score": float(score),
        "rs_estimated_rate": float(np.clip(2.0 * score, 0.0, 1.0)),
    }


# --------------------------------------------------------------------------- #
# Sample-pair heuristic
# --------------------------------------------------------------------------- #
def sample_pair_score(gray: np.ndarray) -> dict[str, float]:
    """Close-pair statistics. LSB embedding increases the number of pairs whose
    values differ by exactly 1 among otherwise-smooth neighbours."""
    g = gray.astype(np.int32)
    hpairs = np.stack([g[:, :-1].ravel(), g[:, 1:].ravel()], axis=1)
    vpairs = np.stack([g[:-1, :].ravel(), g[1:, :].ravel()], axis=1)
    pairs = np.concatenate([hpairs, vpairs], axis=0)
    diff = np.abs(pairs[:, 0] - pairs[:, 1])
    close = float(np.count_nonzero(diff == 1) / pairs.shape[0])
    same_parity = float(np.count_nonzero((pairs[:, 0] & 1) == (pairs[:, 1] & 1)) / pairs.shape[0])
    return {"spa_close_ratio": close, "spa_same_parity_ratio": same_parity}


def statistical_features(gray: np.ndarray, block_p: int = 32) -> dict[str, float]:
    feats: dict[str, float] = {}
    feats.update(chi_square_attack(gray))
    feats["chi2_p_value_blockmean"] = chi_square_blocks(gray, block_p)
    feats.update(rs_analysis(gray))
    feats.update(sample_pair_score(gray))
    return feats
