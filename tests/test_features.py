from __future__ import annotations

import numpy as np

from src.features import (
    chi_square_attack,
    combined_feature_length,
    combined_feature_length_color,
    extract_features,
    extract_features_color,
    lsb_features,
    rs_analysis,
    spam_features,
    srm_features,
)
from src.features.statistical import chi_square_blocks


def test_feature_shapes_and_finiteness(paired_images):
    cover, _ = paired_images
    feats = extract_features(cover)
    assert feats["vector"].shape == (combined_feature_length(),)
    assert np.isfinite(feats["vector"]).all()
    assert spam_features(cover).shape[0] == 2 * 7 ** 3
    assert srm_features(cover).ndim == 1


def test_lsb_asymmetry_drops_after_embedding(paired_images):
    cover, stego = paired_images
    a = lsb_features(cover)["lsb_even_odd_asymmetry"]
    b = lsb_features(stego)["lsb_even_odd_asymmetry"]
    assert b < a


def test_rs_score_increases_after_embedding(paired_images):
    cover, stego = paired_images
    assert rs_analysis(stego)["rs_score"] > rs_analysis(cover)["rs_score"]


def test_chi_square_full_embedding_looks_stego(paired_images):
    cover, _ = paired_images
    rng = np.random.default_rng(0)
    full = (cover & 0xFE) | rng.integers(0, 2, cover.shape).astype(np.uint8)
    p_cover = chi_square_attack(cover)["chi2_p_value"]
    p_stego = chi_square_attack(full)["chi2_p_value"]
    # A single-image p-value is ~Uniform(0,1) under H0, so compare the statistic
    # and use the more stable per-block mean.
    assert chi_square_attack(full)["chi2_statistic"] < chi_square_attack(cover)["chi2_statistic"]
    assert p_stego > p_cover
    assert chi_square_blocks(full) > 0.3
    assert chi_square_blocks(full) > chi_square_blocks(cover)


def test_feature_vector_invariant_to_image_size():
    rng = np.random.default_rng(1)
    for size in (64, 96, 128):
        img = rng.integers(0, 256, (size, size), dtype=np.uint8)
        assert extract_features(img)["vector"].shape == (combined_feature_length(),)


def test_color_feature_vector_stacks_channels(paired_images):
    cover, _ = paired_images
    rgb = np.stack([cover, cover, cover], axis=-1)
    feats = extract_features_color(rgb)
    assert feats["vector"].shape == (combined_feature_length_color(),)
    assert feats["vector"].shape[0] == 3 * combined_feature_length()
    assert "rs_score_R" in feats["stat"]
    assert "lsb_even_odd_asymmetry_G" in feats["lsb"]
