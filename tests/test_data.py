from __future__ import annotations

from src.data.download_check import derive_group_id, leakage_report
from src.data.split import regroup


def test_group_id_links_cover_and_stego():
    assert derive_group_id("img_00001") == derive_group_id("img_00001_stego")
    assert derive_group_id("cat_001_cover") == derive_group_id("cat_001_lsb")


def test_group_id_handles_hf_variant_names():
    assert derive_group_id("1000_hr_0.25_hb_1_random") == derive_group_id("1000")
    assert derive_group_id("1000__v1") == derive_group_id("1000")


def test_manifest_counts_and_no_leakage(synthetic_dataset):
    _, _, df = synthetic_dataset
    report = leakage_report(df)
    assert report["groups_across_splits"] == 0
    counts = df.groupby(["split", "cls"]).size().to_dict()
    assert counts[("train", "clean")] == 40
    assert counts[("test", "stego")] == 12


def test_regroup_keeps_groups_together(synthetic_dataset):
    _, _, df = synthetic_dataset
    new = regroup(df, (0.7, 0.15, 0.15), seed=1)
    per_group = new.groupby("group_id")["split"].nunique()
    assert (per_group == 1).all()
