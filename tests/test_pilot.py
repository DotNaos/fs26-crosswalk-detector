import importlib.util

import pytest

from crosswalk_detector import build_pilot_summary, load_pilot_config, raw_size_mb
from crosswalk_detector.pilot import (
    build_full_capped_dataset,
    crosswalk_score,
    derive_split_targets,
    export_balanced_seed_set,
    iter_city_tile_jobs,
    repo_root,
    sliding_window_count,
)


ROOT = repo_root()


def test_raw_size_mb_uses_current_planning_estimate() -> None:
    assert raw_size_mb(10.0) == 550.0


def test_sliding_window_count_matches_documented_examples() -> None:
    assert sliding_window_count(1.0, 25, 25) == 1600
    assert sliding_window_count(1.0, 25, 10) == 9604
    assert sliding_window_count(1.0, 25, 5) == 38416


def test_local_debug_config_stays_within_budget() -> None:
    config = load_pilot_config(ROOT / "configs/local-debug.json")

    summary = build_pilot_summary(config)

    assert summary["totals"]["raw_gb"] == 0.55
    assert summary["totals"]["within_budget"] is True
    assert summary["target_per_class"] == 64
    assert summary["split_targets"] == {"train": 45, "val": 10, "test": 9}


def test_local_debug_config_yields_small_tile_fetch_grid() -> None:
    config = load_pilot_config(ROOT / "configs/local-debug.json")

    jobs = iter_city_tile_jobs(config.cities[0], config.wmts_zoom, config.wmts_tile_radius)

    assert len(jobs) == 25
    assert jobs[0]["zoom"] == 20


def test_export_balanced_seed_set_uses_equal_class_counts() -> None:
    review_csv = ROOT / "data/processed/local-debug-v1/reviews/manual-review-v1.csv"
    if not review_csv.exists():
        pytest.skip("local debug review fixture is not available")

    split_targets = derive_split_targets(64, {"train": 0.7, "val": 0.15, "test": 0.15})
    summary = export_balanced_seed_set(
        review_csv,
        export_name="balanced-seed-test",
        target_per_class=64,
        split_targets=split_targets,
    )

    assert summary["positive_count"] == summary["negative_count"] == 5
    assert summary["target_per_class"] == 64
    assert summary["is_target_full"] is False
    assert summary["actual_split_caps"] == {"train": 3, "val": 0, "test": 2}


def test_derive_split_targets_sums_to_target_per_class() -> None:
    assert derive_split_targets(50, {"train": 0.7, "val": 0.15, "test": 0.15}) == {
        "train": 35,
        "val": 8,
        "test": 7,
    }


def test_crosswalk_score_distinguishes_known_positive_from_known_negative() -> None:
    if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("PIL") is None:
        return

    positive = ROOT / "data/raw/local-debug-v1/wmts-3857-z20/zurich/20/549168/367195.jpeg"
    negative = ROOT / "data/raw/local-debug-v1/wmts-3857-z20/chur/20/552052/369445.jpeg"
    if not positive.exists() or not negative.exists():
        pytest.skip("local debug image fixtures are not available")

    assert crosswalk_score(positive) > crosswalk_score(negative)


def test_full_build_uses_unique_raw_images_only() -> None:
    if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("PIL") is None:
        return

    review_csv = ROOT / "data/processed/local-debug-v1/reviews/manual-review-v1.csv"
    if not review_csv.exists():
        pytest.skip("local debug review fixture is not available")

    summary = build_full_capped_dataset(
        ROOT / "configs/local-debug.json",
        review_csv,
        export_name="full-capped-test-raw-only",
    )

    assert summary["positive_count"] == summary["negative_count"] == 31
    assert summary["actual_split_caps"] == {"train": 12, "val": 10, "test": 9}
    assert summary["available_raw_per_split"] == {"train": 25, "val": 25, "test": 25}
    assert summary["model_name"] == "model-centroid-v1"
    assert summary["is_target_full"] is False
