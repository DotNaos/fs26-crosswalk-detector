import json
from pathlib import Path

from crosswalk_detector.sam3_results import summarize_sam3_results


def test_summarize_sam3_results_counts_scores_and_masks(tmp_path: Path) -> None:
    results = tmp_path / "results"
    mask = results / "masks" / "shard-a" / "tile-a.png"
    mask.parent.mkdir(parents=True)
    mask.write_bytes(b"png")
    (results / "shard-0000.result.json").write_text(
        json.dumps(
            {
                "summary": {"total": 3, "crosswalk": 1, "no_crosswalk": 2},
                "tiles": [
                    {"tile_id": "tile-a", "label": "crosswalk", "score": 0.65, "mask_artifact": {"path": str(mask)}},
                    {"tile_id": "tile-b", "label": "no_crosswalk", "score": 0.0},
                    {"tile_id": "tile-c", "label": "no_crosswalk", "score": 0.0},
                ],
            }
        ),
        encoding="utf8",
    )

    summary = summarize_sam3_results(results, expected_shards=2)

    assert summary.result_files == 1
    assert summary.remaining_shards == 1
    assert summary.tiles == 3
    assert summary.crosswalk == 1
    assert summary.no_crosswalk == 2
    assert summary.masks == 1
    assert summary.score_bins[">=0.6"] == 1
    assert summary.score_bins[">=0.7"] == 0
    assert summary.mask_consistent


def test_summarize_sam3_results_flags_bad_mask_refs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    (results / "shard-0000.result.json").write_text(
        json.dumps(
            {
                "summary": {"total": 2, "crosswalk": 1, "no_crosswalk": 1},
                "tiles": [
                    {"tile_id": "tile-a", "label": "crosswalk", "score": 0.5},
                    {"tile_id": "tile-b", "label": "no_crosswalk", "score": 0.0, "mask_artifact": {"path": "bad.png"}},
                ],
            }
        ),
        encoding="utf8",
    )

    summary = summarize_sam3_results(results)

    assert summary.missing_crosswalk_mask_refs == 1
    assert summary.bad_no_crosswalk_mask_refs == 1
    assert not summary.mask_consistent
