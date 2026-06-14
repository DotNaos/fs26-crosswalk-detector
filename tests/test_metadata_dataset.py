import json
from pathlib import Path

from crosswalk_detector.metadata_dataset import resolve_label_votes, validate_metadata_dataset
from crosswalk_detector.sam3_metadata import (
    build_sam3_shard_jobs,
    merge_sam3_metadata_dataset,
    prepare_sam3_metadata_dataset,
)


def _image_row(image_id: str, *, selected: bool = True) -> dict[str, object]:
    return {
        "image_id": image_id,
        "tile_id": image_id,
        "scene_id": "scene-a",
        "source_scene_id": "swissimage-scene-a",
        "perimeter_id": "p0000",
        "city": "Zurich",
        "split": "train",
        "row": 0,
        "col": 0,
        "bbox_mercator": [0, 0, 25, 25],
        "swisstopo": {
            "provider": "swisstopo",
            "product": "SWISSIMAGE",
            "access": "stac-cog",
            "crs": "EPSG:2056",
            "asset_id": "swissimage-demo",
            "resolution_m": 0.1,
        },
        "reconstruction": {
            "source_scene_id": "swissimage-scene-a",
            "row": 0,
            "col": 0,
            "tile_size_m": 25,
            "tile_bbox_mercator": [0, 0, 25, 25],
            "crop_px": {"left": 0, "top": 0, "width": 250, "height": 250},
            "relative_path": f"images/{image_id}.jpg",
        },
        "road_overlay_ref": {
            "overlay_id": "roads-v1",
            "perimeter_id": "p0000",
            "cell_id": image_id,
            "surface_ratio": 0.4,
        },
        "labels": [
            {
                "vote_id": f"sam3.1:{image_id}",
                "source": {
                    "source_id": "sam3.1",
                    "kind": "model",
                    "priority": 100,
                    "display_name": "SAM3.1",
                },
                "decision": "crosswalk",
                "confidence": 0.9,
                "created_at": "2026-05-15T00:00:00Z",
            }
        ],
        "resolved_label": {
            "decision": "crosswalk",
            "source_id": "sam3.1",
            "source_kind": "model",
            "resolved_by": "priority",
            "confidence": 0.9,
            "updated_at": "2026-05-15T00:00:00Z",
        },
        "review_state": "unreviewed",
        "selected_for_training": selected,
    }


def _write_dataset(root: Path, rows: list[dict[str, object]], selected_count: int | None = None) -> Path:
    dataset_root = root / "sam3-100k-v1"
    shard_path = dataset_root / "scenes" / "scene-a" / "perimeters" / "p0000" / "tiles.jsonl"
    shard_path.parent.mkdir(parents=True)
    shard_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf8")
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "format": "crosswalk-jsonl-v1",
                "dataset_id": "sam3-100k-v1",
                "run_name": "real-v2-sam3",
                "export_name": "sam3-balanced-100k-v1",
                "tile_count": len(rows),
                "selected_count": selected_count if selected_count is not None else sum(bool(row["selected_for_training"]) for row in rows),
                "shard_target_count": 2000,
                "shards": [
                    {
                        "shard_id": "scene-a-p0000",
                        "path": "scenes/scene-a/perimeters/p0000/tiles.jsonl",
                        "tile_count": len(rows),
                        "scene_id": "scene-a",
                        "perimeter_id": "p0000",
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf8",
    )
    return dataset_root


def test_validate_metadata_dataset_accepts_valid_jsonl_dataset(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, [_image_row("a"), _image_row("b", selected=False)])

    summary = validate_metadata_dataset(dataset_root)

    assert summary.ok
    assert summary.dataset_id == "sam3-100k-v1"
    assert summary.row_count == 2
    assert summary.selected_count == 1
    assert summary.shard_count == 1


def test_validate_metadata_dataset_reports_contract_errors(tmp_path: Path) -> None:
    bad_row = _image_row("a")
    del bad_row["swisstopo"]
    dataset_root = _write_dataset(tmp_path, [bad_row], selected_count=999)
    (dataset_root / "leaked-image.jpg").write_text("not really an image", encoding="utf8")

    summary = validate_metadata_dataset(dataset_root)

    messages = [issue.message for issue in summary.errors]
    assert not summary.ok
    assert "Expected an object." in messages
    assert "Image file is inside metadata dataset root." in messages
    assert "Index declares 999 selected rows but shards contain 1." in messages


def test_resolve_label_votes_prefers_human_over_model_priority() -> None:
    resolved = resolve_label_votes(
        [
            {
                "vote_id": "model",
                "source": {"source_id": "sam3.1", "kind": "model", "priority": 100, "display_name": "SAM3.1"},
                "decision": "crosswalk",
                "confidence": 0.99,
                "created_at": "2026-05-15T00:00:00Z",
            },
            {
                "vote_id": "human",
                "source": {"source_id": "human:oli", "kind": "human", "priority": 1000, "display_name": "Oli"},
                "decision": "no_crosswalk",
                "created_at": "2026-05-15T01:00:00Z",
            },
        ]
    )

    assert resolved == {
        "decision": "no_crosswalk",
        "source_id": "human:oli",
        "source_kind": "human",
        "resolved_by": "human_override",
        "updated_at": "2026-05-15T01:00:00Z",
    }


def test_prepare_build_and_merge_sam3_metadata_dataset(tmp_path: Path) -> None:
    config_path = tmp_path / "sam3.toml"
    config_path.write_text(
        """
dataset_id = "sam3-test"
display_name = "SAM3 Test"
run_name = "sam3-test"
export_name = "metadata-test"
target_count = 3
tile_size_m = 25
scene_size_m = 50
image_px = 100
shard_target_count = 4

[source]
access = "wmts"
crs = "EPSG:2056"
resolution_m = 0.1

[[scene_groups]]
city = "Zurich"
split = "train"
latitude = 47.3769
longitude = 8.5417
grid_rows = 1
grid_cols = 1
""".strip(),
        encoding="utf8",
    )
    dataset_root = tmp_path / "datasets" / "sam3-test"

    prepare_summary = prepare_sam3_metadata_dataset(config_path, dataset_root)

    assert prepare_summary["tile_count"] == 3
    assert validate_metadata_dataset(dataset_root).ok

    jobs_summary = build_sam3_shard_jobs(dataset_root, tmp_path / "jobs", limit_shards=1, limit_tiles=2)
    assert jobs_summary["job_count"] == 1
    job_payload = json.loads((tmp_path / "jobs" / "shard-0000.json").read_text(encoding="utf8"))
    assert len(job_payload["tiles"]) == 2

    result_root = tmp_path / "results"
    result_root.mkdir()
    (result_root / "shard-0000.result.json").write_text(
        json.dumps(
            {
                "scanner": {"backend": "sam31", "sam_model": "sam3.1"},
                "tiles": [
                    {"tile_id": job_payload["tiles"][0]["tile_id"], "label": "crosswalk", "score": 0.87},
                    {"tile_id": job_payload["tiles"][1]["tile_id"], "label": "no_crosswalk", "score": 0.22},
                ],
            }
        ),
        encoding="utf8",
    )

    merge_summary = merge_sam3_metadata_dataset(dataset_root, result_root, write=True)
    validation = validate_metadata_dataset(dataset_root)

    assert merge_summary["updated_rows"] == 2
    assert validation.ok
    assert validation.selected_count == 2


def test_merge_sam3_metadata_dataset_preserves_mask_artifact(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path, [_image_row("a")])
    mask_path = tmp_path / "results" / "masks" / "shard-0000" / "a.png"
    mask_path.parent.mkdir(parents=True)
    mask_path.write_bytes(b"mask")
    result_root = tmp_path / "results"
    (result_root / "shard-0000.result.json").write_text(
        json.dumps(
            {
                "scanner": {"backend": "sam31", "sam_model": "sam3.1"},
                "tiles": [
                    {
                        "tile_id": "a",
                        "label": "crosswalk",
                        "score": 0.87,
                        "mask_artifact": {
                            "kind": "sam3-pseudo-mask",
                            "format": "png",
                            "path": str(mask_path),
                            "width": 64,
                            "height": 64,
                        },
                    }
                ],
            }
        ),
        encoding="utf8",
    )

    merge_sam3_metadata_dataset(dataset_root, result_root, write=True)
    row = json.loads((dataset_root / "scenes" / "scene-a" / "perimeters" / "p0000" / "tiles.jsonl").read_text(encoding="utf8").splitlines()[0])

    assert row["labels"][-1]["metadata"]["mask_artifact"]["path"] == str(mask_path)
