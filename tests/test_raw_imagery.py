import json
from pathlib import Path

from crosswalk_detector.data.raw_imagery import download_dataset_scenes, estimate_scene_cache_size


def _write_dataset(root: Path) -> Path:
    dataset_root = root / "sam3-test"
    scene_root = dataset_root / "scenes" / "scene-a"
    scene_root.mkdir(parents=True)
    (scene_root / "scene.json").write_text(
        json.dumps(
            {
                "scene_id": "scene-a",
                "latitude": 47.3769,
                "longitude": 8.5417,
                "size_m": 800,
                "image_px": 2048,
                "tile_size_m": 25,
            }
        ),
        encoding="utf8",
    )
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "format": "crosswalk-jsonl-v1",
                "dataset_id": "sam3-test",
                "shards": [
                    {"shard_id": "scene-a-p0000", "path": "unused.jsonl", "tile_count": 1, "scene_id": "scene-a"},
                    {"shard_id": "scene-a-p0001", "path": "unused-2.jsonl", "tile_count": 1, "scene_id": "scene-a"},
                ],
            }
        ),
        encoding="utf8",
    )
    return dataset_root


def _write_static_dataset(root: Path) -> Path:
    dataset_root = root / "sam3-static"
    dataset_root.mkdir(parents=True)
    (dataset_root / "dataset.json").write_text(
        json.dumps(
            {
                "format": "crosswalk-static-jsonl-v1",
                "dataset_id": "sam3-static",
                "scenes": [
                    {
                        "scene_id": "static-scene-a",
                        "city": "Zurich",
                        "split": "train",
                        "tile_count": 1024,
                        "bbox_mercator": [946322.877, 5999278.373, 947122.877, 6000078.373],
                    }
                ],
                "shards": [
                    {"shard_id": "static-scene-a-p0000", "path": "shards/a.jsonl.gz", "tile_count": 1024, "scene_id": "static-scene-a"}
                ],
            }
        ),
        encoding="utf8",
    )
    return dataset_root


def test_download_dataset_scenes_caches_scene_files(tmp_path: Path) -> None:
    dataset_root = _write_dataset(tmp_path)
    calls = []

    def fetcher(url: str) -> bytes:
        calls.append(url)
        return b"fake-jpeg"

    first = download_dataset_scenes(dataset_root, raw_root=tmp_path / "raw", fetcher=fetcher)
    second = download_dataset_scenes(dataset_root, raw_root=tmp_path / "raw", fetcher=fetcher)

    assert first["scene_count"] == 1
    assert first["downloaded"] == 1
    assert first["cached"] == 0
    assert first["bytes"] == len(b"fake-jpeg")
    assert second["downloaded"] == 0
    assert second["cached"] == 1
    assert len(calls) == 1
    assert (tmp_path / "raw" / "wms-mosaics" / "scene-a.jpg").read_bytes() == b"fake-jpeg"


def test_download_dataset_scenes_accepts_static_release_index(tmp_path: Path) -> None:
    dataset_root = _write_static_dataset(tmp_path)
    urls = []

    def fetcher(url: str) -> bytes:
        urls.append(url)
        return b"static-fake-jpeg"

    summary = download_dataset_scenes(dataset_root, raw_root=tmp_path / "raw", fetcher=fetcher)

    assert summary["scene_count"] == 1
    assert summary["downloaded"] == 1
    assert "BBOX=946322.877%2C5999278.373%2C947122.877%2C6000078.373" in urls[0]
    assert "WIDTH=2048" in urls[0]


def test_estimate_scene_cache_size() -> None:
    estimate = estimate_scene_cache_size([1_000, 2_000, 3_000], 10)

    assert estimate["sample_count"] == 3
    assert estimate["scene_count"] == 10
    assert estimate["average_bytes"] == 2_000
    assert estimate["estimated_bytes"] == 20_000
