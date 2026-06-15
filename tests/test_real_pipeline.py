import gzip
import json
from pathlib import Path

from crosswalk_detector.data.real_config import SceneSpec
from crosswalk_detector.data.real_pipeline import write_compact_manifest


def test_write_compact_manifest_stores_coordinates_and_labels(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="demo",
        city="Demo",
        split="train",
        latitude=47.0,
        longitude=8.0,
        size_m=800,
        image_px=2048,
    )
    rows = [
        {
            "tile_id": "demo:r00:c00",
            "scene_id": "demo",
            "split": "train",
            "label": "crosswalk",
            "row": 0,
            "col": 0,
            "bbox_mercator": [1.1, 2.2, 3.3, 4.4],
        }
    ]

    outputs = write_compact_manifest(tmp_path, "run", "export", (scene,), rows)

    with gzip.open(outputs["compact_json"], "rt", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with gzip.open(outputs["compact_csv"], "rt", encoding="utf-8") as handle:
        csv_text = handle.read()

    assert manifest["format"] == "crosswalk-compact-v1"
    assert manifest["tiles"][0]["label"] == "crosswalk"
    assert manifest["tiles"][0]["bbox_mercator"] == [1.1, 2.2, 3.3, 4.4]
    assert "demo:r00:c00,demo,train,crosswalk,0,0,1.1,2.2,3.3,4.4" in csv_text
