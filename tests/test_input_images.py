from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from crosswalk_detector import input_images
from crosswalk_detector.workflow import _image_counts
from crosswalk_detector.train_crossmask import MaskCandidate


def test_download_input_images_writes_flat_input_folder(tmp_path: Path, monkeypatch) -> None:
    positive = _candidate("positive-tile", "crosswalk")
    negative = _candidate("negative-tile", "no_crosswalk")

    monkeypatch.setattr(input_images, "_load_candidates", lambda _root, **_kwargs: ([positive], [negative]))
    monkeypatch.setattr(input_images, "_scene_request", lambda _root, scene_id: SimpleNamespace(scene_id=scene_id))
    monkeypatch.setattr(input_images, "load_cached_scene_image", lambda _root, _scene: Image.new("RGB", (32, 32), (100, 100, 100)))
    monkeypatch.setattr(input_images, "crop_tile", lambda _image, _scene, _tile: Image.new("RGB", (32, 32), (200, 200, 200)))

    output_dir = tmp_path / "inputs"
    summary = input_images.download_input_images(tmp_path / "dataset", output_dir, positive_count=1, negative_count=1)

    assert summary["total"] == 2
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert len(list(output_dir.glob("*.jpg"))) == 2
    assert (output_dir / "input_images.csv").exists()
    assert (output_dir / "summary.json").exists()


def test_image_counts_from_total_and_ratio() -> None:
    assert _image_counts(20, 0.5, None, None) == (10, 10)
    assert _image_counts(5, 0.6, None, None) == (3, 2)


def test_image_counts_can_be_explicit() -> None:
    assert _image_counts(20, 0.5, 2, 7) == (2, 7)


def _candidate(tile_id: str, label: str) -> MaskCandidate:
    return MaskCandidate(
        image_id=tile_id,
        tile_id=tile_id,
        scene_id="scene-a",
        city="Zurich",
        row=0,
        col=0,
        bbox_mercator=(0.0, 0.0, 1.0, 1.0),
        relative_path=f"{tile_id}.jpg",
        label=label,
        confidence=0.9,
        mask_path="",
        mask_coverage=0.1 if label == "crosswalk" else 0.0,
    )
