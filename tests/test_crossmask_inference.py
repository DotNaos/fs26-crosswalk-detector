import json
from pathlib import Path

import numpy as np
from PIL import Image

from crosswalk_detector.models import crossmask_inference


def test_run_crossmask_image_directory_writes_classified_outputs(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    Image.new("RGB", (16, 16), (255, 255, 255)).save(input_dir / "crosswalk.jpg")
    Image.new("RGB", (16, 16), (0, 0, 0)).save(input_dir / "background.jpg")

    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "metrics.json").write_text(
        json.dumps({"image_size": 8, "base_channels": 4, "input_channels": 3, "road_channel": False}),
        encoding="utf8",
    )

    def fake_predict(_model, image, _image_size, _device, *, road_channel):
        assert not road_channel
        if image.getpixel((0, 0)) == (255, 255, 255):
            return np.ones((8, 8), dtype=np.float32)
        return np.zeros((8, 8), dtype=np.float32)

    monkeypatch.setattr(crossmask_inference, "_device", lambda: "cpu")
    monkeypatch.setattr(crossmask_inference, "_load_model", lambda _metrics, _model_root, _device: object())
    monkeypatch.setattr(crossmask_inference, "_predict_probability", fake_predict)

    output_dir = tmp_path / "output"
    summary = crossmask_inference.run_crossmask_image_directory(input_dir, output_dir, model_root)

    assert summary["total"] == 2
    assert summary["positive"] == 1
    assert summary["negative"] == 1
    assert (output_dir / "positive" / "crosswalk.jpg").exists()
    assert (output_dir / "negative" / "background.jpg").exists()
    assert (output_dir / "positive_overlays" / "crosswalk.png").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "predictions.csv").exists()
