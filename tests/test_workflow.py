import json
import sys
from pathlib import Path

from crosswalk_detector import workflow


def test_custom_model_root_does_not_download_release_assets(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "fresh-model"
    model_root.mkdir()
    (model_root / "metrics.json").write_text(
        json.dumps(
            {
                "test": {
                    "image_accuracy": 1.0,
                    "image_precision": 1.0,
                    "image_recall": 1.0,
                    "positive_dice": 1.0,
                    "positive_iou": 1.0,
                },
                "model_path": str(model_root / "crossmasknet_best.pt"),
            }
        ),
        encoding="utf8",
    )
    calls = []

    monkeypatch.setattr(sys, "argv", ["test", "--model-root", str(model_root)])
    monkeypatch.setattr(workflow, "_ensure_project_assets", lambda **kwargs: calls.append(kwargs))

    assert workflow.test_main() == 0
    assert calls == []
