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


def test_train_defaults_are_small_and_local() -> None:
    args = workflow._train_parser().parse_args([])

    assert args.export == workflow.DEFAULT_TRAIN_EXPORT
    assert args.model_output == workflow.DEFAULT_TRAIN_MODEL
    assert args.positive_limit == 30
    assert args.epochs == 1
    assert args.batch_size == 4
    assert args.base_channels == 4
    assert args.num_workers == 0
    assert args.max_train_seconds == 120
    assert args.skip_raw_cache is True


def test_train_can_disable_time_limit_and_prefetch_raw_cache() -> None:
    args = workflow._train_parser().parse_args(["--max-train-seconds", "0", "--prefetch-raw-cache"])

    assert args.max_train_seconds == 0
    assert args.skip_raw_cache is False
