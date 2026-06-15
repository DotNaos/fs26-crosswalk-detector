import json
import sys
from pathlib import Path

from crosswalk_detector.models import crossmask_inference
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

    monkeypatch.setattr(sys, "argv", ["test", "--model-root", str(model_root), "--metrics-only"])
    monkeypatch.setattr(workflow, "_ensure_project_assets", lambda **kwargs: calls.append(kwargs))

    assert workflow.test_main() == 0
    assert calls == []


def test_test_downloads_inputs_when_input_dir_is_missing(tmp_path: Path, monkeypatch) -> None:
    model_root = tmp_path / "fresh-model"
    model_root.mkdir()
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "predictions"
    calls = {"assets": [], "download": [], "classify": []}

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--model-root",
            str(model_root),
            "--download-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--count",
            "4",
            "--no-progress",
        ],
    )
    monkeypatch.setattr(workflow, "_ensure_project_assets", lambda **kwargs: calls["assets"].append(kwargs))
    monkeypatch.setattr(
        workflow,
        "_prepare_input_images",
        lambda _args, path, **kwargs: calls["download"].append((path, kwargs))
        or {"total": 4, "positive": 2, "negative": 2, "manifest": str(path / "input_images.csv")},
    )
    monkeypatch.setattr(
        crossmask_inference,
        "run_crossmask_image_directory",
        lambda in_dir, out_dir, model, **_kwargs: calls["classify"].append((in_dir, out_dir, model))
        or {
            "total": 4,
            "positive": 2,
            "negative": 2,
            "positive_dir": str(out_dir / "positive"),
            "negative_dir": str(out_dir / "negative"),
            "positive_overlays_dir": str(out_dir / "positive_overlays"),
        },
    )

    assert workflow.test_main() == 0
    assert calls["assets"] == [{"skip_model": True}]
    assert calls["download"][0][0] == input_dir
    assert calls["download"][0][1]["positive_count"] == 2
    assert calls["download"][0][1]["negative_count"] == 2
    assert calls["download"][0][1]["overwrite"] is True
    assert calls["classify"] == [(input_dir, output_dir, model_root)]


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


def test_test_progress_is_enabled_by_default() -> None:
    args = workflow._test_parser().parse_args([])

    assert args.no_progress is False
    assert args.input_dir is None
    assert args.download_dir == workflow.DEFAULT_INPUT_IMAGES
    assert args.count == 20
    assert args.metrics_only is False


def test_test_progress_can_be_disabled() -> None:
    args = workflow._test_parser().parse_args(["--no-progress"])

    assert args.no_progress is True


def test_download_images_progress_can_be_disabled() -> None:
    args = workflow._download_images_parser().parse_args(["--no-progress"])

    assert args.no_progress is True


def test_test_can_print_metrics_only() -> None:
    args = workflow._test_parser().parse_args(["--metrics-only"])

    assert args.metrics_only is True
