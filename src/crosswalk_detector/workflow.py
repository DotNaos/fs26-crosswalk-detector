"""Simple workflow commands for training and testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .raw_imagery import download_dataset_scenes
from .train_crossmask import prepare_crossmask_export, train_crossmask

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path("web/public/static-datasets/sam3-500k-masks-v1")
DEFAULT_EXPORT = Path("data/processed/crossmask/sam3-500k-road-channel-v4")
DEFAULT_INPUT_IMAGES = Path("data/input/crossmask-images")
DEFAULT_MODEL = Path("models/crossmask/sam3-500k-road-channel-v4")
DEFAULT_TRAIN_EXPORT = Path("data/processed/crossmask/local-run")
DEFAULT_TRAIN_MODEL = Path("models/crossmask/local-run")


def dataset_main() -> int:
    args = _dataset_parser().parse_args()
    _require_default_profile(args.profile)
    dataset_root = _resolve(args.dataset)
    export_root = _resolve(args.export)
    _prepare_dataset(
        dataset_root,
        export_root,
        limit_scenes=args.limit_scenes,
        workers=args.workers,
        positive_limit=args.positive_limit,
        negative_ratio=args.negative_ratio,
        min_confidence=args.min_confidence,
        min_mask_coverage=args.min_mask_coverage,
        image_size=args.image_size,
        seed=args.seed,
        rebuild_export=args.rebuild_export,
        skip_raw_cache=args.skip_raw_cache,
    )
    return 0


def download_images_main() -> int:
    args = _download_images_parser().parse_args()
    _require_default_profile(args.profile)
    _stage("Checking project assets...", enabled=not args.no_progress)
    _ensure_project_assets(skip_model=True)

    output_dir = _resolve(args.output_dir)
    positive_count, negative_count = _image_counts(args.count, args.positive_ratio, args.positive_count, args.negative_count)
    summary = _prepare_input_images(
        args,
        output_dir,
        positive_count=positive_count,
        negative_count=negative_count,
        overwrite=args.overwrite,
    )
    _print_input_summary(summary, output_dir)
    return 0


def train_main() -> int:
    args = _train_parser().parse_args()
    _require_default_profile(args.profile)
    dataset_root = _resolve(args.dataset)
    export_root = _resolve(args.export)
    model_root = _resolve(args.model_output)

    _prepare_dataset(
        dataset_root,
        export_root,
        limit_scenes=args.limit_scenes,
        workers=args.workers,
        positive_limit=args.positive_limit,
        negative_ratio=args.negative_ratio,
        min_confidence=args.min_confidence,
        min_mask_coverage=args.min_mask_coverage,
        image_size=args.image_size,
        seed=args.seed,
        rebuild_export=args.rebuild_export,
        skip_raw_cache=args.skip_raw_cache,
        show_progress=not args.no_progress,
    )

    metrics = train_crossmask(
        export_root,
        model_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        base_channels=args.base_channels,
        input_channels=4 if args.road_channel else 3,
        road_channel=args.road_channel,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_seconds=args.max_train_seconds,
        show_progress=not args.no_progress,
    )
    print(
        f"Training complete: accuracy={metrics['test']['image_accuracy']:.6f}, "
        f"recall={metrics['test']['image_recall']:.6f}, "
        f"positive_dice={metrics['test']['positive_dice']:.6f}"
    )
    print(f"Model: {metrics['model_path']}")
    return 0


def test_main() -> int:
    args = _test_parser().parse_args()
    _require_default_profile(args.profile)
    model_root = _resolve(args.model_root)
    needs_release_model = _uses_default_model_root(model_root) and not _model_files_exist(model_root)
    should_download_inputs = args.input_dir is None and not args.metrics_only
    if needs_release_model or should_download_inputs:
        _stage("Checking project assets...", enabled=not args.no_progress)
        _ensure_project_assets(skip_model=not needs_release_model)
    if not args.metrics_only:
        from .crossmask_inference import run_crossmask_image_directory

        if args.input_dir:
            input_dir = _resolve(args.input_dir)
        else:
            input_dir = _resolve(args.download_dir)
            positive_count, negative_count = _image_counts(args.count, args.positive_ratio, args.positive_count, args.negative_count)
            summary = _prepare_input_images(
                args,
                input_dir,
                positive_count=positive_count,
                negative_count=negative_count,
                overwrite=True,
            )
            _print_input_summary(summary, input_dir)

        output_dir = _resolve(args.output_dir)
        _stage(f"Testing image directory: {input_dir}", enabled=not args.no_progress)
        summary = run_crossmask_image_directory(
            input_dir,
            output_dir,
            model_root,
            threshold=args.positive_threshold,
            include_overlays=not args.no_overlays,
            show_progress=not args.no_progress,
        )
        print("CrossMaskNet directory test complete")
        print(f"Input images: {summary['total']}")
        print(f"Positive: {summary['positive']}")
        print(f"Negative: {summary['negative']}")
        print(f"Positive images: {summary['positive_dir']}")
        print(f"Negative images: {summary['negative_dir']}")
        if not args.no_overlays:
            print(f"Positive overlays: {summary['positive_overlays_dir']}")
        print(f"Summary: {output_dir / 'summary.json'}")
        return 0

    metrics_path = model_root / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing model metrics: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf8"))
    test_metrics: dict[str, Any] = metrics.get("test", {})
    print("CrossMaskNet test metrics")
    print(f"Accuracy: {test_metrics.get('image_accuracy')}")
    print(f"Precision: {test_metrics.get('image_precision')}")
    print(f"Recall: {test_metrics.get('image_recall')}")
    print(f"Positive Dice: {test_metrics.get('positive_dice')}")
    print(f"Positive IoU: {test_metrics.get('positive_iou')}")
    print(f"Model: {metrics.get('model_path', model_root / 'crossmasknet_best.pt')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m crosswalk_detector.workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("dataset", parents=[_dataset_parser(add_help=False)], add_help=True)
    subparsers.add_parser("download-images", parents=[_download_images_parser(add_help=False)], add_help=True)
    subparsers.add_parser("train", parents=[_train_parser(add_help=False)], add_help=True)
    subparsers.add_parser("test", parents=[_test_parser(add_help=False)], add_help=True)
    args = parser.parse_args()
    if args.command == "dataset":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return dataset_main()
    if args.command == "download-images":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return download_images_main()
    if args.command == "train":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return train_main()
    if args.command == "test":
        sys.argv = [sys.argv[0], *sys.argv[2:]]
        return test_main()
    parser.error("Unknown command")
    return 2


def _dataset_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset", add_help=add_help)
    parser.add_argument("--profile", default="default")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--positive-limit", type=int, default=2500)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--min-mask-coverage", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--skip-raw-cache", action="store_true")
    parser.add_argument("--rebuild-export", action="store_true")
    return parser


def _train_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = _dataset_parser(add_help=add_help)
    parser.prog = "train"
    parser.set_defaults(export=DEFAULT_TRAIN_EXPORT, positive_limit=30, skip_raw_cache=True)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_TRAIN_MODEL)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-seconds", type=int, default=120)
    parser.add_argument("--road-channel", action="store_true")
    parser.add_argument("--prefetch-raw-cache", action="store_false", dest="skip_raw_cache")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def _download_images_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="download-images", add_help=add_help)
    parser.add_argument("--profile", default="default", help="Prepared project configuration.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Metadata dataset location.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_IMAGES, help="Where downloaded input images are written.")
    parser.add_argument("--count", type=int, default=20, help="Total number of input images to download.")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Share of downloaded images that should come from positive source examples.")
    parser.add_argument("--positive-count", type=int, default=None, help="Explicit positive image count. Overrides --count/--positive-ratio when used with --negative-count.")
    parser.add_argument("--negative-count", type=int, default=None, help="Explicit negative image count. Overrides --count/--positive-ratio when used with --positive-count.")
    parser.add_argument("--image-size", type=int, default=128, help="Output image size in pixels.")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Minimum source-label confidence for positive examples.")
    parser.add_argument("--min-mask-coverage", type=float, default=0.01, help="Minimum source mask coverage for positive examples.")
    parser.add_argument("--seed", type=int, default=7, help="Repeatable sampling seed.")
    parser.add_argument("--overwrite", action="store_true", help="Remove existing image files in the output folder first.")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress output.")
    return parser


def _test_parser(add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test", add_help=add_help)
    parser.add_argument("--profile", default="default", help="Prepared project configuration.")
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL, help="Model checkpoint and metrics directory.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Folder of new images to classify.")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_INPUT_IMAGES, help="Where test downloads input images when --input-dir is omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/predictions/crossmask-test"), help="Where classified outputs are written.")
    parser.add_argument("--count", type=int, default=20, help="Total number of input images to download when --input-dir is omitted.")
    parser.add_argument("--positive-ratio", type=float, default=0.5, help="Share of downloaded images that should come from positive source examples.")
    parser.add_argument("--positive-count", type=int, default=None, help="Explicit positive image count for automatic test input download.")
    parser.add_argument("--negative-count", type=int, default=None, help="Explicit negative image count for automatic test input download.")
    parser.add_argument("--image-size", type=int, default=128, help="Downloaded input image size in pixels.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Metadata dataset location for automatic input download.")
    parser.add_argument("--min-confidence", type=float, default=0.4, help="Minimum source-label confidence for downloaded positive examples.")
    parser.add_argument("--min-mask-coverage", type=float, default=0.01, help="Minimum source mask coverage for downloaded positive examples.")
    parser.add_argument("--seed", type=int, default=7, help="Repeatable sampling seed for automatic input download.")
    parser.add_argument(
        "--positive-threshold",
        "--threshold",
        dest="positive_threshold",
        type=float,
        default=0.005,
        help="Minimum mask coverage for positive. Images below this value are negative.",
    )
    parser.add_argument("--no-overlays", action="store_true", help="Do not write positive overlay images.")
    parser.add_argument("--metrics-only", action="store_true", help="Only print stored model metrics. Do not download or classify input images.")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress output.")
    return parser


def _ensure_project_assets(*, skip_model: bool) -> None:
    script = REPO_ROOT / "scripts" / "download_submission_assets.py"
    command = [sys.executable, str(script)]
    if skip_model:
        command.append("--skip-model")
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def _uses_default_model_root(model_root: Path) -> bool:
    return model_root == _resolve(DEFAULT_MODEL)


def _model_files_exist(model_root: Path) -> bool:
    return (model_root / "metrics.json").exists() and (model_root / "crossmasknet_best.pt").exists()


def _prepare_dataset(
    dataset_root: Path,
    export_root: Path,
    *,
    limit_scenes: int | None,
    workers: int,
    positive_limit: int,
    negative_ratio: float,
    min_confidence: float,
    min_mask_coverage: float,
    image_size: int,
    seed: int,
    rebuild_export: bool,
    skip_raw_cache: bool,
    show_progress: bool = True,
) -> None:
    _stage("Checking project assets...", enabled=show_progress)
    _ensure_project_assets(skip_model=True)
    if not skip_raw_cache:
        _stage("Preparing raw image cache...", enabled=show_progress)
        _prepare_raw_cache(dataset_root, limit_scenes, workers)
    if rebuild_export or not (export_root / "manifest.csv").exists():
        _stage(f"Preparing training export: {export_root}", enabled=show_progress)
        summary = prepare_crossmask_export(
            dataset_root,
            export_root,
            positive_limit=positive_limit,
            negative_ratio=negative_ratio,
            min_confidence=min_confidence,
            min_mask_coverage=min_mask_coverage,
            image_size=image_size,
            seed=seed,
            overwrite=rebuild_export,
            show_progress=show_progress,
        )
        print(json.dumps(summary, indent=2))
    else:
        print(f"Dataset export already exists: {export_root}")


def _prepare_raw_cache(dataset_root: Path, limit_scenes: int | None, workers: int) -> None:
    summary = download_dataset_scenes(dataset_root, limit_scenes=limit_scenes, workers=workers)
    print(
        f"Raw scenes ready: {summary['scene_count']} scene(s), "
        f"{summary['downloaded']} downloaded, {summary['cached']} cached, "
        f"{summary['size_mb']} MB"
    )


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _require_default_profile(profile: str) -> None:
    if profile != "default":
        raise ValueError("Only --profile default is defined.")


def _stage(message: str, *, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)


def _prepare_input_images(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    positive_count: int,
    negative_count: int,
    overwrite: bool,
) -> dict[str, Any]:
    from .input_images import download_input_images

    _stage(f"Preparing {positive_count + negative_count} input image(s): {output_dir}", enabled=not args.no_progress)
    return download_input_images(
        _resolve(args.dataset),
        output_dir,
        positive_count=positive_count,
        negative_count=negative_count,
        image_size=args.image_size,
        seed=args.seed,
        min_confidence=args.min_confidence,
        min_mask_coverage=args.min_mask_coverage,
        overwrite=overwrite,
        show_progress=not args.no_progress,
    )


def _print_input_summary(summary: dict[str, Any], output_dir: Path) -> None:
    print("Input images ready")
    print(f"Images: {summary['total']}")
    print(f"Positive source examples: {summary['positive']}")
    print(f"Negative source examples: {summary['negative']}")
    print(f"Input directory: {output_dir}")
    print(f"Manifest: {summary['manifest']}")


def _image_counts(count: int, positive_ratio: float, positive_count: int | None, negative_count: int | None) -> tuple[int, int]:
    if positive_count is not None or negative_count is not None:
        if positive_count is None or negative_count is None:
            raise ValueError("Use both --positive-count and --negative-count, or use --count with --positive-ratio.")
        if positive_count < 0 or negative_count < 0:
            raise ValueError("Image counts must be zero or greater.")
        return positive_count, negative_count
    if count < 0:
        raise ValueError("--count must be zero or greater.")
    if not 0.0 <= positive_ratio <= 1.0:
        raise ValueError("--positive-ratio must be between 0 and 1.")
    positive = round(count * positive_ratio)
    return positive, count - positive


if __name__ == "__main__":
    raise SystemExit(main())
