"""Command line entry points for the local pilot pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pilot import (
    bootstrap_local_pilot,
    build_pilot_summary,
    build_full_capped_dataset,
    derive_split_targets,
    export_balanced_seed_set,
    fetch_wmts_search_tiles,
    fetch_wmts_debug_tiles,
    load_pilot_config,
    write_default_config,
)
from .real_pipeline import build_real_dataset, write_compact_manifest
from .scan_batch import load_scan_batch_job, run_scan_batch_job, summarize_scan_batch_result, write_scan_batch_result
from .metadata_dataset import validate_metadata_dataset
from .raw_imagery import download_dataset_scenes
from .sam3_metadata import (
    build_sam3_shard_jobs,
    export_training_dataset,
    merge_sam3_metadata_dataset,
    prepare_sam3_metadata_dataset,
)
from .sam3_results import summarize_sam3_results
from .train_mobilenet import train_mobilenet, train_scratch_cnn, train_scratch_mobilenet
from .train_crossmask import prepare_crossmask_export, train_crossmask
from .crossmask_inference import run_crossmask_request


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crosswalk-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_config = subparsers.add_parser("write-default-config")
    write_config.add_argument(
        "--output",
        type=Path,
        default=Path("configs/local-debug.json"),
        help="Where to write the default local pilot config.",
    )

    plan = subparsers.add_parser("plan-local-pilot")
    plan.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local-debug.json"),
        help="Pilot config to summarize.",
    )

    bootstrap = subparsers.add_parser("bootstrap-local-pilot")
    bootstrap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local-debug.json"),
        help="Pilot config to materialize into local folders and manifests.",
    )

    fetch = subparsers.add_parser("fetch-wmts-debug-tiles")
    fetch.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local-debug.json"),
        help="Pilot config that defines which city-center WMTS tiles to fetch.",
    )

    fetch_search = subparsers.add_parser("fetch-wmts-search-tiles")
    fetch_search.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local-debug.json"),
        help="Pilot config that defines which broader WMTS tiles to fetch.",
    )
    fetch_search.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Override the configured search radius.",
    )

    export = subparsers.add_parser("export-balanced-seed-set")
    export.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/processed/local-debug-v1/reviews/manual-review-v1.csv"),
        help="Reviewed label CSV to rebalance.",
    )
    export.add_argument(
        "--name",
        default="balanced-seed-v1",
        help="Name of the export directory.",
    )
    export.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Maximum number of images to keep per class.",
    )

    build_full = subparsers.add_parser("build-full-capped-dataset")
    build_full.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local-debug.json"),
        help="Pilot config that defines targets and search geometry.",
    )
    build_full.add_argument(
        "--review-csv",
        type=Path,
        default=Path("data/processed/local-debug-v1/reviews/manual-review-v1.csv"),
        help="Optional reviewed label CSV used as hard overrides.",
    )
    build_full.add_argument(
        "--name",
        default="full-capped-v1",
        help="Name of the export directory.",
    )
    build_full.add_argument(
        "--radius",
        type=int,
        default=None,
        help="Override the configured full-run search radius.",
    )

    real_build = subparsers.add_parser("build-real-dataset")
    real_build.add_argument(
        "--run-name",
        default=None,
        help="Name of the raw/processed run folder.",
    )
    real_build.add_argument(
        "--name",
        default=None,
        help="Name of the export directory.",
    )
    real_build.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Target number of images per class.",
    )
    real_build.add_argument(
        "--config",
        type=Path,
        default=Path("configs/real-dataset.toml"),
        help="TOML config for the real dataset pipeline.",
    )

    train_mobile = subparsers.add_parser("train-mobilenet")
    train_mobile.add_argument("--run-name", default="real-v1")
    train_mobile.add_argument("--name", default="real-balanced-256")
    train_mobile.add_argument("--epochs", type=int, default=6)

    train_scratch = subparsers.add_parser("train-scratch-cnn")
    train_scratch.add_argument("--run-name", default="osm-v1-10k")
    train_scratch.add_argument("--name", default="balanced-10k-v1")
    train_scratch.add_argument("--epochs", type=int, default=40)
    train_scratch.add_argument("--batch-size", type=int, default=64)
    train_scratch.add_argument("--learning-rate", type=float, default=1e-3)

    train_scratch_mobile = subparsers.add_parser("train-scratch-mobilenet")
    train_scratch_mobile.add_argument("--run-name", default="osm-v2-50k")
    train_scratch_mobile.add_argument("--name", default="balanced-50k-v1")
    train_scratch_mobile.add_argument("--epochs", type=int, default=60)
    train_scratch_mobile.add_argument("--batch-size", type=int, default=128)
    train_scratch_mobile.add_argument("--learning-rate", type=float, default=8e-4)
    train_scratch_mobile.add_argument("--image-size", type=int, default=160)

    prepare_crossmask = subparsers.add_parser("prepare-crossmask-export")
    prepare_crossmask.add_argument("--dataset", type=Path, required=True, help="Merged metadata dataset root.")
    prepare_crossmask.add_argument("--output", type=Path, required=True, help="Output directory for tile images, masks, and manifest.")
    prepare_crossmask.add_argument("--positive-limit", type=int, default=2500)
    prepare_crossmask.add_argument("--negative-ratio", type=float, default=1.0)
    prepare_crossmask.add_argument("--min-confidence", type=float, default=0.4)
    prepare_crossmask.add_argument("--min-mask-coverage", type=float, default=0.01)
    prepare_crossmask.add_argument("--image-size", type=int, default=128)
    prepare_crossmask.add_argument("--seed", type=int, default=7)
    prepare_crossmask.add_argument("--overwrite", action="store_true")

    train_crossmask_parser = subparsers.add_parser("train-crossmask")
    train_crossmask_parser.add_argument("--export", type=Path, required=True, help="CrossMask export directory containing manifest.csv.")
    train_crossmask_parser.add_argument("--model-output", type=Path, required=True, help="Directory for model and metrics.")
    train_crossmask_parser.add_argument("--epochs", type=int, default=8)
    train_crossmask_parser.add_argument("--batch-size", type=int, default=64)
    train_crossmask_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_crossmask_parser.add_argument("--image-size", type=int, default=128)
    train_crossmask_parser.add_argument("--base-channels", type=int, default=24)
    train_crossmask_parser.add_argument("--input-channels", type=int, default=3)
    train_crossmask_parser.add_argument("--road-channel", action="store_true")
    train_crossmask_parser.add_argument("--num-workers", type=int, default=2)
    train_crossmask_parser.add_argument("--seed", type=int, default=7)

    compact_manifest = subparsers.add_parser("write-compact-manifest")
    compact_manifest.add_argument("--run-name", default="real-v1")
    compact_manifest.add_argument("--name", default="real-balanced-256")
    compact_manifest.add_argument(
        "--config",
        type=Path,
        default=Path("configs/real-dataset.toml"),
        help="TOML config used to recover source scene metadata.",
    )

    batch_scan = subparsers.add_parser("run-scan-job")
    batch_scan.add_argument("--job-file", type=Path, required=True, help="Exported scan job JSON.")
    batch_scan.add_argument("--output", type=Path, required=True, help="Where to write the result JSON.")
    batch_scan.add_argument("--mask-output-dir", type=Path, default=None, help="Optional directory for SAM3 pseudo-mask PNG artifacts.")
    batch_scan.add_argument("--progress", action="store_true", help="Print one line per processed tile.")

    crossmask_tiles = subparsers.add_parser("run-crossmask-tiles")
    crossmask_tiles.add_argument("--request", type=Path, required=True, help="JSON request containing model, dataset, and tile geometry.")
    crossmask_tiles.add_argument("--output", type=Path, required=True, help="Where to write CrossMask predictions.")

    validate_metadata = subparsers.add_parser("validate-metadata-dataset")
    validate_metadata.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root.")
    validate_metadata.add_argument("--max-errors", type=int, default=200, help="Maximum validation errors to print.")

    prepare_metadata = subparsers.add_parser("prepare-sam3-metadata-dataset")
    prepare_metadata.add_argument("--config", type=Path, required=True, help="SAM3 metadata dataset TOML config.")
    prepare_metadata.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root to create.")
    prepare_metadata.add_argument("--overwrite", action="store_true", help="Replace an existing metadata dataset root.")

    shard_jobs = subparsers.add_parser("build-sam3-shard-jobs")
    shard_jobs.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root.")
    shard_jobs.add_argument("--output", type=Path, required=True, help="Directory where job JSON files are written.")
    shard_jobs.add_argument("--limit-shards", type=int, default=None, help="Only write the first N shard jobs.")
    shard_jobs.add_argument("--limit-tiles", type=int, default=None, help="Only include the first N tiles per shard.")

    merge_metadata = subparsers.add_parser("merge-sam3-metadata-dataset")
    merge_metadata.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root.")
    merge_metadata.add_argument("--results", type=Path, required=True, help="Directory containing scan result JSON files.")
    merge_metadata.add_argument("--write", action="store_true", help="Write merged votes back into JSONL shards.")

    export_training = subparsers.add_parser("export-training-dataset")
    export_training.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root.")
    export_training.add_argument("--output", type=Path, required=True, help="Training export output directory.")
    export_training.add_argument("--limit", type=int, default=None, help="Export only the first N selected rows.")

    raw_scenes = subparsers.add_parser("download-raw-scenes")
    raw_scenes.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root.")
    raw_scenes.add_argument("--raw-root", type=Path, default=None, help="Where to cache raw scene imagery.")
    raw_scenes.add_argument("--limit-scenes", type=int, default=None, help="Only download the first N scenes.")
    raw_scenes.add_argument("--workers", type=int, default=4, help="Concurrent scene downloads.")

    summarize_results = subparsers.add_parser("summarize-sam3-results")
    summarize_results.add_argument("--results", type=Path, required=True, help="Directory containing SAM3 scan result JSON files.")
    summarize_results.add_argument("--expected-shards", type=int, default=None, help="Expected result shard count.")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "write-default-config":
        path = write_default_config(args.output)
        print(f"Wrote local pilot config to {path}")
        return 0

    if args.command == "plan-local-pilot":
        summary = build_pilot_summary(load_pilot_config(args.config))
        totals = summary["totals"]
        print(f"Run: {summary['run_name']}")
        print(
            "Totals: "
            f"{totals['area_km2']} km², "
            f"{totals['raw_gb']} GB raw imagery, "
            f"{totals['non_overlapping_tiles']} non-overlapping tiles, "
            f"{totals['windows']} windows"
        )
        print(f"Within budget: {totals['within_budget']}")
        return 0

    if args.command == "bootstrap-local-pilot":
        outputs = bootstrap_local_pilot(args.config)
        print(f"Prepared local pilot run at {outputs['processed_root']}")
        print(f"Plan report: {outputs['report_md']}")
        print(f"Area manifest: {outputs['areas_csv']}")
        return 0

    if args.command == "fetch-wmts-debug-tiles":
        summary = fetch_wmts_debug_tiles(args.config)
        print(
            f"Downloaded {summary['downloaded_tiles']} tiles "
            f"({summary['total_size_mb']} MB) into {summary['raw_root']}"
        )
        print(f"Manifest: {summary['manifest_csv']}")
        return 0

    if args.command == "fetch-wmts-search-tiles":
        summary = fetch_wmts_search_tiles(args.config, args.radius)
        print(
            f"Downloaded {summary['downloaded_tiles']} search tiles "
            f"({summary['total_size_mb']} MB) into {summary['raw_root']}"
        )
        print(f"Manifest: {summary['manifest_csv']}")
        return 0

    if args.command == "export-balanced-seed-set":
        split_targets = None
        if args.target_per_class is not None:
            config = load_pilot_config(Path("configs/local-debug.json"))
            split_targets = derive_split_targets(args.target_per_class, config.split_ratios)
        summary = export_balanced_seed_set(
            args.review_csv,
            args.name,
            args.target_per_class,
            split_targets,
        )
        print(
            f"Exported {summary['total_count']} balanced items "
            f"({summary['positive_count']} positive, {summary['negative_count']} negative)"
        )
        if summary["target_per_class"] is not None:
            print(
                f"Target per class: {summary['target_per_class']} "
                f"(full: {summary['is_target_full']})"
            )
            print(f"Split targets: {summary['split_targets']}")
            print(f"Actual split fill: {summary['actual_split_caps']}")
        print(f"Manifest: {summary['manifest_csv']}")
        return 0

    if args.command == "build-full-capped-dataset":
        summary = build_full_capped_dataset(
            args.config,
            args.review_csv,
            args.name,
            args.radius,
        )
        print(
            f"Built dataset with {summary['total_count']} items "
            f"({summary['positive_count']} positive, {summary['negative_count']} negative)"
        )
        print(f"Split targets: {summary['split_targets']}")
        print(f"Actual split fill: {summary['actual_split_caps']}")
        print(f"Target full: {summary['is_target_full']}")
        print(f"Manifest: {summary['manifest_csv']}")
        return 0

    if args.command == "build-real-dataset":
        summary = build_real_dataset(
            run_name=args.run_name,
            export_name=args.name,
            target_per_class=args.target_per_class,
            config_path=args.config,
        )
        print(
            f"Built real dataset with {summary['selected_total']} items "
            f"({summary['positive_count']} positive, {summary['negative_count']} negative)"
        )
        print(f"Split targets: {summary['split_targets']}")
        print(f"Per split counts: {summary['per_split_counts']}")
        print(f"Labels CSV: {summary['labels_csv']}")
        print(f"Tiles JSON: {summary['tiles_json']}")
        return 0

    if args.command == "write-compact-manifest":
        import csv

        from .real_config import load_real_pipeline_config

        config = load_real_pipeline_config(args.config)
        export_root = Path("data") / "processed" / args.run_name / "exports" / args.name
        labels_csv = export_root / "labels.csv"
        tiles_json = export_root / "tiles.json"
        tile_lookup = {}
        if tiles_json.exists():
            tile_lookup = {
                str(tile["tile_id"]): tile["bbox_mercator"]
                for tile in json.loads(tiles_json.read_text())["tiles"]
            }
        rows: list[dict[str, object]] = []
        with labels_csv.open() as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        **row,
                        "row": int(row["row"]),
                        "col": int(row["col"]),
                        "bbox_mercator": tile_lookup.get(str(row["tile_id"]), [0.0, 0.0, 0.0, 0.0]),
                    }
                )
        outputs = write_compact_manifest(export_root, args.run_name, args.name, config.scenes, rows)
        print(f"Compact JSON: {outputs['compact_json']}")
        print(f"Compact CSV: {outputs['compact_csv']}")
        return 0

    if args.command == "train-mobilenet":
        metrics = train_mobilenet(
            run_name=args.run_name,
            export_name=args.name,
            epochs=args.epochs,
        )
        print(
            f"Trained MobileNetV3-Small with test accuracy {metrics['test_accuracy']} "
            f"(best val {metrics['best_val_accuracy']})"
        )
        print(f"Model: {metrics['model_path']}")
        return 0

    if args.command == "train-scratch-cnn":
        metrics = train_scratch_cnn(
            run_name=args.run_name,
            export_name=args.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
        print(
            f"Trained scratch CNN with test accuracy {metrics['test_accuracy']} "
            f"(best val {metrics['best_val_accuracy']})"
        )
        print(f"Model: {metrics['model_path']}")
        return 0

    if args.command == "train-scratch-mobilenet":
        metrics = train_scratch_mobilenet(
            run_name=args.run_name,
            export_name=args.name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
        )
        print(
            f"Trained scratch MobileNetV3-Small with test accuracy {metrics['test_accuracy']} "
            f"(best val {metrics['best_val_accuracy']})"
        )
        print(f"Model: {metrics['model_path']}")
        return 0

    if args.command == "prepare-crossmask-export":
        summary = prepare_crossmask_export(
            args.dataset,
            args.output,
            positive_limit=args.positive_limit,
            negative_ratio=args.negative_ratio,
            min_confidence=args.min_confidence,
            min_mask_coverage=args.min_mask_coverage,
            image_size=args.image_size,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "train-crossmask":
        metrics = train_crossmask(
            args.export,
            args.model_output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            image_size=args.image_size,
            base_channels=args.base_channels,
            input_channels=args.input_channels,
            road_channel=args.road_channel,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        print(
            f"Trained CrossMaskNet with positive Dice {metrics['test']['positive_dice']} "
            f"and image recall {metrics['test']['image_recall']}"
        )
        print(f"Model: {metrics['model_path']}")
        return 0

    if args.command == "run-scan-job":
        job_payload = load_scan_batch_job(args.job_file)
        result = run_scan_batch_job(job_payload, progress=args.progress, mask_output_dir=args.mask_output_dir)
        output_path = write_scan_batch_result(args.output, result)
        print(summarize_scan_batch_result(result))
        print(f"Results: {output_path}")
        return 0

    if args.command == "run-crossmask-tiles":
        result = run_crossmask_request(args.request, args.output)
        print(json.dumps(result["summary"], indent=2))
        return 0

    if args.command == "validate-metadata-dataset":
        summary = validate_metadata_dataset(args.dataset, max_errors=args.max_errors)
        print(
            f"Dataset {summary.dataset_id or args.dataset}: "
            f"{summary.row_count} rows, "
            f"{summary.selected_count} selected, "
            f"{summary.shard_count} shards"
        )
        if summary.ok:
            print("Validation: ok")
            return 0
        print(f"Validation: failed with {len(summary.errors)} error(s)")
        for issue in summary.errors:
            print(f"- {issue.path}: {issue.message}")
        return 1

    if args.command == "prepare-sam3-metadata-dataset":
        summary = prepare_sam3_metadata_dataset(args.config, args.dataset, overwrite=args.overwrite)
        print(
            f"Prepared {summary['tile_count']} metadata rows "
            f"across {summary['shard_count']} shard(s) at {summary['dataset_root']}"
        )
        return 0

    if args.command == "build-sam3-shard-jobs":
        summary = build_sam3_shard_jobs(
            args.dataset,
            args.output,
            limit_shards=args.limit_shards,
            limit_tiles=args.limit_tiles,
        )
        print(
            f"Wrote {summary['job_count']} SAM3 job(s) with "
            f"{summary['tile_count']} tile(s) to {summary['output_root']}"
        )
        return 0

    if args.command == "merge-sam3-metadata-dataset":
        summary = merge_sam3_metadata_dataset(args.dataset, args.results, write=args.write)
        mode = "updated" if args.write else "would update"
        print(
            f"{mode.capitalize()} {summary['updated_rows']} row(s) in {summary['dataset_id']} "
            f"({summary['selected_count']} selected)"
        )
        return 0

    if args.command == "export-training-dataset":
        summary = export_training_dataset(args.dataset, args.output, limit=args.limit)
        print(f"Exported {summary['exported_count']} image(s) to {args.output}")
        print(f"Labels CSV: {summary['labels_csv']}")
        return 0

    if args.command == "download-raw-scenes":
        summary = download_dataset_scenes(
            args.dataset,
            raw_root=args.raw_root,
            limit_scenes=args.limit_scenes,
            workers=args.workers,
        )
        print(
            f"Raw scenes ready: {summary['scene_count']} scene(s), "
            f"{summary['downloaded']} downloaded, {summary['cached']} cached, "
            f"{summary['size_mb']} MB"
        )
        print(f"Raw root: {summary['raw_root']}")
        return 0

    if args.command == "summarize-sam3-results":
        summary = summarize_sam3_results(args.results, expected_shards=args.expected_shards)
        print(json.dumps(summary.to_dict(), indent=2))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
