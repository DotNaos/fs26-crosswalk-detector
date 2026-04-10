"""Command line entry points for the local pilot pipeline."""

from __future__ import annotations

import argparse
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
from .real_pipeline import build_real_dataset
from .scan_batch import load_scan_batch_job, run_scan_batch_job, summarize_scan_batch_result, write_scan_batch_result
from .train_mobilenet import train_mobilenet


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

    batch_scan = subparsers.add_parser("run-scan-job")
    batch_scan.add_argument("--job-file", type=Path, required=True, help="Exported scan job JSON.")
    batch_scan.add_argument("--output", type=Path, required=True, help="Where to write the result JSON.")
    batch_scan.add_argument("--progress", action="store_true", help="Print one line per processed tile.")

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

    if args.command == "run-scan-job":
        job_payload = load_scan_batch_job(args.job_file)
        result = run_scan_batch_job(job_payload, progress=args.progress)
        output_path = write_scan_batch_result(args.output, result)
        print(summarize_scan_batch_result(result))
        print(f"Results: {output_path}")
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
