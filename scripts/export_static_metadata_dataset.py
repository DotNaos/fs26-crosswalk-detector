#!/usr/bin/env python3
"""Export a metadata JSONL dataset as browser-friendly static shards."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Metadata dataset root containing dataset.json.")
    parser.add_argument("--output", type=Path, required=True, help="Static dataset output directory.")
    parser.add_argument("--dataset-url", default=None, help="Optional public URL prefix for this dataset.")
    parser.add_argument("--include-masks", action="store_true", help="Copy referenced SAM mask PNGs into the static dataset.")
    parser.add_argument("--crossmask-results", type=Path, default=None, help="Optional CrossMaskNet prediction result directory to merge as static label votes.")
    parser.add_argument("--project-root", type=Path, default=Path("."), help="Project root used to resolve relative mask artifact paths.")
    parser.add_argument("--public-base-path", default=None, help="Public URL prefix for copied masks. Defaults to /static-datasets/<dataset_id>.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset
    output_root = args.output
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "shards").mkdir(parents=True, exist_ok=True)

    source_index = read_json(dataset_root / "dataset.json")
    public_base_path = (args.public_base_path or f"/static-datasets/{source_index['dataset_id']}").rstrip("/")
    label_sources = read_json(dataset_root / "label-sources.json") if (dataset_root / "label-sources.json").exists() else {"sources": []}
    crossmask_predictions = load_crossmask_predictions(args.crossmask_results) if args.crossmask_results else {}
    if crossmask_predictions:
        label_sources = ensure_crossmask_label_sources(label_sources, crossmask_predictions)
    static_shards: list[dict[str, Any]] = []
    scene_stats: dict[str, dict[str, Any]] = {}
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    city_counts: Counter[str] = Counter()
    selected_count = 0

    for shard in source_index["shards"]:
        source_path = dataset_root / shard["path"]
        target_path = Path("shards") / f"{safe_name(shard['shard_id'])}.jsonl.gz"
        stats = compress_shard(
            source_path,
            output_root / target_path,
            include_masks=args.include_masks,
            output_root=output_root,
            project_root=args.project_root,
            public_base_path=public_base_path,
            crossmask_predictions=crossmask_predictions,
        )
        if stats["tile_count"] == 0:
            continue
        static_shard = {
            **shard,
            "path": target_path.as_posix(),
            "bbox_mercator": stats["bbox_mercator"],
            "center_mercator": stats["center_mercator"],
            "labels": stats["labels"],
            "mask_count": stats["mask_count"],
            "cities": stats["cities"],
            "splits": stats["splits"],
            "selected_count": stats["selected_count"],
            "tile_count": stats["tile_count"],
        }
        static_shards.append(static_shard)
        selected_count += stats["selected_count"]
        label_counts.update(stats["labels"])
        split_counts.update(stats["splits"])
        city_counts.update(stats["cities"])
        for scene_id, scene in stats["scenes"].items():
            existing = scene_stats.setdefault(
                scene_id,
                {
                    "scene_id": scene_id,
                    "city": scene["city"],
                    "split": scene["split"],
                    "tile_count": 0,
                    "labels": Counter(),
                    "bbox": None,
                },
            )
            existing["tile_count"] += scene["tile_count"]
            existing["labels"].update(scene["labels"])
            existing["bbox"] = merge_bbox(existing["bbox"], scene["bbox"])

    scenes = []
    for scene in scene_stats.values():
        scenes.append(
            {
                "scene_id": scene["scene_id"],
                "city": scene["city"],
                "split": scene["split"],
                "tile_count": scene["tile_count"],
                "labels": dict(scene["labels"]),
                "bbox_mercator": round_bbox(scene["bbox"]),
            }
        )

    static_index = {
        **source_index,
        "format": "crosswalk-static-jsonl-v1",
        "source_format": source_index.get("format"),
        "static_export_version": 1,
        "dataset_url": args.dataset_url,
        "label_sources": label_sources,
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "city_counts": dict(city_counts),
        "selected_count": selected_count,
        "scenes": sorted(scenes, key=lambda item: (item["city"], item["scene_id"])),
        "shards": static_shards,
    }

    (output_root / "dataset.json").write_text(json.dumps(static_index, separators=(",", ":")), encoding="utf-8")
    (output_root / "label-sources.json").write_text(json.dumps(label_sources, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(dataset_root),
                "output": str(output_root),
                "shards": len(static_shards),
                "tile_count": source_index.get("tile_count"),
                "compressed_bytes": directory_size(output_root),
                "masks": sum(shard.get("mask_count", 0) for shard in static_shards),
                "crossmask_predictions": len(crossmask_predictions),
            },
            indent=2,
        )
    )
    return 0


def compress_shard(
    source_path: Path,
    target_path: Path,
    *,
    include_masks: bool,
    output_root: Path,
    project_root: Path,
    public_base_path: str,
    crossmask_predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    cities: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    scenes: dict[str, dict[str, Any]] = {}
    bbox: list[float] | None = None
    tile_count = 0
    mask_count = 0
    selected_count = 0

    with source_path.open("rt", encoding="utf-8") as source, gzip.open(target_path, "wt", encoding="utf-8", compresslevel=9) as target:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            image = json.loads(stripped)
            merge_crossmask_vote(image, crossmask_predictions.get(str(image.get("tile_id"))))
            if include_masks:
                mask_count += copy_mask_artifact(image, output_root, project_root, public_base_path)
            target.write(json.dumps(image, separators=(",", ":")))
            target.write("\n")
            tile_count += 1
            if image.get("selected_for_training"):
                selected_count += 1
            label = image.get("resolved_label", {}).get("decision", "unknown")
            city = image.get("city", "unknown")
            split = image.get("split", "unknown")
            scene_id = image.get("scene_id", "unknown")
            tile_bbox = image.get("bbox_mercator")
            labels[label] += 1
            cities[city] += 1
            splits[split] += 1
            bbox = merge_bbox(bbox, tile_bbox)
            scene = scenes.setdefault(scene_id, {"city": city, "split": split, "tile_count": 0, "labels": Counter(), "bbox": None})
            scene["tile_count"] += 1
            scene["labels"][label] += 1
            scene["bbox"] = merge_bbox(scene["bbox"], tile_bbox)

    return {
        "bbox_mercator": round_bbox(bbox),
        "center_mercator": bbox_center(bbox),
        "cities": dict(cities),
        "labels": dict(labels),
        "mask_count": mask_count,
        "scenes": scenes,
        "selected_count": selected_count,
        "splits": dict(splits),
        "tile_count": tile_count,
    }


def copy_mask_artifact(image: dict[str, Any], output_root: Path, project_root: Path, public_base_path: str) -> int:
    copied = 0
    for vote in image.get("labels", []):
        source_id = str((vote.get("source") or {}).get("source_id") or "unknown")
        metadata = vote.get("metadata")
        if not isinstance(metadata, dict):
            continue
        artifact = metadata.get("mask_artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            source = resolve_artifact_path(artifact.get("path"), project_root)
            if not source or not source.exists():
                continue
            shard_id = image.get("scene_id", "unknown")
            target_relative = Path("masks") / safe_name(source_id) / safe_name(str(shard_id)) / source.name
            target = output_root / target_relative
            target.parent.mkdir(parents=True, exist_ok=True)
            color = mask_color(source_id, artifact)
            write_mask_overlay(source, target, color=color)
            artifact["static_path"] = target_relative.as_posix()
            artifact["url"] = f"{public_base_path}/{target_relative.as_posix()}"
            artifact["overlay"] = {"color": rgb_hex(color[:3]), "alpha": color[3]}
            copied += 1
    return copied


def merge_crossmask_vote(image: dict[str, Any], prediction: dict[str, Any] | None) -> None:
    if not prediction:
        return
    source_id = str(prediction.get("source_id", "crossmasknet-v4-full"))
    labels = image.setdefault("labels", [])
    if any(((vote.get("source") or {}).get("source_id") == source_id) for vote in labels if isinstance(vote, dict)):
        return
    metadata = {
        "mask_coverage": prediction.get("mask_coverage"),
        "mask_score": prediction.get("mask_score"),
        "model": prediction.get("model", "CrossMaskNet v4"),
        "run_id": prediction.get("run_id", "crossmasknet-v4-full"),
    }
    if prediction.get("mask_artifact"):
        metadata["mask_artifact"] = prediction["mask_artifact"]
    labels.append(
        {
            "vote_id": f"{source_id}:{image.get('tile_id')}",
            "source": {
                "source_id": source_id,
                "kind": "model",
                "priority": 120,
                "display_name": prediction.get("display_name", "CrossMaskNet v4"),
            },
            "decision": prediction.get("decision", "no_crosswalk"),
            "confidence": prediction.get("confidence"),
            "created_at": prediction.get("created_at", "2026-05-22T00:00:00Z"),
            "metadata": metadata,
        }
    )


def load_crossmask_predictions(results_root: Path) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for path in sorted(results_root.glob("shards/*.jsonl")):
        with path.open("rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                predictions[str(row["tile_id"])] = row
    return predictions


def ensure_crossmask_label_sources(label_sources: Any, predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = label_sources if isinstance(label_sources, dict) else {"sources": label_sources if isinstance(label_sources, list) else []}
    sources = payload.setdefault("sources", [])
    existing = {str(source.get("source_id")) for source in sources if isinstance(source, dict)}
    additions: dict[str, dict[str, Any]] = {}
    for prediction in predictions.values():
        source_id = str(prediction.get("source_id", "crossmasknet-v4-full"))
        if source_id in existing or source_id in additions:
            continue
        additions[source_id] = {
            "source_id": source_id,
            "kind": "model",
            "priority": 120,
            "display_name": prediction.get("display_name", "CrossMaskNet v4"),
            "description": "CrossMaskNet segmentation prediction generated by the project-trained model.",
        }
    sources.extend(additions.values())
    return payload


def resolve_artifact_path(value: Any, project_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def mask_color(source_id: str, artifact: dict[str, Any]) -> tuple[int, int, int, int]:
    kind = str(artifact.get("kind", ""))
    if source_id.startswith("crossmask") or "crossmask" in kind:
        return (236, 72, 153, 170)
    return (34, 211, 238, 160)


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def write_mask_overlay(source: Path, target: Path, *, color: tuple[int, int, int, int]) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required for --include-masks.") from exc

    mask = Image.open(source).convert("L")
    overlay = Image.new("RGBA", mask.size, (color[0], color[1], color[2], 0))
    overlay.putalpha(mask.point(lambda value: color[3] if value else 0))
    overlay.save(target, optimize=True)


def merge_bbox(left: list[float] | None, right: Any) -> list[float] | None:
    if not isinstance(right, list) or len(right) != 4:
        return left
    values = [float(value) for value in right]
    if left is None:
        return values
    return [min(left[0], values[0]), min(left[1], values[1]), max(left[2], values[2]), max(left[3], values[3])]


def bbox_center(bbox: list[float] | None) -> list[float]:
    if bbox is None:
        return [0, 0]
    return [round((bbox[0] + bbox[2]) / 2, 3), round((bbox[1] + bbox[3]) / 2, 3)]


def round_bbox(bbox: list[float] | None) -> list[float]:
    if bbox is None:
        return [0, 0, 0, 0]
    return [round(value, 3) for value in bbox]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "-" for char in value)


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


if __name__ == "__main__":
    raise SystemExit(main())
