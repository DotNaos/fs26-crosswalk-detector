"""SAM3 metadata dataset scaffold, job, merge, and export helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import shutil
import tomllib
from typing import Any, Iterable

from .metadata_dataset import resolve_label_votes, resolve_shard_path
from .raw_imagery import load_cached_scene_image
from .real_pipeline import scene_bbox
from .real_config import SceneSpec
from .scan_backend import SceneRequest, TileRequest, crop_tile


@dataclass(frozen=True)
class SceneGroup:
    city: str
    split: str
    latitude: float
    longitude: float
    grid_rows: int
    grid_cols: int


@dataclass(frozen=True)
class Sam3DatasetConfig:
    dataset_id: str
    display_name: str
    run_name: str
    export_name: str
    target_count: int
    tile_size_m: int
    scene_size_m: int
    image_px: int
    shard_target_count: int
    source_access: str
    source_crs: str
    source_resolution_m: float
    groups: tuple[SceneGroup, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_sam3_dataset_config(path: Path) -> Sam3DatasetConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    source = raw.get("source", {})
    groups = tuple(
        SceneGroup(
            city=str(group["city"]),
            split=str(group["split"]),
            latitude=float(group["latitude"]),
            longitude=float(group["longitude"]),
            grid_rows=int(group["grid_rows"]),
            grid_cols=int(group["grid_cols"]),
        )
        for group in raw.get("scene_groups", [])
    )
    if not groups:
        raise ValueError(f"No scene_groups configured in {path}")
    return Sam3DatasetConfig(
        dataset_id=str(raw.get("dataset_id", "sam3-100k-v1")),
        display_name=str(raw.get("display_name", "SAM3 100k v1")),
        run_name=str(raw.get("run_name", "sam3-100k-v1")),
        export_name=str(raw.get("export_name", "metadata-100k-v1")),
        target_count=int(raw.get("target_count", 100_000)),
        tile_size_m=int(raw.get("tile_size_m", 25)),
        scene_size_m=int(raw.get("scene_size_m", 800)),
        image_px=int(raw.get("image_px", 2048)),
        shard_target_count=int(raw.get("shard_target_count", 1024)),
        source_access=str(source.get("access", "wmts")),
        source_crs=str(source.get("crs", "EPSG:2056")),
        source_resolution_m=float(source.get("resolution_m", 0.1)),
        groups=groups,
    )


def prepare_sam3_metadata_dataset(config_path: Path, dataset_root: Path, overwrite: bool = False) -> dict[str, Any]:
    config = load_sam3_dataset_config(config_path)
    if dataset_root.exists() and overwrite:
        shutil.rmtree(dataset_root)
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(f"Dataset root already exists and is not empty: {dataset_root}")

    scenes = _expand_scene_groups(config)
    dataset_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "sources").mkdir(parents=True, exist_ok=True)
    total_rows = 0
    selected_rows = 0
    shards: list[dict[str, Any]] = []

    _write_json(dataset_root / "label-sources.json", _label_sources())
    _write_json(
        dataset_root / "sources" / "swisstopo.json",
        {
            "provider": "swisstopo",
            "product": "SWISSIMAGE",
            "access": config.source_access,
            "crs": config.source_crs,
            "resolution_m": config.source_resolution_m,
        },
    )

    for scene in scenes:
        if total_rows >= config.target_count:
            break
        rows = _rows_for_scene(config, scene, remaining=config.target_count - total_rows)
        if not rows:
            continue
        scene_root = dataset_root / "scenes" / scene.scene_id
        perimeter_root = scene_root / "perimeters" / "p0000"
        perimeter_root.mkdir(parents=True, exist_ok=True)
        bbox = [round(value, 6) for value in scene_bbox(scene)]
        scene_payload = {
            **asdict(scene),
            "tile_size_m": config.tile_size_m,
            "bbox_mercator": bbox,
            "source_scene_id": f"swissimage-{config.dataset_id}-{scene.scene_id}",
        }
        _write_json(scene_root / "scene.json", scene_payload)
        _write_json(
            scene_root / "road-overlay.json",
            {"overlay_id": f"roads-v1-{scene.scene_id}", "method": "metadata-scaffold", "cells": []},
        )
        _write_json(
            perimeter_root / "perimeter.json",
            {
                "perimeter_id": "p0000",
                "scene_id": scene.scene_id,
                "bbox_mercator": bbox,
                "tile_count": len(rows),
            },
        )
        shard_path = perimeter_root / "tiles.jsonl"
        _write_jsonl(shard_path, rows)
        shard_rel = shard_path.relative_to(dataset_root).as_posix()
        shards.append(
            {
                "shard_id": f"{scene.scene_id}-p0000",
                "path": shard_rel,
                "tile_count": len(rows),
                "scene_id": scene.scene_id,
                "perimeter_id": "p0000",
            }
        )
        total_rows += len(rows)
        selected_rows += sum(1 for row in rows if row["selected_for_training"])

    index = {
        "format": "crosswalk-jsonl-v1",
        "dataset_id": config.dataset_id,
        "display_name": config.display_name,
        "run_name": config.run_name,
        "export_name": config.export_name,
        "tile_count": total_rows,
        "selected_count": selected_rows,
        "shard_target_count": config.shard_target_count,
        "shards": shards,
    }
    _write_json(dataset_root / "dataset.json", index)
    return {"dataset_root": str(dataset_root), "tile_count": total_rows, "shard_count": len(shards)}


def build_sam3_shard_jobs(
    dataset_root: Path,
    output_root: Path,
    *,
    limit_shards: int | None = None,
    limit_tiles: int | None = None,
) -> dict[str, Any]:
    index = _read_json(dataset_root / "dataset.json")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    for shard_index, shard in enumerate(index["shards"]):
        if limit_shards is not None and shard_index >= limit_shards:
            break
        scene = _read_json(dataset_root / "scenes" / shard["scene_id"] / "scene.json")
        rows = list(_read_jsonl(resolve_shard_path(dataset_root, shard["path"])))
        if limit_tiles is not None:
            rows = rows[:limit_tiles]
        payload = {
            "version": 1,
            "dataset_id": index["dataset_id"],
            "shard_id": shard["shard_id"],
            "created_at": _utc_now(),
            "scene": {
                "scene_id": scene["scene_id"],
                "latitude": scene["latitude"],
                "longitude": scene["longitude"],
                "size_m": scene["size_m"],
                "image_px": scene["image_px"],
                "tile_size_m": scene["tile_size_m"],
            },
            "tiles": [
                {
                    "tile_id": row["tile_id"],
                    "image_id": row["image_id"],
                    "row": row["row"],
                    "col": row["col"],
                    "bbox_mercator": row["bbox_mercator"],
                    "relative_path": row["reconstruction"]["relative_path"],
                }
                for row in rows
            ],
        }
        path = output_root / f"shard-{shard_index:04d}.json"
        _write_json(path, payload)
        jobs.append({"path": str(path), "tile_count": len(rows), "shard_id": shard["shard_id"]})
    _write_json(output_root / "jobs.json", {"dataset_id": index["dataset_id"], "jobs": jobs})
    return {"output_root": str(output_root), "job_count": len(jobs), "tile_count": sum(job["tile_count"] for job in jobs)}


def merge_sam3_metadata_dataset(dataset_root: Path, results_root: Path, write: bool = False) -> dict[str, Any]:
    index = _read_json(dataset_root / "dataset.json")
    predictions = _load_result_predictions(results_root)
    updated = 0
    selected = 0
    now = _utc_now()
    for shard in index["shards"]:
        path = resolve_shard_path(dataset_root, shard["path"])
        rows = []
        for row in _read_jsonl(path):
            prediction = predictions.get(row["tile_id"])
            if prediction is not None:
                _merge_prediction_vote(row, prediction, now)
                updated += 1
            if row.get("selected_for_training") is True:
                selected += 1
            rows.append(row)
        if write:
            _write_jsonl(path, rows)
    if write:
        index["selected_count"] = selected
        _write_json(dataset_root / "dataset.json", index)
    return {"dataset_id": index["dataset_id"], "updated_rows": updated, "selected_count": selected}


def export_training_dataset(dataset_root: Path, output_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    index = _read_json(dataset_root / "dataset.json")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = _selected_rows(dataset_root, index)
    if limit is not None:
        rows = rows[:limit]
    scene_cache: dict[str, Any] = {}
    csv_path = output_root / "labels.csv"
    with csv_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "tile_id", "scene_id", "split", "label", "relative_path", "mask_path"])
        writer.writeheader()
        for row in rows:
            label = row["resolved_label"]["decision"]
            image_path = output_root / "images" / label / f"{row['image_id'].replace(':', '__')}.jpg"
            exported_mask_path = _export_row_mask(row, output_root)
            if not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                scene = _scene_request(dataset_root, row["scene_id"])
                scene_image = scene_cache.setdefault(row["scene_id"], load_cached_scene_image(dataset_root, scene))
                tile = TileRequest(
                    tile_id=row["tile_id"],
                    row=row["row"],
                    col=row["col"],
                    bbox_mercator=tuple(float(value) for value in row["bbox_mercator"]),
                    relative_path=row["reconstruction"]["relative_path"],
                )
                crop_tile(scene_image, scene, tile).save(image_path, quality=95)
            writer.writerow(
                {
                    "image_id": row["image_id"],
                    "tile_id": row["tile_id"],
                    "scene_id": row["scene_id"],
                    "split": row["split"],
                    "label": label,
                    "relative_path": image_path.relative_to(output_root).as_posix(),
                    "mask_path": exported_mask_path,
                }
            )
    summary = {"dataset_id": index["dataset_id"], "exported_count": len(rows), "labels_csv": str(csv_path)}
    _write_json(output_root / "summary.json", summary)
    return summary


def _expand_scene_groups(config: Sam3DatasetConfig) -> list[SceneSpec]:
    scenes = []
    for group in config.groups:
        lat_step = config.scene_size_m / 111_320.0
        lon_step = config.scene_size_m / (111_320.0 * max(0.1, math.cos(math.radians(group.latitude))))
        for row in range(group.grid_rows):
            for col in range(group.grid_cols):
                lat = group.latitude + (row - ((group.grid_rows - 1) / 2.0)) * lat_step
                lon = group.longitude + (col - ((group.grid_cols - 1) / 2.0)) * lon_step
                slug = _slug(group.city)
                scenes.append(
                    SceneSpec(
                        scene_id=f"{slug}-r{row:02d}-c{col:02d}",
                        city=group.city,
                        split=group.split,
                        latitude=round(lat, 7),
                        longitude=round(lon, 7),
                        size_m=config.scene_size_m,
                        image_px=config.image_px,
                    )
                )
    return scenes


def _rows_for_scene(config: Sam3DatasetConfig, scene: SceneSpec, remaining: int) -> list[dict[str, Any]]:
    tiles_per_axis = scene.size_m // config.tile_size_m
    tile_px = scene.image_px // tiles_per_axis
    min_x, min_y, max_x, max_y = scene_bbox(scene)
    meters_per_px = scene.size_m / scene.image_px
    rows = []
    for row in range(tiles_per_axis):
        for col in range(tiles_per_axis):
            if len(rows) >= remaining:
                return rows
            left = col * tile_px
            top = row * tile_px
            tile_min_x = min_x + (left * meters_per_px)
            tile_min_y = max_y - ((top + tile_px) * meters_per_px)
            tile_max_x = min_x + ((left + tile_px) * meters_per_px)
            tile_max_y = max_y - (top * meters_per_px)
            tile_id = f"{scene.scene_id}-r{row:02d}-c{col:02d}"
            image_id = f"{config.dataset_id}:{tile_id}"
            rows.append(_metadata_row(config, scene, image_id, tile_id, row, col, left, top, tile_px, [tile_min_x, tile_min_y, tile_max_x, tile_max_y]))
    return rows


def _metadata_row(config: Sam3DatasetConfig, scene: SceneSpec, image_id: str, tile_id: str, row: int, col: int, left: int, top: int, tile_px: int, bbox: list[float]) -> dict[str, Any]:
    created_at = _utc_now()
    source_scene_id = f"swissimage-{config.dataset_id}-{scene.scene_id}"
    vote = {
        "vote_id": f"unlabeled:{image_id}",
        "source": {"source_id": "unlabeled", "kind": "model", "priority": 0, "display_name": "Unlabeled scaffold"},
        "decision": "drop",
        "confidence": 0.0,
        "created_at": created_at,
    }
    return {
        "image_id": image_id,
        "tile_id": tile_id,
        "scene_id": scene.scene_id,
        "source_scene_id": source_scene_id,
        "perimeter_id": "p0000",
        "city": scene.city,
        "split": scene.split,
        "row": row,
        "col": col,
        "bbox_mercator": [round(value, 6) for value in bbox],
        "swisstopo": {"provider": "swisstopo", "product": "SWISSIMAGE", "access": config.source_access, "crs": config.source_crs, "asset_id": source_scene_id, "resolution_m": config.source_resolution_m},
        "reconstruction": {"source_scene_id": source_scene_id, "row": row, "col": col, "tile_size_m": config.tile_size_m, "tile_bbox_mercator": [round(value, 6) for value in bbox], "crop_px": {"left": left, "top": top, "width": tile_px, "height": tile_px}, "relative_path": f"images/{scene.scene_id}/r{row:02d}-c{col:02d}.jpg"},
        "road_overlay_ref": {"overlay_id": f"roads-v1-{scene.scene_id}", "perimeter_id": "p0000", "cell_id": tile_id, "surface_ratio": 0.0},
        "labels": [vote],
        "resolved_label": {"decision": "drop", "source_id": "unlabeled", "source_kind": "model", "resolved_by": "priority", "confidence": 0.0, "updated_at": created_at},
        "review_state": "unreviewed",
        "selected_for_training": False,
    }


def _merge_prediction_vote(row: dict[str, Any], prediction: dict[str, Any], created_at: str) -> None:
    source_id = str(prediction.get("source_id", "sam3.1"))
    labels = [vote for vote in row.get("labels", []) if vote.get("source", {}).get("source_id") != source_id]
    labels.append(
        {
            "vote_id": f"{source_id}:{created_at}:{row['tile_id']}",
            "source": {"source_id": source_id, "kind": "model", "priority": int(prediction.get("priority", 100)), "display_name": str(prediction.get("display_name", "SAM3.1"))},
            "decision": prediction["decision"],
            "confidence": prediction.get("confidence"),
            "created_at": created_at,
            "metadata": prediction.get("metadata", {}),
        }
    )
    row["labels"] = labels
    resolved = resolve_label_votes(labels)
    if resolved is not None:
        previous = row.get("resolved_label", {})
        row["resolved_label"] = resolved
        row["selected_for_training"] = resolved["decision"] in {"crosswalk", "no_crosswalk"}
        if previous.get("decision") not in {None, "drop", resolved["decision"]} and previous.get("source_kind") != "human":
            row["review_state"] = "disputed"


def _load_result_predictions(results_root: Path) -> dict[str, dict[str, Any]]:
    predictions = {}
    for path in sorted(results_root.rglob("*.json")):
        payload = _read_json(path)
        scanner = payload.get("scanner", {})
        source_id = scanner.get("sam_model") or scanner.get("backend") or "sam3.1"
        for tile in payload.get("tiles", []):
            label = tile.get("label")
            if label not in {"crosswalk", "no_crosswalk", "drop"}:
                continue
            predictions[str(tile["tile_id"])] = {
                "source_id": source_id,
                "display_name": str(source_id).upper() if str(source_id).startswith("sam") else str(source_id),
                "decision": label,
                "confidence": tile.get("score"),
                "metadata": {key: value for key, value in tile.items() if key not in {"tile_id", "label"}},
            }
    return predictions


def _export_row_mask(row: dict[str, Any], output_root: Path) -> str:
    artifact = _row_mask_artifact(row)
    if artifact is None:
        return ""
    source_path = Path(str(artifact.get("path", "")))
    if not source_path.exists():
        return ""
    label = row.get("resolved_label", {}).get("decision", "unknown")
    target_path = output_root / "masks" / str(label) / f"{row['image_id'].replace(':', '__')}.png"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not target_path.exists():
        shutil.copyfile(source_path, target_path)
    return target_path.relative_to(output_root).as_posix()


def _row_mask_artifact(row: dict[str, Any]) -> dict[str, Any] | None:
    resolved_source = row.get("resolved_label", {}).get("source_id")
    candidates = []
    for label in row.get("labels", []):
        if not isinstance(label, dict):
            continue
        metadata = label.get("metadata")
        if not isinstance(metadata, dict):
            continue
        artifact = metadata.get("mask_artifact")
        if isinstance(artifact, dict) and artifact.get("path"):
            candidates.append((label.get("source", {}).get("source_id"), artifact))
    for source_id, artifact in candidates:
        if source_id == resolved_source:
            return artifact
    return candidates[-1][1] if candidates else None


def _selected_rows(dataset_root: Path, index: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for shard in index["shards"]:
        for row in _read_jsonl(resolve_shard_path(dataset_root, shard["path"])):
            if row.get("selected_for_training") and row.get("resolved_label", {}).get("decision") in {"crosswalk", "no_crosswalk"}:
                rows.append(row)
    return rows


def _scene_request(dataset_root: Path, scene_id: str) -> SceneRequest:
    scene = _read_json(dataset_root / "scenes" / scene_id / "scene.json")
    return SceneRequest(scene_id=scene_id, latitude=float(scene["latitude"]), longitude=float(scene["longitude"]), size_m=int(scene["size_m"]), image_px=int(scene["image_px"]), tile_size_m=int(scene["tile_size_m"]))


def _label_sources() -> dict[str, Any]:
    return {
        "sources": [
            {"source_id": "unlabeled", "kind": "model", "priority": 0, "display_name": "Unlabeled scaffold"},
            {"source_id": "sam3.1", "kind": "model", "priority": 100, "display_name": "SAM3.1"},
            {"source_id": "human:oli", "kind": "human", "priority": 1000, "display_name": "Oli"},
        ]
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf8")


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-").replace("--", "-")
