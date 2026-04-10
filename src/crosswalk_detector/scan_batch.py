"""Batch job runner for Slurm or other offline scan workflows."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .scan_backend import (
    CLIP_MODEL_ID,
    DETECTOR_MODEL_ID,
    HybridCrosswalkScanner,
    SceneRequest,
    TileRequest,
    _sigmoid,
    _tile_pixel_bounds,
    crop_tile,
    crosswalk_score_image,
    decide_tile_label,
    detector_overlap_score,
    fetch_scene_image,
    mercator_to_lat_lon,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_scan_batch_job(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf8"))
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported scan job version in {path}")
    return payload


def _parse_scene(payload: dict[str, Any]) -> SceneRequest:
    scene = payload["scene"]
    return SceneRequest(
        scene_id=str(scene["scene_id"]),
        latitude=float(scene["latitude"]),
        longitude=float(scene["longitude"]),
        size_m=int(scene["size_m"]),
        image_px=int(scene["image_px"]),
        tile_size_m=int(scene["tile_size_m"]),
    )


def _parse_tiles(payload: dict[str, Any]) -> list[TileRequest]:
    return [
        TileRequest(
            tile_id=str(tile["tile_id"]),
            row=int(tile["row"]),
            col=int(tile["col"]),
            bbox_mercator=tuple(float(value) for value in tile["bbox_mercator"]),
            relative_path=str(tile["relative_path"]),
        )
        for tile in payload["tiles"]
    ]


def run_scan_batch_job(job_payload: dict[str, Any], progress: bool = False) -> dict[str, Any]:
    scene = _parse_scene(job_payload)
    tiles = _parse_tiles(job_payload)
    threshold = float(job_payload.get("threshold", 0.32))
    scanner = HybridCrosswalkScanner()

    scene_image = fetch_scene_image(scene)
    detections = scanner.detect_boxes(scene_image)
    context_images = [crop_tile(scene_image, scene, tile, padding_tiles=1.0) for tile in tiles]
    center_images = [crop_tile(scene_image, scene, tile) for tile in tiles]
    clip_scores = scanner.score_context_images(context_images)

    results: dict[str, dict[str, Any]] = {}
    result_tiles: list[dict[str, Any]] = []
    crosswalk_count = 0
    no_crosswalk_count = 0
    for index, tile in enumerate(tiles, start=1):
        tile_bounds = _tile_pixel_bounds(scene, tile.bbox_mercator)
        detector_score = detector_overlap_score(detections, tile_bounds)
        heuristic_probability = _sigmoid(crosswalk_score_image(center_images[index - 1]) / 8.0)
        clip_positive, clip_negative = clip_scores[index - 1]
        metrics = decide_tile_label(clip_positive, clip_negative, heuristic_probability, detector_score, threshold)
        prediction = {
            "tile_id": tile.tile_id,
            "label": metrics.label,
            "score": round(metrics.combined_score, 6),
            "peak": round(max(metrics.clip_positive, metrics.detector_score), 6),
            "coverage": round(metrics.heuristic_probability, 6),
            "prompt": "server-hybrid",
            "selected": True,
            "review_source": "python-hybrid-scan",
        }
        results[tile.tile_id] = prediction
        center_x = (tile.bbox_mercator[0] + tile.bbox_mercator[2]) / 2.0
        center_y = (tile.bbox_mercator[1] + tile.bbox_mercator[3]) / 2.0
        center_lat, center_lng = mercator_to_lat_lon(center_x, center_y)
        result_tiles.append(
            {
                "tile_id": tile.tile_id,
                "row": tile.row,
                "col": tile.col,
                "bbox_mercator": [round(value, 6) for value in tile.bbox_mercator],
                "center_mercator": {
                    "x": round(center_x, 6),
                    "y": round(center_y, 6),
                },
                "center_latlon": {
                    "latitude": round(center_lat, 7),
                    "longitude": round(center_lng, 7),
                },
                "relative_path": tile.relative_path,
                **prediction,
            }
        )
        if metrics.label == "crosswalk":
            crosswalk_count += 1
        else:
            no_crosswalk_count += 1
        if progress:
            print(f"[{index:>4}/{len(tiles)}] {tile.tile_id} -> {metrics.label} ({metrics.combined_score:.3f})")

    return {
        "version": 1,
        "created_at": _utc_now(),
        "completed_at": _utc_now(),
        "job": job_payload,
        "scanner": {
            "detector_model": DETECTOR_MODEL_ID,
            "clip_model": CLIP_MODEL_ID,
            "device": scanner.device,
        },
        "summary": {
            "total": len(tiles),
            "crosswalk": crosswalk_count,
            "no_crosswalk": no_crosswalk_count,
        },
        "tiles": result_tiles,
        "results": results,
    }


def write_scan_batch_result(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf8")
    return path


def summarize_scan_batch_result(result: dict[str, Any]) -> str:
    summary = result["summary"]
    scene_id = result["job"]["scene"]["scene_id"]
    return (
        f"Scene {scene_id}: {summary['total']} tiles, "
        f"{summary['crosswalk']} crosswalk, {summary['no_crosswalk']} no_crosswalk"
    )


__all__ = [
    "load_scan_batch_job",
    "run_scan_batch_job",
    "summarize_scan_batch_result",
    "write_scan_batch_result",
]
