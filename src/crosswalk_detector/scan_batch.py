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
    SceneRequest,
    SAM31_MODEL_VERSION,
    SUPERVISED_CLIP_MODEL_PATH,
    TileRequest,
    _sigmoid,
    _tile_pixel_bounds,
    crop_tile,
    crosswalk_score_image,
    create_scan_backend,
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
    scanner = create_scan_backend()

    scene_image = fetch_scene_image(scene)
    if scanner.backend_name == "sam31":
        detections = scanner.detect_scene(scene_image)
        sam31_scores = scanner.score_tiles(scene, tiles, detections)
        effective_threshold = scanner.tile_threshold
    else:
        effective_threshold = scanner.supervised_threshold if scanner.has_supervised_classifier else threshold
        detections = [] if scanner.has_supervised_classifier else scanner.detect_boxes(scene_image)
        sam31_scores = []

    results: dict[str, dict[str, Any]] = {}
    result_tiles: list[dict[str, Any]] = []
    crosswalk_count = 0
    no_crosswalk_count = 0
    context_images = [] if scanner.backend_name == "sam31" else [crop_tile(scene_image, scene, tile, padding_tiles=1.0) for tile in tiles]
    supervised_scores = (
        scanner.score_supervised_context_images(context_images)
        if scanner.backend_name != "sam31" and scanner.has_supervised_classifier
        else []
    )
    for index, tile in enumerate(tiles, start=1):
        if scanner.backend_name == "sam31":
            sam_metrics = sam31_scores[index - 1]
            prediction = {
                "tile_id": tile.tile_id,
                "label": sam_metrics.label,
                "score": round(sam_metrics.score, 6),
                "peak": round(sam_metrics.peak, 6),
                "coverage": round(sam_metrics.coverage, 6),
                "box_overlap": round(sam_metrics.box_overlap, 6),
                "sam_detection_count": sam_metrics.detection_count,
                "prompt": sam_metrics.prompt,
                "selected": True,
                "review_source": "python-sam31-scan",
            }
            label = sam_metrics.label
        else:
            tile_bounds = _tile_pixel_bounds(scene, tile.bbox_mercator)
            detector_score = detector_overlap_score(detections, tile_bounds)
            center_image = crop_tile(scene_image, scene, tile)
            heuristic_probability = _sigmoid(crosswalk_score_image(center_image) / 8.0)
            supervised_probability = supervised_scores[index - 1] if supervised_scores else None
            if supervised_probability is not None:
                clip_positive, clip_negative = supervised_probability, 1.0 - supervised_probability
            else:
                clip_positive, clip_negative = scanner.score_context_images([context_images[index - 1]])[0]
            metrics = decide_tile_label(
                clip_positive,
                clip_negative,
                heuristic_probability,
                detector_score,
                effective_threshold,
                supervised_probability=supervised_probability,
            )
            prediction = {
                "tile_id": tile.tile_id,
                "label": metrics.label,
                "score": round(metrics.combined_score, 6),
                "peak": round(max(metrics.clip_positive, metrics.detector_score), 6),
                "coverage": round(metrics.heuristic_probability, 6),
                "supervised_probability": round(metrics.supervised_probability, 6) if metrics.supervised_probability is not None else None,
                "prompt": "server-clip-linear" if supervised_probability is not None else "server-hybrid",
                "selected": True,
                "review_source": "python-clip-linear-scan" if supervised_probability is not None else "python-hybrid-scan",
            }
            label = metrics.label
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
        if label == "crosswalk":
            crosswalk_count += 1
        else:
            no_crosswalk_count += 1
        if progress:
            print(
                f"[{index:>4}/{len(tiles)}] {tile.tile_id} -> {label} ({prediction['score']:.3f})",
                flush=True,
            )

    return {
        "version": 1,
        "created_at": _utc_now(),
        "completed_at": _utc_now(),
        "job": job_payload,
        "scanner": {
            "detector_model": DETECTOR_MODEL_ID,
            "clip_model": CLIP_MODEL_ID,
            "backend": scanner.backend_name,
            "sam_model": SAM31_MODEL_VERSION if scanner.backend_name == "sam31" else None,
            "sam_prompts": list(scanner.prompts) if scanner.backend_name == "sam31" else None,
            "sam_confidence_threshold": scanner.confidence_threshold if scanner.backend_name == "sam31" else None,
            "sam_tile_threshold": scanner.tile_threshold if scanner.backend_name == "sam31" else None,
            "supervised_model": "clip_linear" if scanner.backend_name != "sam31" and scanner.has_supervised_classifier else None,
            "supervised_model_path": str(SUPERVISED_CLIP_MODEL_PATH) if scanner.backend_name != "sam31" and scanner.has_supervised_classifier else None,
            "supervised_threshold": scanner.supervised_threshold if scanner.backend_name != "sam31" and scanner.has_supervised_classifier else None,
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
