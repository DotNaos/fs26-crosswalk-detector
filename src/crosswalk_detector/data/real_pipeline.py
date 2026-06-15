"""Real-data pipeline based on large WMS mosaics and tile-level auto-labeling."""

from __future__ import annotations

from dataclasses import asdict
import csv
import gzip
import json
import math
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from PIL import Image
import torch
from transformers import AutoProcessor, CLIPModel

from ..geo.pilot import derive_split_targets, crosswalk_score, repo_root
from .real_config import load_real_pipeline_config, SceneSpec

WMS_BASE_URL = "https://wms.geo.admin.ch/"
EARTH_RADIUS_M = 6_378_137.0


POSITIVE_PROMPTS = (
    "an aerial image tile containing a zebra crossing on a road",
    "an overhead satellite tile showing a pedestrian crosswalk",
)
NEGATIVE_PROMPTS = (
    "an aerial image tile of a road surface without a zebra crossing",
    "an overhead satellite tile of a street or intersection without a pedestrian crossing",
)
OFFROAD_PROMPTS = (
    "an aerial image tile of water, grass, forest, or fields",
    "an overhead satellite tile of rooftops, courtyards, or non-road surfaces",
)


def mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(longitude)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return x, y


def scene_bbox(scene: SceneSpec) -> tuple[float, float, float, float]:
    center_x, center_y = mercator_from_lat_lon(scene.latitude, scene.longitude)
    half = scene.size_m / 2.0
    return (
        center_x - half,
        center_y - half,
        center_x + half,
        center_y + half,
    )


def scene_tile_size_px(scene: SceneSpec, tile_size_m: int) -> int:
    pixels_per_meter = scene.image_px / scene.size_m
    tile_px = int(round(tile_size_m * pixels_per_meter))
    if tile_px <= 0:
        raise ValueError("tile size in pixels must stay positive")
    return tile_px


def build_wms_url(scene: SceneSpec) -> str:
    bbox = ",".join(str(value) for value in scene_bbox(scene))
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "ch.swisstopo.swissimage-product",
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": bbox,
            "WIDTH": scene.image_px,
            "HEIGHT": scene.image_px,
            "FORMAT": "image/jpeg",
        }
    )
    return f"{WMS_BASE_URL}?{query}"


def raw_scene_path(run_name: str, scene: SceneSpec) -> Path:
    return repo_root() / "data" / "raw" / run_name / "wms-mosaics" / f"{scene.scene_id}.jpg"


def tile_root(run_name: str) -> Path:
    return repo_root() / "data" / "processed" / run_name / "tiles"


def ensure_scene_download(run_name: str, scene: SceneSpec) -> Path:
    destination = raw_scene_path(run_name, scene)
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(build_wms_url(scene)) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination


def extract_tiles_for_scene(run_name: str, scene: SceneSpec, tile_size_m: int = 25) -> list[dict[str, object]]:
    source = ensure_scene_download(run_name, scene)
    image = Image.open(source).convert("RGB")
    tile_px = scene_tile_size_px(scene, tile_size_m)
    tiles_per_axis = scene.image_px // tile_px
    scene_dir = tile_root(run_name) / scene.scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    tile_rows: list[dict[str, object]] = []
    min_x, min_y, max_x, max_y = scene_bbox(scene)
    meters_per_px = scene.size_m / scene.image_px

    for row in range(tiles_per_axis):
        for col in range(tiles_per_axis):
            left = col * tile_px
            top = row * tile_px
            tile = image.crop((left, top, left + tile_px, top + tile_px))
            rel_path = f"{scene.scene_id}/r{row:02d}-c{col:02d}.jpg"
            output_path = scene_dir / f"r{row:02d}-c{col:02d}.jpg"
            if not output_path.exists():
                tile.save(output_path, quality=95)
            tile_min_x = min_x + (left * meters_per_px)
            tile_min_y = max_y - ((top + tile_px) * meters_per_px)
            tile_max_x = min_x + ((left + tile_px) * meters_per_px)
            tile_max_y = max_y - (top * meters_per_px)
            tile_rows.append(
                {
                    "tile_id": f"{scene.scene_id}:r{row:02d}:c{col:02d}",
                    "scene_id": scene.scene_id,
                    "city": scene.city,
                    "split": scene.split,
                    "row": row,
                    "col": col,
                    "relative_path": rel_path,
                    "image_path": str(output_path),
                    "bbox_mercator": [tile_min_x, tile_min_y, tile_max_x, tile_max_y],
                }
            )
    return tile_rows


class ClipCrosswalkScorer:
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        token = os.getenv("HF_TOKEN")
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = CLIPModel.from_pretrained(model_id, token=token)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def score_paths(self, paths: list[Path], batch_size: int = 64) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        texts = list(POSITIVE_PROMPTS + NEGATIVE_PROMPTS + OFFROAD_PROMPTS)
        positive_count = len(POSITIVE_PROMPTS)
        negative_count = len(NEGATIVE_PROMPTS)

        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits_per_image
                positive_logits = logits[:, :positive_count].mean(dim=1)
                negative_logits = logits[:, positive_count : positive_count + negative_count].mean(dim=1)
                offroad_logits = logits[:, positive_count + negative_count :].mean(dim=1)
                probs = torch.softmax(
                    torch.stack([positive_logits, negative_logits, offroad_logits], dim=1),
                    dim=1,
                )

            for path, prob, pos_logit, neg_logit, offroad_logit in zip(
                batch_paths,
                probs.cpu().tolist(),
                positive_logits.cpu().tolist(),
                negative_logits.cpu().tolist(),
                offroad_logits.cpu().tolist(),
            ):
                results.append(
                    {
                        "image_path": str(path),
                        "clip_probability": float(prob[0]),
                        "road_negative_probability": float(prob[1]),
                        "offroad_probability": float(prob[2]),
                        "clip_margin": float(pos_logit - neg_logit),
                        "offroad_margin": float(neg_logit - offroad_logit),
                    }
                )

        return results


def _heuristic_probability(path: Path) -> float:
    score = crosswalk_score(path)
    return 1.0 / (1.0 + math.exp(-(score / 8.0)))


def build_real_dataset(
    run_name: str | None = None,
    export_name: str | None = None,
    target_per_class: int | None = None,
    tile_size_m: int | None = None,
    config_path: Path | None = None,
) -> dict[str, object]:
    config = load_real_pipeline_config(config_path)
    run_name = run_name or config.run_name
    export_name = export_name or config.export_name
    target_per_class = target_per_class or config.target_per_class
    tile_size_m = tile_size_m or config.tile_size_m

    processed_root = repo_root() / "data" / "processed" / run_name
    manifests_root = processed_root / "manifests"
    exports_root = processed_root / "exports" / export_name
    manifests_root.mkdir(parents=True, exist_ok=True)
    exports_root.mkdir(parents=True, exist_ok=True)

    all_tiles: list[dict[str, object]] = []
    for scene in config.scenes:
        all_tiles.extend(extract_tiles_for_scene(run_name, scene, tile_size_m))

    scorer = ClipCrosswalkScorer()
    all_paths = [Path(tile["image_path"]) for tile in all_tiles]
    score_rows = scorer.score_paths(all_paths)
    score_by_path = {row["image_path"]: row for row in score_rows}

    scene_lookup = {scene.scene_id: scene for scene in config.scenes}
    for tile in all_tiles:
        path = Path(tile["image_path"])
        tile["relative_path"] = f"{tile['scene_id']}/r{tile['row']:02d}-c{tile['col']:02d}.jpg"
        clip = score_by_path[str(path)]
        heuristic_probability = _heuristic_probability(path)
        combined_probability = (clip["clip_probability"] * 0.7) + (heuristic_probability * 0.3)
        road_surface_score = float(clip["road_negative_probability"]) - float(clip["offroad_probability"])
        hard_negative_score = (
            float(clip["road_negative_probability"])
            - float(clip["offroad_probability"])
            - (float(clip["clip_probability"]) * config.selection.negative_positive_penalty)
        )
        tile["clip_probability"] = round(float(clip["clip_probability"]), 6)
        tile["road_negative_probability"] = round(float(clip["road_negative_probability"]), 6)
        tile["offroad_probability"] = round(float(clip["offroad_probability"]), 6)
        tile["heuristic_probability"] = round(float(heuristic_probability), 6)
        tile["combined_probability"] = round(float(combined_probability), 6)
        tile["road_surface_score"] = round(float(road_surface_score), 6)
        tile["hard_negative_score"] = round(float(hard_negative_score), 6)
        tile["predicted_label"] = "crosswalk" if combined_probability >= 0.5 else "no_crosswalk"
        tile["selected"] = False
        tile["status"] = "dropped"
        tile["label"] = tile["predicted_label"]
        tile["review_source"] = "auto"
        tile["scene"] = asdict(scene_lookup[tile["scene_id"]])

    split_targets = derive_split_targets(target_per_class, config.split_ratios)
    selected_rows: list[dict[str, object]] = []
    per_split_counts: dict[str, dict[str, int]] = {}

    for split, target in split_targets.items():
        split_tiles = [tile for tile in all_tiles if tile["split"] == split]
        positive_candidates = [
            tile
            for tile in split_tiles
            if float(tile["combined_probability"]) >= config.selection.positive_min_combined
            and float(tile["road_surface_score"]) >= config.selection.positive_min_road_surface
            and float(tile["heuristic_probability"]) >= config.selection.positive_min_heuristic
        ]
        positive_candidates.sort(
            key=lambda tile: (
                float(tile["combined_probability"]),
                float(tile["road_surface_score"]),
                float(tile["heuristic_probability"]),
            ),
            reverse=True,
        )
        selected_positive = positive_candidates[:target]
        if len(selected_positive) < target:
            selected_ids = {tile["tile_id"] for tile in selected_positive}
            fallback_positives = [
                tile
                for tile in sorted(
                    split_tiles,
                    key=lambda tile: (
                        float(tile["combined_probability"]),
                        float(tile["road_surface_score"]),
                        float(tile["heuristic_probability"]),
                    ),
                    reverse=True,
                )
                if tile["tile_id"] not in selected_ids
            ]
            selected_positive.extend(fallback_positives[: target - len(selected_positive)])
        positive_ids = {tile["tile_id"] for tile in selected_positive}
        remaining_tiles = [
            tile
            for tile in split_tiles
            if tile["tile_id"] not in positive_ids
        ]
        hard_negative_candidates = [
            tile
            for tile in remaining_tiles
            if float(tile["combined_probability"]) < config.selection.negative_max_combined
        ]
        hard_negative_candidates.sort(
            key=lambda tile: (
                float(tile["hard_negative_score"]),
                float(tile["road_negative_probability"]),
                -float(tile["combined_probability"]),
            ),
            reverse=True,
        )
        selected_negative = hard_negative_candidates[:target]
        if len(selected_negative) < target:
            selected_ids = {tile["tile_id"] for tile in selected_negative}
            fallback_negatives = [
                tile
                for tile in sorted(
                    remaining_tiles,
                    key=lambda tile: (
                        float(tile["hard_negative_score"]),
                        float(tile["road_negative_probability"]),
                        -float(tile["combined_probability"]),
                    ),
                    reverse=True,
                )
                if tile["tile_id"] not in selected_ids
            ]
            selected_negative.extend(fallback_negatives[: target - len(selected_negative)])

        for tile in selected_positive:
            tile["selected"] = True
            tile["status"] = "selected"
            tile["label"] = "crosswalk"
        for tile in selected_negative:
            tile["selected"] = True
            tile["status"] = "selected"
            tile["label"] = "no_crosswalk"

        selected_rows.extend(selected_positive)
        selected_rows.extend(selected_negative)
        per_split_counts[split] = {
            "crosswalk": len(selected_positive),
            "no_crosswalk": len(selected_negative),
        }

    selected_rows = sorted(selected_rows, key=lambda row: (str(row["split"]), str(row["tile_id"])))
    all_tiles = sorted(all_tiles, key=lambda row: (str(row["split"]), str(row["scene_id"]), int(row["row"]), int(row["col"])))

    labels_csv = exports_root / "labels.csv"
    with labels_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "scene_id",
                "city",
                "split",
                "row",
                "col",
                "relative_path",
                "image_path",
                "label",
                "clip_probability",
                "road_negative_probability",
                "offroad_probability",
                "heuristic_probability",
                "combined_probability",
                "road_surface_score",
                "hard_negative_score",
                "review_source",
            ],
        )
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    tiles_json = exports_root / "tiles.json"
    tiles_json.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "export_name": export_name,
                "target_per_class": target_per_class,
                "split_targets": split_targets,
                "scenes": [asdict(scene) for scene in config.scenes],
                "tiles": all_tiles,
            },
            indent=2,
        )
    )

    summary = {
        "run_name": run_name,
        "export_name": export_name,
        "target_per_class": target_per_class,
        "split_targets": split_targets,
        "selected_total": len(selected_rows),
        "positive_count": sum(1 for row in selected_rows if row["label"] == "crosswalk"),
        "negative_count": sum(1 for row in selected_rows if row["label"] == "no_crosswalk"),
        "per_split_counts": per_split_counts,
        "labels_csv": str(labels_csv),
        "tiles_json": str(tiles_json),
    }
    (exports_root / "summary.json").write_text(json.dumps(summary, indent=2))
    (manifests_root / "real-scenes.json").write_text(json.dumps([asdict(scene) for scene in config.scenes], indent=2))
    write_compact_manifest(exports_root, run_name, export_name, config.scenes, selected_rows)
    return summary


def write_compact_manifest(
    export_root: Path,
    run_name: str,
    export_name: str,
    scenes: tuple[SceneSpec, ...],
    selected_rows: list[dict[str, object]],
) -> dict[str, str]:
    """Write a small gzip manifest that can recreate the selected tile dataset."""

    compact_json = export_root / "compact-manifest.json.gz"
    compact_csv = export_root / "labels.compact.csv.gz"
    scenes_payload = []
    for scene in scenes:
        scenes_payload.append(
            {
                **asdict(scene),
                "bbox_mercator": [round(value, 3) for value in scene_bbox(scene)],
                "wms_url": build_wms_url(scene),
            }
        )

    rows_payload = []
    for row in selected_rows:
        rows_payload.append(
            {
                "tile_id": row["tile_id"],
                "scene_id": row["scene_id"],
                "split": row["split"],
                "label": row["label"],
                "row": row["row"],
                "col": row["col"],
                "bbox_mercator": [round(float(value), 3) for value in row["bbox_mercator"]],
            }
        )

    payload = {
        "run_name": run_name,
        "export_name": export_name,
        "format": "crosswalk-compact-v1",
        "source_layer": "ch.swisstopo.swissimage-product",
        "tile_size_m": 25,
        "scenes": scenes_payload,
        "tiles": rows_payload,
    }
    with gzip.open(compact_json, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    with gzip.open(compact_csv, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "tile_id",
                "scene_id",
                "split",
                "label",
                "row",
                "col",
                "min_x",
                "min_y",
                "max_x",
                "max_y",
            ],
        )
        writer.writeheader()
        for row in selected_rows:
            min_x, min_y, max_x, max_y = [round(float(value), 3) for value in row["bbox_mercator"]]
            writer.writerow(
                {
                    "tile_id": row["tile_id"],
                    "scene_id": row["scene_id"],
                    "split": row["split"],
                    "label": row["label"],
                    "row": row["row"],
                    "col": row["col"],
                    "min_x": min_x,
                    "min_y": min_y,
                    "max_x": max_x,
                    "max_y": max_y,
                }
            )

    return {
        "compact_json": str(compact_json),
        "compact_csv": str(compact_csv),
    }
