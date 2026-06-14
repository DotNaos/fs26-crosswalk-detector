"""Run CrossMaskNet on ad-hoc metadata tiles."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .scan_backend import SceneRequest, TileRequest, crop_tile, fetch_scene_image
from .train_crossmask import CrossMaskNet
from .urban_vision import build_urban_road_density_mask


@dataclass(frozen=True)
class CrossMaskTile:
    tile_id: str
    scene_id: str
    row: int
    col: int
    bbox_mercator: tuple[float, float, float, float]
    relative_path: str


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def run_crossmask_request(request_path: Path, output_path: Path) -> dict[str, Any]:
    request = json.loads(request_path.read_text(encoding="utf8"))
    dataset_root = Path(request["dataset_root"])
    model_root = Path(request["model_root"])
    threshold = float(request.get("threshold", 0.005))
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf8"))
    device = _device()
    model = _load_model(metrics, model_root, device)
    image_size = int(metrics["image_size"])
    tiles = [_parse_tile(tile) for tile in request.get("tiles", [])]
    scene_cache: dict[str, Image.Image] = {}
    predictions = []

    for tile in tiles:
        scene = _scene_request(dataset_root, tile.scene_id)
        scene_image = scene_cache.setdefault(tile.scene_id, fetch_scene_image(scene))
        crop = crop_tile(scene_image, scene, TileRequest(tile.tile_id, tile.row, tile.col, tile.bbox_mercator, tile.relative_path))
        probability = _predict_probability(model, crop, image_size, device, road_channel=bool(metrics.get("road_channel")))
        mask = probability >= 0.5
        mask_coverage = float(mask.mean())
        mask_score = float(probability.mean())
        decision = "crosswalk" if mask_coverage >= threshold else "no_crosswalk"
        predictions.append(
            {
                "tile_id": tile.tile_id,
                "decision": decision,
                "confidence": round(max(mask_score, mask_coverage), 6),
                "mask_coverage": round(mask_coverage, 6),
                "mask_score": round(mask_score, 6),
            }
        )

    summary = {
        "total": len(predictions),
        "crosswalk": sum(1 for prediction in predictions if prediction["decision"] == "crosswalk"),
        "no_crosswalk": sum(1 for prediction in predictions if prediction["decision"] == "no_crosswalk"),
    }
    result = {
        "model": "CrossMaskNet v4",
        "model_root": str(model_root),
        "run_id": str(request.get("run_id", "")),
        "positive_threshold": threshold,
        "device": device,
        "summary": summary,
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


def run_crossmask_image_directory(
    input_dir: Path,
    output_dir: Path,
    model_root: Path,
    *,
    threshold: float = 0.005,
    include_overlays: bool = True,
) -> dict[str, Any]:
    metrics = json.loads((model_root / "metrics.json").read_text(encoding="utf8"))
    device = _device()
    model = _load_model(metrics, model_root, device)
    image_size = int(metrics["image_size"])
    road_channel = bool(metrics.get("road_channel"))

    positive_dir = output_dir / "positive"
    negative_dir = output_dir / "negative"
    overlay_dir = output_dir / "positive_overlays"
    for directory in (positive_dir, negative_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_path in _iter_image_paths(input_dir):
        image = Image.open(image_path).convert("RGB")
        probability = _predict_probability(model, image, image_size, device, road_channel=road_channel)
        mask = probability >= 0.5
        mask_coverage = float(mask.mean())
        mask_score = float(probability.mean())
        decision = "positive" if mask_coverage >= threshold else "negative"

        target_dir = positive_dir if decision == "positive" else negative_dir
        copied_path = _copy_unique(image_path, target_dir / image_path.name)
        overlay_path = ""
        if include_overlays and decision == "positive":
            overlay_path = str(_save_overlay(image, mask, overlay_dir / image_path.name))

        rows.append(
            {
                "input_path": str(image_path),
                "decision": decision,
                "mask_coverage": round(mask_coverage, 6),
                "mask_score": round(mask_score, 6),
                "output_path": str(copied_path),
                "overlay_path": overlay_path,
            }
        )

    summary = {
        "model": "CrossMaskNet v4",
        "model_root": str(model_root),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "positive_threshold": threshold,
        "device": device,
        "total": len(rows),
        "positive": sum(1 for row in rows if row["decision"] == "positive"),
        "negative": sum(1 for row in rows if row["decision"] == "negative"),
        "positive_dir": str(positive_dir),
        "negative_dir": str(negative_dir),
        "positive_overlays_dir": str(overlay_dir),
    }
    _write_prediction_files(output_dir, rows, summary)
    return {**summary, "predictions": rows}


def _parse_tile(raw: dict[str, Any]) -> CrossMaskTile:
    return CrossMaskTile(
        tile_id=str(raw["tile_id"]),
        scene_id=str(raw["scene_id"]),
        row=int(raw["row"]),
        col=int(raw["col"]),
        bbox_mercator=tuple(float(value) for value in raw["bbox_mercator"]),
        relative_path=str(raw.get("relative_path") or ""),
    )


def _scene_request(dataset_root: Path, scene_id: str) -> SceneRequest:
    scene = json.loads((dataset_root / "scenes" / scene_id / "scene.json").read_text(encoding="utf8"))
    return SceneRequest(
        scene_id=scene_id,
        latitude=float(scene["latitude"]),
        longitude=float(scene["longitude"]),
        size_m=int(scene["size_m"]),
        image_px=int(scene["image_px"]),
        tile_size_m=int(scene["tile_size_m"]),
    )


def _load_model(metrics: dict[str, Any], model_root: Path, device: str) -> CrossMaskNet:
    model = CrossMaskNet(base_channels=int(metrics["base_channels"]), input_channels=int(metrics.get("input_channels", 3))).to(device)
    model.load_state_dict(torch.load(model_root / "crossmasknet_best.pt", map_location=device))
    model.eval()
    return model


def _predict_probability(model: CrossMaskNet, image: Image.Image, image_size: int, device: str, *, road_channel: bool) -> np.ndarray:
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image_tensor = transform(image.convert("RGB"))
    if road_channel:
        road_mask = build_urban_road_density_mask(image.resize((image_size, image_size)), max_width=image_size, threshold=0.46).mask
        road_tensor = torch.from_numpy(road_mask.astype("float32")).unsqueeze(0)
        image_tensor = torch.cat([image_tensor, road_tensor], dim=0)
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        return torch.sigmoid(logits)[0, 0].detach().cpu().numpy()


def _iter_image_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _copy_unique(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = _unique_path(destination)
    shutil.copy2(source, target)
    return target


def _save_overlay(image: Image.Image, mask: np.ndarray, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = _unique_path(destination.with_suffix(".png"))
    mask_image = Image.fromarray(mask.astype("uint8") * 180).resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", image.size, (255, 32, 32, 0))
    overlay.putalpha(mask_image)
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    combined.convert("RGB").save(target)
    return target


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output path for {path}")


def _write_prediction_files(output_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({**summary, "predictions": rows}, indent=2), encoding="utf8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["input_path", "decision", "mask_coverage", "mask_score", "output_path", "overlay_path"])
        writer.writeheader()
        writer.writerows(rows)


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
