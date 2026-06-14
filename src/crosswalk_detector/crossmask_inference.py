"""Run CrossMaskNet on ad-hoc metadata tiles."""

from __future__ import annotations

from dataclasses import dataclass
import json
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
        "threshold": threshold,
        "device": device,
        "summary": summary,
        "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf8")
    return result


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


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
