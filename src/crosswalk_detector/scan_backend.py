"""Server-side scan backend for scene-level crosswalk labeling."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel

EARTH_RADIUS_M = 6_378_137.0
WMS_BASE_URL = "https://wms.geo.admin.ch/"
DETECTOR_MODEL_ID = os.getenv("CROSSWALK_DETECTOR_MODEL", "IDEA-Research/grounding-dino-tiny")
CLIP_MODEL_ID = os.getenv("CROSSWALK_CLIP_MODEL", "openai/clip-vit-base-patch32")
POSITIVE_PROMPTS = (
    "an aerial photo of a zebra crossing painted across a road",
    "an aerial photo of a pedestrian crosswalk with parallel road stripes",
)
NEGATIVE_PROMPTS = (
    "an aerial photo of a road without a crosswalk",
    "an aerial photo of a street intersection without a crosswalk",
    "an aerial photo of a building rooftop",
    "an aerial photo of a railway platform or tram stop",
    "an aerial photo of railway tracks",
    "an aerial photo of a parking lot",
)


@dataclass(frozen=True)
class SceneRequest:
    scene_id: str
    latitude: float
    longitude: float
    size_m: int
    image_px: int
    tile_size_m: int


@dataclass(frozen=True)
class TileRequest:
    tile_id: str
    row: int
    col: int
    bbox_mercator: tuple[float, float, float, float]
    relative_path: str


@dataclass(frozen=True)
class TileMetrics:
    clip_positive: float
    clip_negative: float
    heuristic_probability: float
    detector_score: float
    combined_score: float
    label: str


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(longitude)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return x, y


def mercator_to_lat_lon(x: float, y: float) -> tuple[float, float]:
    longitude = math.degrees(x / EARTH_RADIUS_M)
    latitude = math.degrees(2.0 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2.0)
    return latitude, longitude


def _scene_bbox(scene: SceneRequest) -> tuple[float, float, float, float]:
    center_x, center_y = _mercator_from_lat_lon(scene.latitude, scene.longitude)
    half = scene.size_m / 2.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def _build_wms_url(scene: SceneRequest) -> str:
    bbox = ",".join(str(value) for value in _scene_bbox(scene))
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


def fetch_scene_image(scene: SceneRequest) -> Image.Image:
    with urlopen(_build_wms_url(scene)) as response:
        return Image.open(BytesIO(response.read())).convert("RGB")


def _tile_pixel_bounds(scene: SceneRequest, bbox_mercator: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    scene_min_x, scene_min_y, scene_max_x, scene_max_y = _scene_bbox(scene)
    tile_min_x, tile_min_y, tile_max_x, tile_max_y = bbox_mercator
    scale_x = scene.image_px / (scene_max_x - scene_min_x)
    scale_y = scene.image_px / (scene_max_y - scene_min_y)
    left = (tile_min_x - scene_min_x) * scale_x
    right = (tile_max_x - scene_min_x) * scale_x
    top = (scene_max_y - tile_max_y) * scale_y
    bottom = (scene_max_y - tile_min_y) * scale_y
    return left, top, right, bottom


def crop_tile(image: Image.Image, scene: SceneRequest, tile: TileRequest, padding_tiles: float = 0.0) -> Image.Image:
    left, top, right, bottom = _tile_pixel_bounds(scene, tile.bbox_mercator)
    tile_px = scene.image_px / max(scene.size_m / scene.tile_size_m, 1)
    pad = tile_px * padding_tiles
    crop_box = (
        max(0, int(round(left - pad))),
        max(0, int(round(top - pad))),
        min(image.width, int(round(right + pad))),
        min(image.height, int(round(bottom + pad))),
    )
    return image.crop(crop_box).convert("RGB")


def crosswalk_score_image(image: Image.Image) -> float:
    hsv = np.array(image.convert("HSV"))
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = ((h >= 18) & (h <= 55) & (s >= 40) & (v >= 100)).astype(np.uint8)

    seen = np.zeros_like(yellow, dtype=bool)
    component_count = medium_components = medium_area = elongated_components = large_components = 0
    height, width = yellow.shape
    stack: list[tuple[int, int]] = []
    for y in range(height):
        for x in range(width):
            if yellow[y, x] == 0 or seen[y, x]:
                continue
            component_count += 1
            stack.append((y, x))
            seen[y, x] = True
            size = 0
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                cy, cx = stack.pop()
                size += 1
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and yellow[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            box_width = max_x - min_x + 1
            box_height = max_y - min_y + 1
            aspect_ratio = max(box_width, box_height) / max(1, min(box_width, box_height))
            if size > 800:
                large_components += 1
            if 35 <= size <= 450 and aspect_ratio >= 1.35:
                medium_components += 1
                medium_area += size
            if 20 <= size <= 450 and aspect_ratio >= 2.3:
                elongated_components += 1

    row_profile = yellow.mean(axis=1)
    col_profile = yellow.mean(axis=0)
    yellow_density = float(yellow.mean())
    stripe_axis_std = float(max(row_profile.std(), col_profile.std()))
    stripe_axis_peak = float(max(np.sort(row_profile)[-6:].sum(), np.sort(col_profile)[-6:].sum()))
    return round(
        (medium_components * 3.5)
        + (medium_area / 220.0)
        + (elongated_components * 2.5)
        + (stripe_axis_std * 60.0)
        + (stripe_axis_peak * 8.0)
        - (max(0.0, yellow_density - 0.12) * 180.0)
        - (large_components * 4.0),
        4,
    )


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class HybridCrosswalkScanner:
    def __init__(self) -> None:
        self.device = _device()
        token = os.getenv("HF_TOKEN")
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID, token=token)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, token=token).to(self.device)
        self.clip_model.eval()
        self.detector_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID, token=token)
        self.detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL_ID, token=token).to(self.device)
        self.detector_model.eval()

    def detect_boxes(self, scene_image: Image.Image) -> list[tuple[float, tuple[float, float, float, float]]]:
        text = "zebra crossing. crosswalk. pedestrian crossing."
        inputs = self.detector_processor(images=scene_image, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.detector_model(**inputs)
        results = self.detector_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.16,
            text_threshold=0.16,
            target_sizes=[scene_image.size[::-1]],
        )[0]
        detections: list[tuple[float, tuple[float, float, float, float]]] = []
        for score, box in zip(results["scores"], results["boxes"]):
            detections.append((float(score), tuple(float(value) for value in box)))
        return detections

    def score_context_images(self, images: list[Image.Image]) -> list[tuple[float, float]]:
        texts = list(POSITIVE_PROMPTS + NEGATIVE_PROMPTS)
        positive_count = len(POSITIVE_PROMPTS)
        inputs = self.clip_processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.clip_model(**inputs).logits_per_image
            positive_logits = logits[:, :positive_count].mean(dim=1)
            negative_logits = logits[:, positive_count:].mean(dim=1)
            probs = torch.softmax(torch.stack([positive_logits, negative_logits], dim=1), dim=1)
        return [(float(prob[0]), float(prob[1])) for prob in probs.cpu().tolist()]


def _center_inside(box: tuple[float, float, float, float], tile_bounds: tuple[float, float, float, float]) -> bool:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return tile_bounds[0] <= cx <= tile_bounds[2] and tile_bounds[1] <= cy <= tile_bounds[3]


def _intersection_ratio(box: tuple[float, float, float, float], tile_bounds: tuple[float, float, float, float]) -> float:
    left = max(box[0], tile_bounds[0])
    top = max(box[1], tile_bounds[1])
    right = min(box[2], tile_bounds[2])
    bottom = min(box[3], tile_bounds[3])
    if right <= left or bottom <= top:
      return 0.0
    overlap = (right - left) * (bottom - top)
    tile_area = max(1.0, (tile_bounds[2] - tile_bounds[0]) * (tile_bounds[3] - tile_bounds[1]))
    return overlap / tile_area


def detector_overlap_score(detections: Iterable[tuple[float, tuple[float, float, float, float]]], tile_bounds: tuple[float, float, float, float]) -> float:
    best = 0.0
    for score, box in detections:
      ratio = _intersection_ratio(box, tile_bounds)
      if _center_inside(box, tile_bounds):
        best = max(best, score)
      elif ratio >= 0.02:
        best = max(best, score * min(1.0, ratio * 8.0))
    return best


def decide_tile_label(clip_positive: float, clip_negative: float, heuristic_probability: float, detector_score: float, threshold: float) -> TileMetrics:
    combined = (clip_positive * 0.45) + (heuristic_probability * 0.2) + (detector_score * 0.35)
    positive_margin = clip_positive - clip_negative
    positive = (
        detector_score >= 0.24
        or (combined >= threshold and positive_margin >= 0.03)
        or (heuristic_probability >= 0.58 and clip_positive >= max(0.18, threshold - 0.08))
    )
    return TileMetrics(
        clip_positive=clip_positive,
        clip_negative=clip_negative,
        heuristic_probability=heuristic_probability,
        detector_score=detector_score,
        combined_score=combined,
        label="crosswalk" if positive else "no_crosswalk",
    )
