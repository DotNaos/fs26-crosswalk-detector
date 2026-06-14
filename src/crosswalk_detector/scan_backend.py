"""Server-side scan backend for scene-level crosswalk labeling."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
import os
from pathlib import Path
import ssl
from typing import Any, Iterable, Protocol
from urllib.parse import urlencode
from urllib.request import urlopen

import certifi
import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPModel

EARTH_RADIUS_M = 6_378_137.0
WMS_BASE_URL = "https://wms.geo.admin.ch/"
DETECTOR_MODEL_ID = os.getenv("CROSSWALK_DETECTOR_MODEL", "IDEA-Research/grounding-dino-tiny")
CLIP_MODEL_ID = os.getenv("CROSSWALK_CLIP_MODEL", "openai/clip-vit-base-patch32")
DEFAULT_SUPERVISED_CLIP_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "real-v1" / "real-balanced-256" / "clip_linear.pt"
SUPERVISED_CLIP_MODEL_PATH = Path(os.getenv("CROSSWALK_SUPERVISED_CLIP_MODEL", str(DEFAULT_SUPERVISED_CLIP_MODEL_PATH)))
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
SAM31_MODEL_VERSION = "sam3.1"
SAM31_PROMPTS = tuple(
    prompt.strip()
    for prompt in os.getenv(
        "CROSSWALK_SAM31_PROMPTS",
        "zebra crossing|pedestrian crosswalk|yellow zebra crossing",
    ).split("|")
    if prompt.strip()
)
SAM31_NEGATIVE_PROMPTS = tuple(
    prompt.strip()
    for prompt in os.getenv(
        "CROSSWALK_SAM31_NEGATIVE_PROMPTS",
        "",
    ).split("|")
    if prompt.strip()
)
SAM31_CONFIDENCE_THRESHOLD = float(os.getenv("CROSSWALK_SAM31_CONFIDENCE", "0.18"))
SAM31_TILE_THRESHOLD = float(os.getenv("CROSSWALK_SAM31_TILE_THRESHOLD", "0.18"))
SAM31_NEGATIVE_TILE_THRESHOLD = float(os.getenv("CROSSWALK_SAM31_NEGATIVE_TILE_THRESHOLD", "0.18"))
SAM31_EDGE_CENTER_THRESHOLD = float(os.getenv("CROSSWALK_SAM31_EDGE_CENTER_THRESHOLD", "0.45"))
SAM31_PARKING_GRID_MAX_SCORE = float(os.getenv("CROSSWALK_SAM31_PARKING_GRID_MAX_SCORE", "0.60"))
SAM31_SUPPRESSION_BOX_IOU = float(os.getenv("CROSSWALK_SAM31_SUPPRESSION_BOX_IOU", "0.20"))
SAM31_MASK_COVERAGE_SCALE = float(os.getenv("CROSSWALK_SAM31_MASK_COVERAGE_SCALE", "120"))
SAM31_BOX_OVERLAP_SCALE = float(os.getenv("CROSSWALK_SAM31_BOX_OVERLAP_SCALE", "10"))


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
    supervised_probability: float | None = None


@dataclass(frozen=True)
class Sam31Detection:
    prompt: str
    score: float
    box: tuple[float, float, float, float]
    mask: np.ndarray
    polarity: str = "positive"


@dataclass(frozen=True)
class Sam31TileMetrics:
    label: str
    score: float
    peak: float
    coverage: float
    box_overlap: float
    prompt: str
    detection_count: int
    mask: np.ndarray | None = None


class ScanBackend(Protocol):
    device: str
    backend_name: str


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
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(_build_wms_url(scene), context=ssl_context) as response:
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


def _cast_floating_input_to_module_dtype(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if not args or not torch.is_tensor(args[0]) or not hasattr(module, "weight"):
        return args
    input_tensor = args[0]
    weight = module.weight
    if input_tensor.is_floating_point() and input_tensor.dtype != weight.dtype:
        return (input_tensor.to(weight.dtype), *args[1:])
    return args


class Sam31CrosswalkScanner:
    backend_name = "sam31"

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompts = SAM31_PROMPTS
        self.negative_prompts = SAM31_NEGATIVE_PROMPTS
        self.confidence_threshold = SAM31_CONFIDENCE_THRESHOLD
        self.tile_threshold = SAM31_TILE_THRESHOLD
        self.negative_tile_threshold = SAM31_NEGATIVE_TILE_THRESHOLD
        self.edge_center_threshold = SAM31_EDGE_CENTER_THRESHOLD
        self.parking_grid_max_score = SAM31_PARKING_GRID_MAX_SCORE
        if self.device != "cuda":
            raise RuntimeError("SAM3.1 scanning requires CUDA on the remote scan host.")
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model, download_ckpt_from_hf

        checkpoint_path = download_ckpt_from_hf(version=SAM31_MODEL_VERSION)
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=self.device,
            compile=False,
        )
        self._register_dtype_guards(model)
        self.processor = Sam3Processor(
            model,
            resolution=1008,
            device=self.device,
            confidence_threshold=self.confidence_threshold,
        )

    def _register_dtype_guards(self, model: nn.Module) -> None:
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LayerNorm)):
                module.register_forward_pre_hook(_cast_floating_input_to_module_dtype)

    def detect_scene(self, scene_image: Image.Image) -> list[Sam31Detection]:
        state = self.processor.set_image(scene_image)
        detections: list[Sam31Detection] = []
        for prompt in self.prompts:
            detections.extend(self._detect_prompt(state, prompt, "positive"))
        for prompt in self.negative_prompts:
            detections.extend(self._detect_prompt(state, prompt, "negative"))
        return detections

    def _detect_prompt(self, state: dict[str, Any], prompt: str, polarity: str) -> list[Sam31Detection]:
        detections: list[Sam31Detection] = []
        try:
            if hasattr(self.processor, "reset_all_prompts"):
                self.processor.reset_all_prompts(state)
            prompt_state = self.processor.set_text_prompt(prompt, state)
            boxes = prompt_state.get("boxes")
            masks = prompt_state.get("masks")
            scores = prompt_state.get("scores")
            if boxes is None or masks is None or scores is None:
                return detections
            boxes_np = boxes.detach().float().cpu().numpy()
            masks_np = masks.detach().cpu().numpy()
            scores_np = scores.detach().float().cpu().numpy()
            for box, mask, score in zip(boxes_np, masks_np, scores_np):
                mask_2d = np.asarray(mask).squeeze().astype(bool)
                detections.append(
                    Sam31Detection(
                        prompt=prompt,
                        score=float(score),
                        box=tuple(float(value) for value in box),
                        mask=mask_2d,
                        polarity=polarity,
                    )
                )
            return detections
        finally:
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def score_tiles(self, scene: SceneRequest, tiles: list[TileRequest], detections: list[Sam31Detection]) -> list[Sam31TileMetrics]:
        return [self.score_tile(scene, tile, detections) for tile in tiles]

    def score_tile(self, scene: SceneRequest, tile: TileRequest, detections: list[Sam31Detection]) -> Sam31TileMetrics:
        tile_bounds = _tile_pixel_bounds(scene, tile.bbox_mercator)
        best_score = best_peak = best_coverage = best_box_overlap = 0.0
        best_prompt = "sam3.1"
        best_mask: np.ndarray | None = None
        suppressed_prompt = None
        positive_detections = [detection for detection in detections if detection.polarity == "positive"]
        negative_detections = [detection for detection in detections if detection.polarity == "negative"]
        for detection in positive_detections:
            coverage = _mask_tile_coverage(detection.mask, tile_bounds)
            box_overlap = _intersection_ratio(detection.box, tile_bounds)
            peak = _sam31_tile_peak(coverage, box_overlap)
            score = detection.score * min(1.0, peak)
            if score < self.edge_center_threshold and not _center_inside(detection.box, tile_bounds):
                suppressed_prompt = "edge-overlap"
                continue
            suppressor = self._suppression_prompt(detection, tile_bounds, negative_detections)
            if suppressor is not None:
                suppressed_prompt = suppressor
                continue
            if score > best_score:
                best_score = score
                best_peak = detection.score
                best_coverage = coverage
                best_box_overlap = box_overlap
                best_prompt = detection.prompt
                best_mask = _crop_mask_to_tile(detection.mask, tile_bounds)
        label = "crosswalk" if best_score >= self.tile_threshold else "no_crosswalk"
        if best_score == 0.0 and suppressed_prompt is not None:
            best_prompt = f"suppressed:{suppressed_prompt}"
        if label != "crosswalk":
            best_mask = None
        return Sam31TileMetrics(
            label=label,
            score=best_score,
            peak=best_peak,
            coverage=best_coverage,
            box_overlap=best_box_overlap,
            prompt=best_prompt,
            detection_count=len(positive_detections),
            mask=best_mask,
        )

    def image_rejection_reason(self, tile_image: Image.Image, metrics: Sam31TileMetrics) -> str | None:
        if metrics.label != "crosswalk" or metrics.score >= self.parking_grid_max_score:
            return None
        if _looks_like_yellow_parking_grid(tile_image):
            return "yellow-parking-grid"
        return None

    def _suppression_prompt(
        self,
        candidate: Sam31Detection,
        tile_bounds: tuple[float, float, float, float],
        negative_detections: list[Sam31Detection],
    ) -> str | None:
        for detection in negative_detections:
            coverage = _mask_tile_coverage(detection.mask, tile_bounds)
            box_overlap = _intersection_ratio(detection.box, tile_bounds)
            peak = _sam31_tile_peak(coverage, box_overlap)
            score = detection.score * min(1.0, peak)
            if score < self.negative_tile_threshold:
                continue
            if _boxes_match(candidate.box, detection.box):
                return detection.prompt
        return None


class HybridCrosswalkScanner:
    backend_name = "hybrid"

    def __init__(self) -> None:
        self.device = _device()
        token = os.getenv("HF_TOKEN")
        self.clip_processor = AutoProcessor.from_pretrained(CLIP_MODEL_ID, token=token)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID, token=token).to(self.device)
        self.clip_model.eval()
        self.detector_processor = None
        self.detector_model = None
        self.supervised_threshold = 0.5
        self.supervised_clip_classifier = self._load_supervised_clip_classifier()

    @property
    def has_supervised_classifier(self) -> bool:
        return self.supervised_clip_classifier is not None

    def _load_supervised_clip_classifier(self) -> nn.Module | None:
        if not SUPERVISED_CLIP_MODEL_PATH.exists():
            return None
        payload = torch.load(SUPERVISED_CLIP_MODEL_PATH, map_location="cpu")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            return None
        classifier = torch.nn.Linear(512, 1)
        classifier.load_state_dict(state_dict)
        classifier.to(self.device)
        classifier.eval()
        self.supervised_threshold = float(payload.get("threshold", 0.5))
        return classifier

    def _ensure_detector(self) -> None:
        if self.detector_processor is not None and self.detector_model is not None:
            return
        token = os.getenv("HF_TOKEN")
        self.detector_processor = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID, token=token)
        self.detector_model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_MODEL_ID, token=token).to(self.device)
        self.detector_model.eval()

    def _clip_image_features(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.clip_processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            vision_output = self.clip_model.vision_model(**inputs)
            features = self.clip_model.visual_projection(vision_output.pooler_output)
            return torch.nn.functional.normalize(features, dim=1)

    def detect_boxes(self, scene_image: Image.Image) -> list[tuple[float, tuple[float, float, float, float]]]:
        self._ensure_detector()
        assert self.detector_processor is not None
        assert self.detector_model is not None
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

    def score_supervised_context_images(self, images: list[Image.Image]) -> list[float]:
        if self.supervised_clip_classifier is None:
            return []
        features = self._clip_image_features(images)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.supervised_clip_classifier(features).squeeze(1))
        return [float(value) for value in probabilities.cpu().tolist()]

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


def _box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    first_area = max(1.0, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return overlap / (first_area + second_area - overlap)


def _boxes_match(candidate: tuple[float, float, float, float], suppressor: tuple[float, float, float, float]) -> bool:
    return (
        _box_iou(candidate, suppressor) >= SAM31_SUPPRESSION_BOX_IOU
        or _center_inside(candidate, suppressor)
        or _center_inside(suppressor, candidate)
    )


def _sam31_tile_peak(coverage: float, box_overlap: float) -> float:
    return max(coverage * SAM31_MASK_COVERAGE_SCALE, box_overlap * SAM31_BOX_OVERLAP_SCALE)


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


def _mask_tile_coverage(mask: np.ndarray, tile_bounds: tuple[float, float, float, float]) -> float:
    tile_mask = _crop_mask_to_tile(mask, tile_bounds)
    return float(tile_mask.mean()) if tile_mask is not None and tile_mask.size else 0.0


def _crop_mask_to_tile(mask: np.ndarray, tile_bounds: tuple[float, float, float, float]) -> np.ndarray | None:
    height, width = mask.shape
    left = max(0, min(width, int(math.floor(tile_bounds[0]))))
    top = max(0, min(height, int(math.floor(tile_bounds[1]))))
    right = max(0, min(width, int(math.ceil(tile_bounds[2]))))
    bottom = max(0, min(height, int(math.ceil(tile_bounds[3]))))
    if right <= left or bottom <= top:
        return None
    return mask[top:bottom, left:right].astype(bool)


def _looks_like_yellow_parking_grid(image: Image.Image) -> bool:
    pixels = np.asarray(image.convert("RGB")).astype(np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    yellow = (red > 105) & (green > 95) & (blue < 115) & ((red - blue) > 35) & ((green - blue) > 25) & (np.abs(red - green) < 70)
    yellow_ratio = float(yellow.mean())
    if yellow_ratio < 0.12:
        return False
    components = _connected_component_boxes(yellow)
    large_components = [component for component in components if component[0] >= 8]
    largest_component = max((component[0] for component in components), default=0)
    return largest_component >= 500 or (len(large_components) >= 8 and largest_component >= 100)


def _connected_component_boxes(mask: np.ndarray) -> list[tuple[int, int, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[int, int, int]] = []
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        count = 0
        min_x = max_x = start_x
        min_y = max_y = start_y
        while stack:
            y, x = stack.pop()
            count += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for next_y in range(max(0, y - 1), min(height, y + 2)):
                for next_x in range(max(0, x - 1), min(width, x + 2)):
                    if mask[next_y, next_x] and not seen[next_y, next_x]:
                        seen[next_y, next_x] = True
                        stack.append((next_y, next_x))
        components.append((count, max_x - min_x + 1, max_y - min_y + 1))
    return components


def detector_overlap_score(detections: Iterable[tuple[float, tuple[float, float, float, float]]], tile_bounds: tuple[float, float, float, float]) -> float:
    best = 0.0
    for score, box in detections:
      ratio = _intersection_ratio(box, tile_bounds)
      if _center_inside(box, tile_bounds):
        best = max(best, score)
      elif ratio >= 0.02:
        best = max(best, score * min(1.0, ratio * 8.0))
    return best


def decide_tile_label(
    clip_positive: float,
    clip_negative: float,
    heuristic_probability: float,
    detector_score: float,
    threshold: float,
    supervised_probability: float | None = None,
) -> TileMetrics:
    combined = (
        supervised_probability
        if supervised_probability is not None
        else (clip_positive * 0.45) + (heuristic_probability * 0.2) + (detector_score * 0.35)
    )
    positive = combined >= threshold
    return TileMetrics(
        clip_positive=clip_positive,
        clip_negative=clip_negative,
        heuristic_probability=heuristic_probability,
        detector_score=detector_score,
        combined_score=combined,
        label="crosswalk" if positive else "no_crosswalk",
        supervised_probability=supervised_probability,
    )


def create_scan_backend() -> ScanBackend:
    backend = os.getenv("CROSSWALK_SCAN_BACKEND", "sam31").strip().lower()
    if backend in {"sam31", "sam3.1", "sam3"}:
        return Sam31CrosswalkScanner()
    if backend in {"hybrid", "clip", "clip-linear"}:
        return HybridCrosswalkScanner()
    raise ValueError(f"Unknown CROSSWALK_SCAN_BACKEND: {backend}")
