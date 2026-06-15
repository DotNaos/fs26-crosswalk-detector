"""Download and cache raw SWISSIMAGE scene imagery for metadata datasets."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import math
import ssl

import certifi
from PIL import Image

EARTH_RADIUS_M = 6_378_137.0
WMS_BASE_URL = "https://wms.geo.admin.ch/"
USER_AGENT = "crosswalk-detector-fs26/0.1"

SceneFetcher = Callable[[str], bytes]


def download_dataset_scenes(
    dataset_root: Path,
    *,
    raw_root: Path | None = None,
    limit_scenes: int | None = None,
    workers: int = 4,
    fetcher: SceneFetcher | None = None,
) -> dict[str, Any]:
    scene_ids = _dataset_scene_ids(dataset_root)
    if limit_scenes is not None:
        scene_ids = scene_ids[:limit_scenes]
    destination_root = _scene_root(dataset_root, raw_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    cached = 0
    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(_ensure_scene_file, dataset_root, scene_id, destination_root, fetcher): scene_id
            for scene_id in scene_ids
        }
        for future in as_completed(futures):
            path, was_downloaded = future.result()
            results.append(path)
            if was_downloaded:
                downloaded += 1
            else:
                cached += 1

    total_bytes = sum(path.stat().st_size for path in results if path.exists())
    return {
        "dataset_root": str(dataset_root),
        "raw_root": str(destination_root.parent),
        "scene_count": len(scene_ids),
        "downloaded": downloaded,
        "cached": cached,
        "bytes": total_bytes,
        "size_mb": round(total_bytes / 1024 / 1024, 2),
    }


def load_cached_scene_image(dataset_root: Path, scene: Any, *, raw_root: Path | None = None) -> Image.Image:
    destination_root = _scene_root(dataset_root, raw_root)
    path, _was_downloaded = _ensure_scene_file(dataset_root, str(scene.scene_id), destination_root, None, scene=scene)
    return Image.open(path).convert("RGB")


def estimate_scene_cache_size(sample_scene_bytes: list[int], scene_count: int) -> dict[str, Any]:
    if not sample_scene_bytes:
        raise ValueError("At least one sample size is required.")
    average = sum(sample_scene_bytes) / len(sample_scene_bytes)
    return {
        "sample_count": len(sample_scene_bytes),
        "scene_count": scene_count,
        "average_bytes": round(average),
        "estimated_bytes": round(average * scene_count),
        "estimated_mb": round((average * scene_count) / 1024 / 1024, 2),
    }


def _ensure_scene_file(
    dataset_root: Path,
    scene_id: str,
    destination_root: Path,
    fetcher: SceneFetcher | None,
    *,
    scene: Any | None = None,
) -> tuple[Path, bool]:
    destination = destination_root / f"{scene_id}.jpg"
    if destination.exists() and destination.stat().st_size > 0:
        return destination, False

    if scene is None:
        scene = _read_scene(dataset_root, scene_id)
    url = _wms_url(scene)
    payload = (fetcher or _fetch_url)(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(".jpg.tmp")
    tmp_path.write_bytes(payload)
    tmp_path.replace(destination)
    return destination, True


def _dataset_scene_ids(dataset_root: Path) -> list[str]:
    index = _read_index(dataset_root)
    seen: set[str] = set()
    scene_ids: list[str] = []
    for scene in index.get("scenes", []):
        scene_id = str(scene["scene_id"])
        if scene_id not in seen:
            scene_ids.append(scene_id)
            seen.add(scene_id)
    for shard in index.get("shards", []):
        scene_id = str(shard["scene_id"])
        if scene_id not in seen:
            scene_ids.append(scene_id)
            seen.add(scene_id)
    return scene_ids


def _scene_root(dataset_root: Path, raw_root: Path | None) -> Path:
    if raw_root is not None:
        return raw_root / "wms-mosaics"
    dataset_id = _read_index(dataset_root)["dataset_id"]
    return Path("data") / "raw" / str(dataset_id) / "wms-mosaics"


def _read_index(dataset_root: Path) -> dict[str, Any]:
    return json.loads((dataset_root / "dataset.json").read_text(encoding="utf8"))


def _read_scene(dataset_root: Path, scene_id: str) -> dict[str, Any]:
    scene_path = dataset_root / "scenes" / scene_id / "scene.json"
    if scene_path.exists():
        return json.loads(scene_path.read_text(encoding="utf8"))

    index = _read_index(dataset_root)
    for scene in index.get("scenes", []):
        if scene.get("scene_id") == scene_id:
            return scene
    for shard in index.get("shards", []):
        if shard.get("scene_id") == scene_id:
            return shard
    raise FileNotFoundError(f"No scene metadata found for {scene_id} in {dataset_root}")


def _scene_bbox(scene: Any) -> tuple[float, float, float, float]:
    if isinstance(scene, dict) and "bbox_mercator" in scene:
        return tuple(float(value) for value in scene["bbox_mercator"])

    latitude = _scene_value(scene, "latitude")
    longitude = _scene_value(scene, "longitude")
    size_m = _scene_value(scene, "size_m")
    center_x, center_y = _mercator_from_lat_lon(latitude, longitude)
    half = size_m / 2.0
    return center_x - half, center_y - half, center_x + half, center_y + half


def _wms_url(scene: Any) -> str:
    image_px = int(_scene_value(scene, "image_px", default=2048))
    bbox = _scene_bbox(scene)
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": "ch.swisstopo.swissimage-product",
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(str(value) for value in bbox),
            "WIDTH": image_px,
            "HEIGHT": image_px,
            "FORMAT": "image/jpeg",
        }
    )
    return f"{WMS_BASE_URL}?{query}"


def _scene_value(scene: Any, key: str, *, default: float | None = None) -> float:
    if isinstance(scene, dict):
        if key not in scene and default is not None:
            return default
        return float(scene[key])
    if not hasattr(scene, key) and default is not None:
        return default
    return float(getattr(scene, key))


def _mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(longitude)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0))
    return x, y


def _fetch_url(url: str) -> bytes:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), context=ssl_context, timeout=60) as response:
        return response.read()
