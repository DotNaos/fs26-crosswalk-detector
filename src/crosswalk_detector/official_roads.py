from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .urban_vision import _box_blur, _close, _component_masks, _dilate, _open

SWISSTOPO_WMS_URL = "https://wms.geo.admin.ch/"
SWISSTLM3D_ROADS_LAYER = "ch.swisstopo.swisstlm3d-strassen"


def _build_wms_url(bbox: tuple[float, float, float, float], image_px: int, layer: str) -> str:
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": layer,
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(str(value) for value in bbox),
            "WIDTH": image_px,
            "HEIGHT": image_px,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "LANG": "en",
        }
    )
    return f"{SWISSTOPO_WMS_URL}?{query}"


def load_official_roads_mosaic(cache_dir: str | Path, bbox: tuple[float, float, float, float], *, image_px: int = 1536) -> Image.Image:
    cache_path = Path(cache_dir) / f"swisstopo-roads-{image_px}.png"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGBA")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(_build_wms_url(bbox, image_px, SWISSTLM3D_ROADS_LAYER), timeout=30) as response:
        cache_path.write_bytes(response.read())
    return Image.open(cache_path).convert("RGBA")


def _official_reliable_road_masks(roads_image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixels = np.asarray(roads_image.convert("RGBA"), dtype=np.uint8)
    alpha = pixels[:, :, 3] > 8
    red = pixels[:, :, 0].astype(np.int16)
    green = pixels[:, :, 1].astype(np.int16)
    blue = pixels[:, :, 2].astype(np.int16)
    mean = (red + green + blue) / 3
    chroma = np.maximum.reduce((red, green, blue)) - np.minimum.reduce((red, green, blue))

    bright_local_roads = alpha & (mean > 150) & (chroma < 80)
    colored_major_roads = alpha & (chroma >= 80)
    return bright_local_roads | colored_major_roads, bright_local_roads, colored_major_roads


def urban_density_mask_from_roads(
    roads_image: Image.Image,
    *,
    surface_quantile: float = 0.64,
    minimum_threshold: float = 0.24,
) -> tuple[np.ndarray, np.ndarray, float]:
    reliable_roads, _bright_local_roads, colored_major_roads = _official_reliable_road_masks(roads_image)
    width = roads_image.width
    local_density = _box_blur(reliable_roads.astype(np.float32), radius=max(5, round(width / 52)))
    mid_density = _box_blur(reliable_roads.astype(np.float32), radius=max(8, round(width / 34)))
    broad_density = _box_blur(reliable_roads.astype(np.float32), radius=max(12, round(width / 22)))
    major_context = _box_blur(colored_major_roads.astype(np.float32), radius=max(8, round(width / 18)))
    density = np.clip(local_density * 0.72 + mid_density * 0.72 + broad_density * 0.34 + major_context * 0.28, 0, 1)

    nonzero = density[density > 0]
    threshold = max(minimum_threshold, float(np.quantile(nonzero, surface_quantile)) if nonzero.size else 1.0)
    seed_threshold = max(threshold, float(np.quantile(nonzero, 0.72)) if nonzero.size else threshold)
    mask = density >= threshold
    seed = density >= seed_threshold
    mask = _open(_close(mask, iterations=max(1, round(width / 180))), iterations=1)

    kept = np.zeros(mask.shape, dtype=bool)
    seed_context = _dilate(seed, iterations=max(2, round(width / 80)))
    min_area = max(96, round(mask.size * 0.003))
    for component in _component_masks(mask, min_area=min_area):
        if (component & seed_context).any():
            kept |= component
    return kept, density, threshold
