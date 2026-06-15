from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .official_roads import (
    SWISSTLM3D_ROADS_LAYER,
    SWISSTOPO_WMS_URL,
    _official_reliable_road_masks,
    urban_density_mask_from_roads,
)
from .urban_vision import _box_blur

WEB_MERCATOR_WORLD_MIN = -20037508.342789244
DEFAULT_MAX_IMAGE_PX = 256
DEFAULT_MAX_GRID_CELLS = 30000


def _build_wms_url(bbox: tuple[float, float, float, float], width: int, height: int) -> str:
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": SWISSTLM3D_ROADS_LAYER,
            "STYLES": "default",
            "CRS": "EPSG:3857",
            "BBOX": ",".join(str(value) for value in bbox),
            "WIDTH": width,
            "HEIGHT": height,
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "LANG": "en",
        }
    )
    return f"{SWISSTOPO_WMS_URL}?{query}"


def _image_size_for_bbox(bbox: tuple[float, float, float, float], max_px: int) -> tuple[int, int]:
    width_m = max(1.0, bbox[2] - bbox[0])
    height_m = max(1.0, bbox[3] - bbox[1])
    scale = max_px / max(width_m, height_m)
    return max(256, round(width_m * scale)), max(256, round(height_m * scale))


def _max_image_px_for_zoom(zoom: float, requested_max_px: int) -> int:
    if zoom >= 15:
        return max(requested_max_px, 1024)
    if zoom >= 13:
        return max(requested_max_px, 768)
    if zoom >= 10:
        return max(requested_max_px, 512)
    return requested_max_px


def _load_roads(cache_dir: Path, bbox: tuple[float, float, float, float], width: int, height: int) -> Image.Image:
    key = hashlib.sha1(f"{bbox}:{width}:{height}:{SWISSTLM3D_ROADS_LAYER}".encode("utf-8")).hexdigest()[:20]
    path = cache_dir / f"roads-visible-{key}.png"
    if path.exists():
        return Image.open(path).convert("RGBA")

    cache_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(_build_wms_url(bbox, width, height), timeout=30) as response:
        path.write_bytes(response.read())
    return Image.open(path).convert("RGBA")


def _cell_size_for_zoom(zoom: float) -> int:
    if zoom >= 15.75:
        return 25
    if zoom >= 14.75:
        return 50
    if zoom >= 13.5:
        return 100
    if zoom >= 12:
        return 200
    if zoom >= 10.5:
        return 400
    if zoom >= 9:
        return 800
    return 6400


def _grid_bounds(bbox: tuple[float, float, float, float], cell_size_m: int) -> tuple[int, int, int, int]:
    min_x, min_y, max_x, max_y = bbox
    col0 = math.floor((min_x - WEB_MERCATOR_WORLD_MIN) / cell_size_m)
    col1 = math.ceil((max_x - WEB_MERCATOR_WORLD_MIN) / cell_size_m)
    row0 = math.floor((min_y - WEB_MERCATOR_WORLD_MIN) / cell_size_m)
    row1 = math.ceil((max_y - WEB_MERCATOR_WORLD_MIN) / cell_size_m)
    return row0, row1, col0, col1


def _bounded_cell_size(bbox: tuple[float, float, float, float], zoom: float, max_cells: int) -> int:
    cell_size = _cell_size_for_zoom(zoom)
    while True:
        row0, row1, col0, col1 = _grid_bounds(bbox, cell_size)
        if (row1 - row0) * (col1 - col0) <= max_cells or cell_size >= 6400:
            return cell_size
        cell_size *= 2


def _cluster_requirements(cell_size_m: int, surface_threshold: float) -> tuple[float, float, float, float]:
    if cell_size_m <= 25:
        return 0.6, 0.18, 0.48, surface_threshold * 0.95
    if cell_size_m <= 50:
        return 0.42, 0.12, 0.32, surface_threshold * 0.86
    if cell_size_m <= 100:
        return 0.3, 0.08, 0.24, surface_threshold * 0.78
    return 0.22, 0.05, 0.18, surface_threshold * 0.68


def _keep_cluster_cell(
    *,
    surface_ratio: float,
    density_score: float,
    road_pixel_ratio: float,
    line_density_score: float,
    cell_size_m: int,
    surface_threshold: float,
) -> bool:
    min_surface_ratio, min_road_pixel_ratio, min_line_density, min_density_score = _cluster_requirements(cell_size_m, surface_threshold)
    city_candidate = surface_ratio >= min_surface_ratio * 0.45 or density_score >= min_density_score * 1.05
    road_inside_dense_surface = surface_ratio >= min_surface_ratio and road_pixel_ratio >= min_road_pixel_ratio
    line_density_cluster = line_density_score >= min_line_density and density_score >= min_density_score
    strong_road_cell = road_pixel_ratio >= min_road_pixel_ratio * 1.6 and density_score >= min_density_score * 1.15
    return city_candidate and (road_inside_dense_surface or line_density_cluster or strong_road_cell)


def _integral(values: np.ndarray) -> np.ndarray:
    return np.pad(values.astype(np.float32), ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)


def _region_sum(integral: np.ndarray, left: int, top: int, right: int, bottom: int) -> float:
    return float(integral[bottom, right] - integral[top, right] - integral[bottom, left] + integral[top, left])


def build_road_cluster_grid(
    bbox: tuple[float, float, float, float],
    *,
    zoom: float,
    cache_dir: str | Path = "data/cache/road-cluster-grid",
    max_image_px: int = DEFAULT_MAX_IMAGE_PX,
    max_grid_cells: int = DEFAULT_MAX_GRID_CELLS,
) -> dict:
    max_image_px = _max_image_px_for_zoom(zoom, max_image_px)
    width_px, height_px = _image_size_for_bbox(bbox, max_image_px)
    roads = _load_roads(Path(cache_dir), bbox, width_px, height_px)
    surface, density, threshold = urban_density_mask_from_roads(roads)
    reliable_roads, _bright_local_roads, _colored_major_roads = _official_reliable_road_masks(roads)
    local_line_density = _box_blur(reliable_roads.astype(np.float32), radius=max(2, round(width_px / 96)))
    cell_size_m = _bounded_cell_size(bbox, zoom, max_grid_cells)
    row0, row1, col0, col1 = _grid_bounds(bbox, cell_size_m)
    mask_integral = _integral(surface)
    density_integral = _integral(density)
    road_integral = _integral(reliable_roads.astype(np.float32))
    line_density_integral = _integral(local_line_density)
    bbox_width = max(1.0, bbox[2] - bbox[0])
    bbox_height = max(1.0, bbox[3] - bbox[1])
    cells = []

    for row in range(row0, row1):
        y0 = WEB_MERCATOR_WORLD_MIN + row * cell_size_m
        y1 = y0 + cell_size_m
        for col in range(col0, col1):
            x0 = WEB_MERCATOR_WORLD_MIN + col * cell_size_m
            x1 = x0 + cell_size_m
            left = max(0, min(width_px, math.floor((x0 - bbox[0]) / bbox_width * width_px)))
            right = max(0, min(width_px, math.ceil((x1 - bbox[0]) / bbox_width * width_px)))
            top = max(0, min(height_px, math.floor((bbox[3] - y1) / bbox_height * height_px)))
            bottom = max(0, min(height_px, math.ceil((bbox[3] - y0) / bbox_height * height_px)))
            pixel_count = max(0, right - left) * max(0, bottom - top)
            if pixel_count <= 0:
                continue
            surface_ratio = _region_sum(mask_integral, left, top, right, bottom) / pixel_count
            density_score = _region_sum(density_integral, left, top, right, bottom) / pixel_count
            road_pixel_ratio = _region_sum(road_integral, left, top, right, bottom) / pixel_count
            line_density_score = _region_sum(line_density_integral, left, top, right, bottom) / pixel_count
            if not _keep_cluster_cell(
                surface_ratio=surface_ratio,
                density_score=density_score,
                road_pixel_ratio=road_pixel_ratio,
                line_density_score=line_density_score,
                cell_size_m=cell_size_m,
                surface_threshold=threshold,
            ):
                continue
            cluster_score = max(
                surface_ratio,
                density_score / max(threshold, 1e-6),
                line_density_score,
                road_pixel_ratio,
            )
            cells.append(
                {
                    "id": f"road-r{row}-c{col}",
                    "row": row,
                    "col": col,
                    "sizeM": cell_size_m,
                    "surfaceRatio": surface_ratio,
                    "densityScore": density_score,
                    "roadPixelRatio": road_pixel_ratio,
                    "lineDensityScore": line_density_score,
                    "clusterScore": cluster_score,
                    "bboxMercator": [x0, y0, x1, y1],
                }
            )

    return {
        "sourceLayer": SWISSTLM3D_ROADS_LAYER,
        "method": "visible-road-density",
        "zoom": zoom,
        "cellSizeM": cell_size_m,
        "bboxMercator": list(bbox),
        "surfaceThreshold": threshold,
        "surfaceCoverage": float(surface.mean()),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", required=True)
    parser.add_argument("--zoom", type=float, required=True)
    parser.add_argument("--cache-dir", default="data/cache/road-cluster-grid")
    args = parser.parse_args()
    bbox = tuple(float(value) for value in args.bbox.split(","))
    if len(bbox) != 4:
        raise SystemExit("--bbox must contain minX,minY,maxX,maxY")
    print(json.dumps(build_road_cluster_grid(bbox, zoom=args.zoom, cache_dir=args.cache_dir)))


if __name__ == "__main__":
    main()
