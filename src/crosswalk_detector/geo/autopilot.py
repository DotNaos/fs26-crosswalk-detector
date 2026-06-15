from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
from urllib.request import urlopen
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter, ImageStat

from .official_roads import SWISSTLM3D_ROADS_LAYER, load_official_roads_mosaic, urban_density_mask_from_roads


@dataclass(frozen=True)
class GeoReference:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class GridCell:
    row: int
    col: int
    bbox: tuple[float, float, float, float]
    urban_score: float
    is_urban: bool


@dataclass(frozen=True)
class UrbanCluster:
    cluster_id: str
    bbox: tuple[float, float, float, float]
    cell_count: int
    urban_score: float
    cells: tuple[GridCell, ...]


@dataclass(frozen=True)
class VisionGridPlan:
    rows: int
    cols: int
    cells: tuple[GridCell, ...]
    clusters: tuple[UrbanCluster, ...]


EARTH_RADIUS_M = 6_378_137
WEB_MERCATOR_WORLD_MIN = -math.pi * EARTH_RADIUS_M
SWISS_BOUNDS = GeoReference(west=5.86, south=45.78, east=10.55, north=47.84)
SWISSIMAGE_URL = "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/current/3857/{z}/{x}/{y}.jpeg"


def load_lowres_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _cell_bbox(georef: GeoReference, row: int, col: int, rows: int, cols: int) -> tuple[float, float, float, float]:
    width = (georef.east - georef.west) / cols
    height = (georef.north - georef.south) / rows
    west = georef.west + col * width
    east = west + width
    north = georef.north - row * height
    south = north - height
    return (west, south, east, north)


def _window_score(image: Image.Image, box: tuple[int, int, int, int]) -> float:
    crop = image.crop(box).convert("RGB")
    edges = crop.convert("L").filter(ImageFilter.FIND_EDGES)
    red, green, blue = (value / 255 for value in ImageStat.Stat(crop).mean)
    brightness = (red + green + blue) / 3
    color_range = max(red, green, blue) - min(red, green, blue)
    edge_density = ImageStat.Stat(edges).mean[0] / 255
    grayness = 1 - min(1, color_range * 3.2)
    road_brightness = 1 - min(1, abs(brightness - 0.52) / 0.32)
    green_dominance = max(0.0, green - max(red, blue))
    blue_dominance = max(0.0, blue - max(red, green))
    snow_or_cloud = max(0.0, brightness - 0.72) * grayness
    score = edge_density * 1.75 + grayness * 0.34 + road_brightness * 0.34
    score -= green_dominance * 2.4 + blue_dominance * 2.1 + snow_or_cloud * 1.7
    return max(0.0, min(1.0, score))


def build_urban_grid(
    image: Image.Image,
    georef: GeoReference,
    *,
    rows: int = 48,
    cols: int = 72,
    threshold: float = 0.45,
    min_cluster_cells: int = 1,
) -> VisionGridPlan:
    width, height = image.size
    cells: list[GridCell] = []
    for row in range(rows):
        for col in range(cols):
            left = round(col * width / cols)
            right = round((col + 1) * width / cols)
            top = round(row * height / rows)
            bottom = round((row + 1) * height / rows)
            score = _window_score(image, (left, top, right, bottom))
            cells.append(
                GridCell(
                    row=row,
                    col=col,
                    bbox=_cell_bbox(georef, row, col, rows, cols),
                    urban_score=score,
                    is_urban=score >= threshold,
                )
            )

    clusters = _connected_clusters(cells, rows, cols, min_cluster_cells)
    return VisionGridPlan(rows=rows, cols=cols, cells=tuple(cells), clusters=tuple(clusters))


def build_urban_grid_from_surface(
    surface: np.ndarray,
    georef: GeoReference,
    *,
    rows: int = 60,
    cols: int = 96,
    threshold: float = 0.08,
    min_cluster_cells: int = 1,
) -> VisionGridPlan:
    height, width = surface.shape
    cells: list[GridCell] = []
    for row in range(rows):
        for col in range(cols):
            left = round(col * width / cols)
            right = round((col + 1) * width / cols)
            top = round(row * height / rows)
            bottom = round((row + 1) * height / rows)
            region = surface[top:bottom, left:right]
            score = float(region.mean()) if region.size else 0.0
            cells.append(
                GridCell(
                    row=row,
                    col=col,
                    bbox=_cell_bbox(georef, row, col, rows, cols),
                    urban_score=score,
                    is_urban=score >= threshold,
                )
            )

    clusters = _connected_clusters(cells, rows, cols, min_cluster_cells)
    return VisionGridPlan(rows=rows, cols=cols, cells=tuple(cells), clusters=tuple(clusters))


def _connected_clusters(cells: Iterable[GridCell], rows: int, cols: int, min_cluster_cells: int) -> list[UrbanCluster]:
    by_pos = {(cell.row, cell.col): cell for cell in cells}
    visited: set[tuple[int, int]] = set()
    clusters: list[UrbanCluster] = []

    for cell in by_pos.values():
        key = (cell.row, cell.col)
        if key in visited or not cell.is_urban:
            continue
        queue: deque[GridCell] = deque([cell])
        visited.add(key)
        cluster_cells: list[GridCell] = []

        while queue:
            current = queue.popleft()
            cluster_cells.append(current)
            for neighbor in _neighbors(current.row, current.col, rows, cols):
                if neighbor in visited:
                    continue
                next_cell = by_pos[neighbor]
                if not next_cell.is_urban:
                    continue
                visited.add(neighbor)
                queue.append(next_cell)

        if len(cluster_cells) < min_cluster_cells:
            continue
        clusters.append(_make_cluster(len(clusters) + 1, cluster_cells))

    return sorted(clusters, key=lambda cluster: (cluster.urban_score, cluster.cell_count), reverse=True)


def _neighbors(row: int, col: int, rows: int, cols: int) -> Iterable[tuple[int, int]]:
    for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
        if 0 <= next_row < rows and 0 <= next_col < cols:
            yield next_row, next_col


def _make_cluster(index: int, cells: list[GridCell]) -> UrbanCluster:
    west = min(cell.bbox[0] for cell in cells)
    south = min(cell.bbox[1] for cell in cells)
    east = max(cell.bbox[2] for cell in cells)
    north = max(cell.bbox[3] for cell in cells)
    score = sum(cell.urban_score for cell in cells) / len(cells)
    return UrbanCluster(
        cluster_id=f"urban-cluster-{index:03d}",
        bbox=(west, south, east, north),
        cell_count=len(cells),
        urban_score=score,
        cells=tuple(cells),
    )


def _split_large_cluster(cluster: UrbanCluster, *, max_cells: int = 72) -> list[list[GridCell]]:
    if cluster.cell_count <= max_cells:
        return [list(cluster.cells)]
    span = max(4, round(math.sqrt(max_cells)))
    groups: dict[tuple[int, int], list[GridCell]] = {}
    for cell in cluster.cells:
        groups.setdefault((cell.row // span, cell.col // span), []).append(cell)
    return [cells for cells in groups.values() if cells]


def _tile_xy(latitude: float, longitude: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(latitude)
    scale = 2**zoom
    x = int((longitude + 180) / 360 * scale)
    y = int((1 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2 * scale)
    return x, y


def _lat_lng_to_mercator(latitude: float, longitude: float) -> tuple[float, float]:
    x = EARTH_RADIUS_M * math.radians(longitude)
    y = EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2))
    return x, y


def _bbox_to_mercator(bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, south, east, north = bbox
    min_x, min_y = _lat_lng_to_mercator(south, west)
    max_x, max_y = _lat_lng_to_mercator(north, east)
    return min_x, min_y, max_x, max_y


def _mercator_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    min_x, min_y, max_x, max_y = bbox
    x = (min_x + max_x) / 2
    y = (min_y + max_y) / 2
    longitude = math.degrees(x / EARTH_RADIUS_M)
    latitude = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    return latitude, longitude


def _mercator_to_lat_lng(x: float, y: float) -> tuple[float, float]:
    longitude = math.degrees(x / EARTH_RADIUS_M)
    latitude = math.degrees(2 * math.atan(math.exp(y / EARTH_RADIUS_M)) - math.pi / 2)
    return latitude, longitude


def load_swissimage_mosaic(cache_dir: str | Path, *, zoom: int = 9, max_width: int = 1536) -> Image.Image:
    cache_path = Path(cache_dir) / f"swissimage-z{zoom}-w{max_width}.jpg"
    if cache_path.exists():
        return load_lowres_image(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    min_x, min_y = _tile_xy(SWISS_BOUNDS.north, SWISS_BOUNDS.west, zoom)
    max_x, max_y = _tile_xy(SWISS_BOUNDS.south, SWISS_BOUNDS.east, zoom)
    tile_size = 256
    mosaic = Image.new("RGB", ((max_x - min_x + 1) * tile_size, (max_y - min_y + 1) * tile_size))

    for tile_x in range(min_x, max_x + 1):
        for tile_y in range(min_y, max_y + 1):
            url = SWISSIMAGE_URL.format(z=zoom, x=tile_x, y=tile_y)
            with urlopen(url, timeout=20) as response:
                tile = Image.open(response).convert("RGB")
            mosaic.paste(tile, ((tile_x - min_x) * tile_size, (tile_y - min_y) * tile_size))

    if mosaic.width > max_width:
        height = round(mosaic.height * max_width / mosaic.width)
        mosaic = mosaic.resize((max_width, height), Image.Resampling.BILINEAR)
    mosaic.save(cache_path, quality=88)
    return mosaic


def _panel_split(rank: int) -> str:
    if rank % 10 == 0:
        return "test"
    if rank % 5 == 0:
        return "val"
    return "train"


def _split_bbox(bbox: tuple[float, float, float, float]) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    min_x, min_y, max_x, max_y = bbox
    if max_x - min_x >= max_y - min_y:
        mid = (min_x + max_x) / 2
        return (min_x, min_y, mid, max_y), (mid, min_y, max_x, max_y)
    mid = (min_y + max_y) / 2
    return (min_x, min_y, max_x, mid), (min_x, mid, max_x, max_y)


def _grid_range_bbox(georef: GeoReference, row0: int, row1: int, col0: int, col1: int, rows: int, cols: int) -> tuple[float, float, float, float]:
    west = georef.west + col0 * (georef.east - georef.west) / cols
    east = georef.west + col1 * (georef.east - georef.west) / cols
    north = georef.north - row0 * (georef.north - georef.south) / rows
    south = georef.north - row1 * (georef.north - georef.south) / rows
    return west, south, east, north


def _intersects(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> bool:
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def _tile_layer_for_size(size_m: int, base_size_m: int) -> int:
    return max(0, round(math.log(size_m / base_size_m, 2)))


def _aligned_grid_origin(bbox: tuple[float, float, float, float], base_size_m: int) -> tuple[float, float]:
    return WEB_MERCATOR_WORLD_MIN, WEB_MERCATOR_WORLD_MIN


def _align_bbox_to_square_grid(
    bbox: tuple[float, float, float, float],
    *,
    origin_x: float,
    origin_y: float,
    size_m: int,
) -> tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = bbox
    col0 = math.floor((min_x - origin_x) / size_m)
    row0 = math.floor((min_y - origin_y) / size_m)
    col1 = math.ceil((max_x - origin_x) / size_m)
    row1 = math.ceil((max_y - origin_y) / size_m)
    return (
        origin_x + col0 * size_m,
        origin_y + row0 * size_m,
        origin_x + col1 * size_m,
        origin_y + row1 * size_m,
    )


def _grid_stats_for_bbox(
    grid: VisionGridPlan,
    cells_by_pos: dict[tuple[int, int], GridCell],
    bbox: tuple[float, float, float, float],
    panel_by_pos: dict[tuple[int, int], str],
) -> dict:
    south, west = _mercator_to_lat_lng(bbox[0], bbox[1])
    north, east = _mercator_to_lat_lng(bbox[2], bbox[3])
    row0 = max(0, math.floor((SWISS_BOUNDS.north - north) / (SWISS_BOUNDS.north - SWISS_BOUNDS.south) * grid.rows) - 1)
    row1 = min(grid.rows, math.ceil((SWISS_BOUNDS.north - south) / (SWISS_BOUNDS.north - SWISS_BOUNDS.south) * grid.rows) + 1)
    col0 = max(0, math.floor((west - SWISS_BOUNDS.west) / (SWISS_BOUNDS.east - SWISS_BOUNDS.west) * grid.cols) - 1)
    col1 = min(grid.cols, math.ceil((east - SWISS_BOUNDS.west) / (SWISS_BOUNDS.east - SWISS_BOUNDS.west) * grid.cols) + 1)
    matched = [cells_by_pos[(row, col)] for row in range(row0, row1) for col in range(col0, col1) if (row, col) in cells_by_pos]
    urban_scores = [cell.urban_score for cell in matched]
    urban_ratio = sum(1 for cell in matched if cell.is_urban) / max(1, len(matched))
    avg_score = sum(urban_scores) / max(1, len(urban_scores))
    max_score = max(urban_scores, default=0.0)
    panel_ids = [panel_by_pos[(cell.row, cell.col)] for cell in matched if (cell.row, cell.col) in panel_by_pos]
    status = "urban" if urban_ratio >= 0.25 or avg_score >= 0.08 else "candidate" if max_score >= 0.04 or avg_score >= 0.015 else "background"
    return {
        "status": status,
        "urbanScore": avg_score,
        "maxUrbanScore": max_score,
        "urbanRatio": urban_ratio,
        "panelId": max(set(panel_ids), key=panel_ids.count) if panel_ids else None,
    }


def _build_bvh_cells(
    grid: VisionGridPlan,
    panel_by_pos: dict[tuple[int, int], str],
    *,
    base_size_m: int,
    scene_size_m: int,
) -> list[dict]:
    swiss_bbox = _bbox_to_mercator((SWISS_BOUNDS.west, SWISS_BOUNDS.south, SWISS_BOUNDS.east, SWISS_BOUNDS.north))
    origin_x, origin_y = _aligned_grid_origin(swiss_bbox, base_size_m)
    scene_layer = _tile_layer_for_size(scene_size_m, base_size_m)
    top_layer = max(scene_layer + 5, math.ceil(math.log(max(swiss_bbox[2] - swiss_bbox[0], swiss_bbox[3] - swiss_bbox[1]) / 4 / base_size_m, 2)))
    top_size = base_size_m * 2**top_layer
    cells_by_pos = {(cell.row, cell.col): cell for cell in grid.cells}
    bvh_cells: list[dict] = []

    def visit(bbox: tuple[float, float, float, float], depth: int, layer_above_base: int, row: int, col: int, cell_id: str) -> None:
        stats = _grid_stats_for_bbox(grid, cells_by_pos, bbox, panel_by_pos)
        bvh_cells.append(
            {
                "id": cell_id,
                "depth": depth,
                "layerAboveBase": layer_above_base,
                "row": row,
                "col": col,
                "sizeM": base_size_m * 2**layer_above_base,
                "status": stats["status"],
                "urbanScore": stats["urbanScore"],
                "maxUrbanScore": stats["maxUrbanScore"],
                "urbanRatio": stats["urbanRatio"],
                "bboxMercator": list(bbox),
                "panelId": stats["panelId"],
            }
        )
        should_split = layer_above_base > scene_layer + 4 and (
            depth < 2 or stats["maxUrbanScore"] >= 0.04 or stats["urbanScore"] >= 0.015 or stats["urbanRatio"] > 0
        )
        if not should_split:
            return
        min_x, min_y, max_x, max_y = bbox
        mid_x = (min_x + max_x) / 2
        mid_y = (min_y + max_y) / 2
        child_layer = layer_above_base - 1
        for child_index, (child_bbox, child_row, child_col) in enumerate(
            (
                ((min_x, min_y, mid_x, mid_y), row * 2, col * 2),
                ((mid_x, min_y, max_x, mid_y), row * 2, col * 2 + 1),
                ((min_x, mid_y, mid_x, max_y), row * 2 + 1, col * 2),
                ((mid_x, mid_y, max_x, max_y), row * 2 + 1, col * 2 + 1),
            )
        ):
            if _intersects(child_bbox, swiss_bbox):
                visit(child_bbox, depth + 1, child_layer, child_row, child_col, f"{cell_id}.{child_index}")

    min_col = math.floor((swiss_bbox[0] - origin_x) / top_size)
    max_col = math.ceil((swiss_bbox[2] - origin_x) / top_size)
    min_row = math.floor((swiss_bbox[1] - origin_y) / top_size)
    max_row = math.ceil((swiss_bbox[3] - origin_y) / top_size)
    for row in range(min_row, max_row):
        for col in range(min_col, max_col):
            bbox = (
                origin_x + col * top_size,
                origin_y + row * top_size,
                origin_x + (col + 1) * top_size,
                origin_y + (row + 1) * top_size,
            )
            if _intersects(bbox, swiss_bbox):
                visit(bbox, 0, top_layer, row, col, f"ch-l{top_layer}-r{row}-c{col}")
    return bvh_cells


def _ranked_panel_cells(
    panel: dict,
    planned_scenes: int,
    scene_size_m: int,
    *,
    base_size_m: int,
    grid: VisionGridPlan,
    panel_by_pos: dict[tuple[int, int], str],
) -> list[dict]:
    origin_x, origin_y = _aligned_grid_origin(_bbox_to_mercator((SWISS_BOUNDS.west, SWISS_BOUNDS.south, SWISS_BOUNDS.east, SWISS_BOUNDS.north)), base_size_m)
    aligned_bbox = _align_bbox_to_square_grid(tuple(panel["bboxMercator"]), origin_x=origin_x, origin_y=origin_y, size_m=scene_size_m)
    cells_by_pos = {(cell.row, cell.col): cell for cell in grid.cells}
    col0 = math.floor((aligned_bbox[0] - origin_x) / scene_size_m)
    col1 = math.ceil((aligned_bbox[2] - origin_x) / scene_size_m)
    row0 = math.floor((aligned_bbox[1] - origin_y) / scene_size_m)
    row1 = math.ceil((aligned_bbox[3] - origin_y) / scene_size_m)
    cells: list[dict] = []
    for row in range(row0, row1):
        for col in range(col0, col1):
            bbox = (
                origin_x + col * scene_size_m,
                origin_y + row * scene_size_m,
                origin_x + (col + 1) * scene_size_m,
                origin_y + (row + 1) * scene_size_m,
            )
            stats = _grid_stats_for_bbox(grid, cells_by_pos, bbox, panel_by_pos)
            if stats["maxUrbanScore"] < 0.02 and stats["panelId"] != panel["id"]:
                continue
            cells.append(
                {
                    "id": f"{panel['id']}.r{row}.c{col}",
                    "panelId": panel["id"],
                    "panelName": panel["name"],
                    "status": "selected",
                    "depth": 0,
                    "layerAboveBase": _tile_layer_for_size(scene_size_m, base_size_m),
                    "bboxMercator": list(bbox),
                    "sizeM": scene_size_m,
                    "score": max(stats["urbanScore"], stats["maxUrbanScore"] * 0.72),
                }
            )
    return sorted(cells, key=lambda cell: cell["score"], reverse=True)[:planned_scenes]


def build_web_plan(
    *,
    target_positive_count: int = 500,
    max_panels: int = 8,
    perimeter_budget: int = 72,
    scene_size_m: int = 800,
    tile_size_m: int = 25,
    image_px: int = 2048,
    cache_dir: str | Path = "data/cache/autopilot",
) -> dict:
    rows, cols = 60, 96
    swiss_bbox_mercator = _bbox_to_mercator((SWISS_BOUNDS.west, SWISS_BOUNDS.south, SWISS_BOUNDS.east, SWISS_BOUNDS.north))
    roads = load_official_roads_mosaic(cache_dir, swiss_bbox_mercator)
    surface, _density, surface_threshold = urban_density_mask_from_roads(roads)
    grid = build_urban_grid_from_surface(surface, SWISS_BOUNDS, rows=rows, cols=cols, threshold=0.08, min_cluster_cells=1)
    scene_budget = max(4, min(120, perimeter_budget))
    active_count = max(1, min(24, max_panels))
    origin_x, origin_y = _aligned_grid_origin(swiss_bbox_mercator, tile_size_m)
    panel_clusters: list[UrbanCluster] = []
    for cluster in grid.clusters:
        for cells in _split_large_cluster(cluster):
            panel_clusters.append(_make_cluster(len(panel_clusters) + 1, cells))
    weighted_clusters = sorted(
        panel_clusters,
        key=lambda cluster: cluster.urban_score * math.sqrt(cluster.cell_count),
        reverse=True,
    )
    active_clusters = weighted_clusters[:active_count]
    total_weight = sum(cluster.urban_score * math.sqrt(cluster.cell_count) for cluster in active_clusters) or 1

    panels = []
    coarse_cells = []
    cluster_by_pos = {}
    for cluster_index, cluster in enumerate(weighted_clusters):
        cluster_id = f"urban-panel-{cluster_index + 1:03d}"
        for cell in cluster.cells:
            cluster_by_pos[(cell.row, cell.col)] = cluster_id
        weight = cluster.urban_score * math.sqrt(cluster.cell_count)
        planned = max(1, math.floor(weight / total_weight * scene_budget)) if cluster_index < active_count else 0
        panel_bbox = _align_bbox_to_square_grid(
            _bbox_to_mercator(cluster.bbox),
            origin_x=origin_x,
            origin_y=origin_y,
            size_m=scene_size_m,
        )
        panels.append(
            {
                "id": cluster_id,
                "name": f"Urban panel {cluster_index + 1:03d}",
                "split": _panel_split(cluster_index + 1),
                "rank": cluster_index + 1,
                "coarseCellCount": cluster.cell_count,
                "urbanScore": cluster.urban_score,
                "bboxMercator": list(panel_bbox),
                "plannedScenes": planned,
                "estimatedPositiveCount": planned * 7,
            }
        )

    planned_total = sum(panel["plannedScenes"] for panel in panels[:active_count])
    for panel in panels[:active_count]:
        if planned_total >= scene_budget:
            break
        panel["plannedScenes"] += 1
        panel["estimatedPositiveCount"] += 7
        planned_total += 1

    for cell in grid.cells:
        status = "urban" if cell.is_urban else "candidate" if cell.urban_score >= 0.42 else "background"
        coarse_cells.append(
            {
                "id": f"ch-r{cell.row:02d}-c{cell.col:02d}",
                "row": cell.row,
                "col": cell.col,
                "status": status,
                "urbanScore": cell.urban_score,
                "bboxMercator": list(_bbox_to_mercator(cell.bbox)),
                "panelId": cluster_by_pos.get((cell.row, cell.col)),
            }
        )
    bvh_cells = _build_bvh_cells(grid, cluster_by_pos, base_size_m=tile_size_m, scene_size_m=scene_size_m)

    selected = []
    for panel in panels[:active_count]:
        selected.extend(
            _ranked_panel_cells(
                panel,
                panel["plannedScenes"],
                scene_size_m,
                base_size_m=tile_size_m,
                grid=grid,
                panel_by_pos=cluster_by_pos,
            )
        )
    selected = sorted(selected, key=lambda cell: cell["score"], reverse=True)[:scene_budget]

    cells = []
    scenes = []
    for index, cell in enumerate(selected, start=1):
        latitude, longitude = _mercator_center(tuple(cell["bboxMercator"]))
        scene_id = f"auto-panel-{index:03d}"
        cells.append({**cell, "rank": index, "center": {"latitude": latitude, "longitude": longitude}, "sceneId": scene_id})
        scenes.append(
            {
                "scene_id": scene_id,
                "city": cell["panelName"],
                "split": next((panel["split"] for panel in panels if panel["id"] == cell["panelId"]), "train"),
                "latitude": latitude,
                "longitude": longitude,
                "size_m": scene_size_m,
                "image_px": image_px,
                "autopilot_rank": index,
                "autopilot_score": cell["score"],
                "autopilot_city_id": cell["panelId"],
                "autopilot_cell_id": cell["id"],
            }
        )

    return {
        "version": 6,
        "mode": "swiss-lowres-urban-grid",
        "source": "swisstopo-official-road-density",
        "segmentation": {
            "sourceLayer": SWISSTLM3D_ROADS_LAYER,
            "method": "official-road-density",
            "surfaceThreshold": surface_threshold,
            "surfaceCoverage": float(surface.mean()),
        },
        "gridGeometry": {
            "baseGridSizeM": tile_size_m,
            "sceneGridSizeM": scene_size_m,
            "sceneLayerAboveBase": _tile_layer_for_size(scene_size_m, tile_size_m),
            "originMercator": [origin_x, origin_y],
            "alignment": "web-mercator-tile-origin",
            "rule": "cellSizeM = baseGridSizeM * 2 ** layerAboveBase",
        },
        "targetPositiveCount": target_positive_count,
        "estimatedPositiveCount": len(scenes) * 7,
        "estimatedPositivePerScene": 7,
        "sceneSizeM": scene_size_m,
        "tileSizeM": tile_size_m,
        "imagePx": image_px,
        "maxPanels": active_count,
        "sceneBudget": scene_budget,
        "coarseGrid": {"rows": rows, "cols": cols, "bboxMercator": list(_bbox_to_mercator((SWISS_BOUNDS.west, SWISS_BOUNDS.south, SWISS_BOUNDS.east, SWISS_BOUNDS.north)))},
        "coarseCells": coarse_cells,
        "bvhCells": bvh_cells,
        "panels": panels,
        "cells": cells,
        "scenes": scenes,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--web-plan", action="store_true")
    parser.add_argument("--target-positive-count", type=int, default=500)
    parser.add_argument("--max-panels", type=int, default=8)
    parser.add_argument("--perimeter-budget", type=int, default=72)
    parser.add_argument("--cache-dir", default="data/cache/autopilot")
    args = parser.parse_args()
    if args.web_plan:
        print(json.dumps(build_web_plan(target_positive_count=args.target_positive_count, max_panels=args.max_panels, perimeter_budget=args.perimeter_budget, cache_dir=args.cache_dir)))


if __name__ == "__main__":
    main()
