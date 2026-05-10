from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import tomllib
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosswalk_detector.autopilot import EARTH_RADIUS_M
from crosswalk_detector.urban_vision import _dilate, _erode
from scripts.official_urban_density_poc import _draw_title, _raw_roads_panel, _urban_density_mask


WMS_BASE_URL = "https://wms.geo.admin.ch/"
SWISSTLM3D_ROADS_LAYER = "ch.swisstopo.swisstlm3d-strassen"

REGIONS = {
    "switzerland": {"latitude": 46.82, "longitude": 8.23, "size_m": 440_000},
    "north-corridor": {"latitude": 47.22, "longitude": 8.25, "size_m": 190_000},
    "zurich-winterthur": {"latitude": 47.44, "longitude": 8.64, "size_m": 70_000},
}


@dataclass(frozen=True)
class GridCell:
    cell_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    level: int
    status: str
    leaf_count: int
    green_leaf_count: int
    surface_ratio: float

    @property
    def has_surface(self) -> bool:
        return self.status in {"green", "orange"}


def _mercator_from_lat_lon(latitude: float, longitude: float) -> tuple[float, float]:
    return (
        EARTH_RADIUS_M * math.radians(longitude),
        EARTH_RADIUS_M * math.log(math.tan(math.pi / 4.0 + math.radians(latitude) / 2.0)),
    )


def _build_wms_url(latitude: float, longitude: float, size_m: int, image_px: int) -> str:
    center_x, center_y = _mercator_from_lat_lon(latitude, longitude)
    half = size_m / 2
    bbox = (center_x - half, center_y - half, center_x + half, center_y + half)
    query = urlencode(
        {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": "1.3.0",
            "LAYERS": SWISSTLM3D_ROADS_LAYER,
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
    return f"{WMS_BASE_URL}?{query}"


def _fetch_official_roads(region_name: str, region: dict, image_px: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{region_name}-{region['size_m']}m-{image_px}px-{SWISSTLM3D_ROADS_LAYER}.png"
    if path.exists():
        return path
    url = _build_wms_url(region["latitude"], region["longitude"], region["size_m"], image_px)
    with urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())
    return path


def _surface_ratio(mask: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> float:
    region = mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0.0
    return float(region.mean())


def _build_bvh_grid(mask: np.ndarray, coarse_cells: int, max_depth: int, leaf_min_surface_ratio: float) -> tuple[list[GridCell], list[GridCell]]:
    image_size = mask.shape[0]
    root_cell = image_size // coarse_cells
    visible: list[GridCell] = []
    all_nodes: list[GridCell] = []

    def visit(x0: int, y0: int, size: int, level: int, cell_id: str) -> GridCell:
        x1 = min(image_size, x0 + size)
        y1 = min(image_size, y0 + size)
        ratio = _surface_ratio(mask, x0, y0, x1, y1)
        if ratio <= 0:
            remaining_depth = max(max_depth - level, 0)
            theoretical_leaf_count = 4**remaining_depth
            cell = GridCell(cell_id, x0, y0, x1, y1, level, "empty", theoretical_leaf_count, 0, ratio)
            visible.append(cell)
            all_nodes.append(cell)
            return cell
        if level >= max_depth or size <= 1:
            status = "green" if ratio > leaf_min_surface_ratio else "empty"
            leaf_count = 1
            green_leaf_count = 1 if status == "green" else 0
            cell = GridCell(cell_id, x0, y0, x1, y1, level, status, leaf_count, green_leaf_count, ratio)
            visible.append(cell)
            all_nodes.append(cell)
            return cell

        half = size // 2
        children: list[GridCell] = []
        for dy in (0, half):
            for dx in (0, half):
                children.append(visit(x0 + dx, y0 + dy, half, level + 1, f"{cell_id}.{len(children)}"))

        leaf_count = sum(child.leaf_count for child in children)
        green_leaf_count = sum(child.green_leaf_count for child in children)
        if leaf_count > 0 and green_leaf_count == leaf_count:
            status = "green"
        elif green_leaf_count > 0:
            status = "orange"
        else:
            status = "empty"
        cell = GridCell(cell_id, x0, y0, x1, y1, level, status, leaf_count, green_leaf_count, ratio)
        all_nodes.append(cell)
        if status == "green":
            visible[:] = [candidate for candidate in visible if not _is_descendant(candidate.cell_id, cell_id)]
            visible.append(cell)
        elif status == "orange":
            visible.append(cell)
        return cell

    for y in range(0, root_cell * coarse_cells, root_cell):
        for x in range(0, root_cell * coarse_cells, root_cell):
            visit(x, y, root_cell, 0, f"r{y // root_cell:02d}c{x // root_cell:02d}")
    return visible, all_nodes


def _is_descendant(cell_id: str, parent_id: str) -> bool:
    return cell_id.startswith(f"{parent_id}.")


def _candidate_mask_from_cells(shape: tuple[int, int], leaves: list[GridCell]) -> np.ndarray:
    candidate = np.zeros(shape, dtype=bool)
    for cell in leaves:
        if cell.status == "green":
            candidate[cell.y0 : cell.y1, cell.x0 : cell.x1] = True
    return candidate


def _draw_surface(image: Image.Image, surface: np.ndarray, title: str) -> Image.Image:
    base = image.convert("RGBA")
    fill = Image.new("RGBA", base.size, (34, 197, 94, 0))
    fill.putalpha(Image.fromarray(surface.astype(np.uint8) * 70))
    base = Image.alpha_composite(base, fill)

    boundary = surface & ~_erode(surface, iterations=1)
    boundary = _dilate(boundary, iterations=1)
    line = Image.new("RGBA", base.size, (34, 197, 94, 0))
    line.putalpha(Image.fromarray(boundary.astype(np.uint8) * 255))
    base = Image.alpha_composite(base, line)
    return _draw_title(base.convert("RGB"), title)


def _draw_grid(image: Image.Image, cells: list[GridCell], surface: np.ndarray, title: str) -> Image.Image:
    base = image.convert("RGBA")
    candidate = _candidate_mask_from_cells(surface.shape, cells)
    candidate_fill = Image.new("RGBA", base.size, (34, 197, 94, 0))
    candidate_fill.putalpha(Image.fromarray(candidate.astype(np.uint8) * 54))
    base = Image.alpha_composite(base, candidate_fill)

    surface_fill = Image.new("RGBA", base.size, (59, 130, 246, 0))
    surface_fill.putalpha(Image.fromarray(surface.astype(np.uint8) * 80))
    base = Image.alpha_composite(base, surface_fill)

    draw = ImageDraw.Draw(base, "RGBA")
    for cell in sorted(cells, key=lambda item: (item.level, item.x0, item.y0)):
        if cell.status == "green":
            color = (34, 197, 94, 230)
            width = max(1, 3 - min(cell.level, 2))
        elif cell.status == "orange":
            color = (251, 146, 60, 235)
            width = max(1, 4 - min(cell.level, 3))
        else:
            color = (148, 163, 184, 65)
            width = 1
        draw.rectangle((cell.x0, cell.y0, cell.x1, cell.y1), outline=color, width=width)
    return _draw_title(base.convert("RGB"), title)


def _build_report(
    region_name: str,
    region: dict,
    visible: list[GridCell],
    all_nodes: list[GridCell],
    surface: np.ndarray,
    candidate: np.ndarray,
    output: Path,
    threshold: float,
    *,
    base_size_m: int,
    leaf_size_m: float,
    leaf_px: float,
) -> dict:
    leaves = [cell for cell in all_nodes if cell.leaf_count == 1]
    roots = [cell for cell in all_nodes if cell.level == 0]
    total_leaf_cells = sum(cell.leaf_count for cell in roots)
    green_leaf_cells = sum(cell.green_leaf_count for cell in roots)
    green = [cell for cell in all_nodes if cell.status == "green"]
    orange = [cell for cell in all_nodes if cell.status == "orange"]
    empty = [cell for cell in all_nodes if cell.status == "empty"]
    missed_surface = int((surface & ~candidate).sum())
    false_grid_area = int((candidate & ~surface).sum())
    return {
        "region": region_name,
        "sizeM": region["size_m"],
        "baseSizeM": base_size_m,
        "leafSizeM": leaf_size_m,
        "leafPixels": leaf_px,
        "datasetExact": leaf_size_m <= base_size_m * 1.01 and leaf_px >= 1,
        "visibleCells": len(visible),
        "allNodes": len(all_nodes),
        "terminalCellsVisited": len(leaves),
        "leafCells": total_leaf_cells,
        "greenCells": len(green),
        "orangeCells": len(orange),
        "emptyCells": len(empty),
        "greenLeafCells": green_leaf_cells,
        "emptyLeafCells": total_leaf_cells - green_leaf_cells,
        "surfaceCoverage": float(surface.mean()),
        "surfaceThreshold": threshold,
        "verifiedGreenGridCoverage": float(candidate.mean()),
        "missedSurfacePixels": missed_surface,
        "extraGridPixelsAroundSurface": false_grid_area,
        "output": str(output),
    }


def _write_cells_json(path: Path, cells: list[GridCell], image_size: int, region_size_m: int) -> None:
    meters_per_pixel = region_size_m / image_size
    cells = []
    for cell in cells:
        cells.append(
            {
                "id": cell.cell_id,
                "x0": cell.x0,
                "y0": cell.y0,
                "x1": cell.x1,
                "y1": cell.y1,
                "level": cell.level,
                "status": cell.status,
                "leafCount": cell.leaf_count,
                "greenLeafCount": cell.green_leaf_count,
                "surfaceRatio": cell.surface_ratio,
                "sizeM": round((cell.x1 - cell.x0) * meters_per_pixel, 3),
            }
        )
    path.write_text(json.dumps({"imageSize": image_size, "regionSizeM": region_size_m, "cells": cells}, indent=2), encoding="utf-8")


def _scene_regions(config_path: Path, scene_size_m: int) -> dict[str, dict]:
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    regions = {}
    for scene in raw["scenes"]:
        regions[f"{scene['scene_id']}-{scene_size_m}m"] = {
            "latitude": float(scene["latitude"]),
            "longitude": float(scene["longitude"]),
            "size_m": scene_size_m,
        }
    return regions


def _max_depth_for_base(region_size_m: int, coarse_cells: int, base_size_m: int) -> int:
    root_size_m = region_size_m / coarse_cells
    if root_size_m <= base_size_m:
        return 0
    return max(0, math.ceil(math.log(root_size_m / base_size_m, 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", default="switzerland,north-corridor,zurich-winterthur")
    parser.add_argument("--image-px", type=int, default=1536)
    parser.add_argument("--coarse-cells", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--base-size-m", type=int, default=25)
    parser.add_argument("--leaf-min-surface-ratio", type=float, default=0.0)
    parser.add_argument("--surface-quantile", type=float, default=0.64)
    parser.add_argument("--minimum-threshold", type=float, default=0.24)
    parser.add_argument("--config", type=Path, default=Path("configs/real-dataset.toml"))
    parser.add_argument("--include-config-scenes", action="store_true")
    parser.add_argument("--scene-size-m", type=int, default=3200)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/official-bvh-grid-poc"))
    parser.add_argument("--output-dir", type=Path, default=Path("validation-output/official-bvh-grid-poc"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    regions = dict(REGIONS)
    if args.include_config_scenes:
        regions.update(_scene_regions(args.config, args.scene_size_m))
    for region_name in [name.strip() for name in args.regions.split(",") if name.strip()]:
        region = regions[region_name]
        max_depth = args.max_depth
        if max_depth is None:
            max_depth = _max_depth_for_base(region["size_m"], args.coarse_cells, args.base_size_m)
        road_path = _fetch_official_roads(region_name, region, args.image_px, args.cache_dir)
        roads = Image.open(road_path).convert("RGBA")
        surface, _density, threshold = _urban_density_mask(
            roads,
            surface_quantile=args.surface_quantile,
            minimum_threshold=args.minimum_threshold,
        )
        visible, all_nodes = _build_bvh_grid(surface, args.coarse_cells, max_depth, args.leaf_min_surface_ratio)
        candidate = _candidate_mask_from_cells(surface.shape, visible)
        root_size_m = region["size_m"] / args.coarse_cells
        leaf_size_m = root_size_m / 2**max_depth
        leaf_px = leaf_size_m * args.image_px / region["size_m"]

        panels = [
            _raw_roads_panel(roads, f"{region_name} · official roads"),
            _draw_surface(roads, surface, f"{region_name} · dense-road surface"),
            _draw_grid(roads, visible, surface, f"{region_name} · DFS verified grid · leaf {leaf_size_m:.1f} m"),
        ]
        gutter = 12
        montage = Image.new("RGB", (args.image_px * len(panels) + gutter * (len(panels) - 1), args.image_px), (18, 20, 24))
        x = 0
        for panel in panels:
            montage.paste(panel, (x, 0))
            x += panel.width + gutter
        output = args.output_dir / f"{region_name}-bvh-grid.jpg"
        montage.save(output, quality=92)
        _write_cells_json(args.output_dir / f"{region_name}-visible-cells.json", visible, surface.shape[0], region["size_m"])
        _write_cells_json(args.output_dir / f"{region_name}-all-nodes.json", all_nodes, surface.shape[0], region["size_m"])
        summaries.append(
            _build_report(
                region_name,
                region,
                visible,
                all_nodes,
                surface,
                candidate,
                output,
                threshold,
                base_size_m=args.base_size_m,
                leaf_size_m=leaf_size_m,
                leaf_px=leaf_px,
            )
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps({"outputDir": str(args.output_dir), "summary": summaries}, indent=2))


if __name__ == "__main__":
    main()
