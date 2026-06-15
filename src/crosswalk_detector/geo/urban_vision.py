from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageFilter

EARTH_RADIUS_M = 6_378_137


@dataclass(frozen=True)
class GeoReference:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class UrbanContour:
    contour_id: str
    bbox: tuple[float, float, float, float]
    bbox_mercator: tuple[float, float, float, float]
    polygon_mercator: tuple[tuple[float, float], ...]
    area_pixels: int
    urban_score: float


@dataclass(frozen=True)
class UrbanMaskResult:
    image: Image.Image
    score: np.ndarray
    mask: np.ndarray
    threshold: float


def _resize_for_mask(image: Image.Image, max_width: int = 320) -> Image.Image:
    if image.width <= max_width:
        return image.convert("RGB")
    height = round(image.height * max_width / image.width)
    return image.resize((max_width, height), Image.Resampling.BILINEAR).convert("RGB")


def _shift(mask: np.ndarray, dy: int, dx: int, fill: bool = False) -> np.ndarray:
    result = np.full(mask.shape, fill, dtype=bool)
    src_y0 = max(0, -dy)
    src_y1 = mask.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = mask.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = mask.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = mask.shape[1] - max(0, -dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]
    return result


def _shift_values(values: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    result = np.full(values.shape, fill, dtype=values.dtype)
    src_y0 = max(0, -dy)
    src_y1 = values.shape[0] - max(0, dy)
    src_x0 = max(0, -dx)
    src_x1 = values.shape[1] - max(0, dx)
    dst_y0 = max(0, dy)
    dst_y1 = values.shape[0] - max(0, -dy)
    dst_x0 = max(0, dx)
    dst_x1 = values.shape[1] - max(0, -dx)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = values[src_y0:src_y1, src_x0:src_x1]
    return result


def _dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    current = mask
    for _ in range(iterations):
        expanded = current.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    expanded |= _shift(current, dy, dx)
        current = expanded
    return current


def _erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    current = mask
    for _ in range(iterations):
        shrunk = current.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    shrunk &= _shift(current, dy, dx, fill=False)
        current = shrunk
    return current


def _close(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _erode(_dilate(mask, iterations), iterations)


def _open(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    return _dilate(_erode(mask, iterations), iterations)


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(values, radius, mode="edge")
    total = np.zeros(values.shape, dtype=np.float32)
    side = radius * 2 + 1
    for y in range(side):
        for x in range(side):
            total += padded[y : y + values.shape[0], x : x + values.shape[1]]
    return total / float(side * side)


def _component_masks(mask: np.ndarray, min_area: int) -> Iterable[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    rows, cols = mask.shape
    for start_row in range(rows):
        for start_col in range(cols):
            if visited[start_row, start_col] or not mask[start_row, start_col]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_row, start_col)])
            visited[start_row, start_col] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                row, col = queue.popleft()
                pixels.append((row, col))
                for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                    if 0 <= next_row < rows and 0 <= next_col < cols and mask[next_row, next_col] and not visited[next_row, next_col]:
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            if len(pixels) < min_area:
                continue
            component = np.zeros(mask.shape, dtype=bool)
            for row, col in pixels:
                component[row, col] = True
            yield component


def _stitch_boundary_loop(mask: np.ndarray) -> list[tuple[float, float]]:
    rows, cols = mask.shape
    edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row in range(rows):
        for col in range(cols):
            if not mask[row, col]:
                continue
            if row == 0 or not mask[row - 1, col]:
                edges.append(((col, row), (col + 1, row)))
            if col == cols - 1 or not mask[row, col + 1]:
                edges.append(((col + 1, row), (col + 1, row + 1)))
            if row == rows - 1 or not mask[row + 1, col]:
                edges.append(((col + 1, row + 1), (col, row + 1)))
            if col == 0 or not mask[row, col - 1]:
                edges.append(((col, row + 1), (col, row)))

    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)

    used: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    loops: list[list[tuple[int, int]]] = []
    for edge in edges:
        if edge in used:
            continue
        start, current = edge
        loop = [start]
        used.add(edge)
        while current != start and len(loop) <= len(edges) + 4:
            loop.append(current)
            candidates = outgoing.get(current, [])
            next_point = next((candidate for candidate in candidates if (current, candidate) not in used), None)
            if next_point is None:
                break
            used.add((current, next_point))
            current = next_point
        if len(loop) >= 4 and current == start:
            loops.append(loop)

    if not loops:
        ys, xs = np.where(mask)
        return [(float(xs.min()), float(ys.min())), (float(xs.max() + 1), float(ys.min())), (float(xs.max() + 1), float(ys.max() + 1)), (float(xs.min()), float(ys.max() + 1))]

    def signed_area(points: list[tuple[int, int]]) -> float:
        area = 0.0
        for index, (x1, y1) in enumerate(points):
            x2, y2 = points[(index + 1) % len(points)]
            area += x1 * y2 - x2 * y1
        return area / 2

    largest = max(loops, key=lambda points: abs(signed_area(points)))
    return [(float(x), float(y)) for x, y in largest]


def _point_line_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)))
    return math.hypot(px - (sx + t * dx), py - (sy + t * dy))


def _rdp(points: list[tuple[float, float]], epsilon: float) -> list[tuple[float, float]]:
    if len(points) <= 3:
        return points
    start = points[0]
    end = points[-1]
    distances = [_point_line_distance(point, start, end) for point in points[1:-1]]
    if not distances:
        return [start, end]
    max_index = int(np.argmax(distances)) + 1
    if distances[max_index - 1] <= epsilon:
        return [start, end]
    return _rdp(points[: max_index + 1], epsilon)[:-1] + _rdp(points[max_index:], epsilon)


def _pixel_to_lon_lat(x: float, y: float, width: int, height: int, georef: GeoReference) -> tuple[float, float]:
    lon = georef.west + (x / width) * (georef.east - georef.west)
    lat = georef.north - (y / height) * (georef.north - georef.south)
    return lon, lat


def _lat_lng_to_mercator(latitude: float, longitude: float) -> tuple[float, float]:
    return (
        EARTH_RADIUS_M * math.radians(longitude),
        EARTH_RADIUS_M * math.log(math.tan(math.pi / 4 + math.radians(latitude) / 2)),
    )


def detect_urban_contours(
    image: Image.Image,
    georef: GeoReference,
    *,
    max_width: int = 320,
    threshold: float = 0.52,
    min_area_pixels: int = 48,
    max_contours: int = 96,
) -> list[UrbanContour]:
    mask_result = build_urban_mask(image, max_width=max_width, threshold=threshold)
    small = mask_result.image
    score = mask_result.score
    mask = _open(_close(mask_result.mask, iterations=5), iterations=2)

    contours: list[UrbanContour] = []
    for index, component in enumerate(_component_masks(mask, min_area=min_area_pixels), start=1):
        ys, xs = np.where(component)
        boundary = _stitch_boundary_loop(component)
        simplified = _rdp(boundary + [boundary[0]], epsilon=1.6)[:-1]
        if len(simplified) > 160:
            step = max(1, len(simplified) // 160)
            simplified = simplified[::step]
        polygon: list[tuple[float, float]] = []
        for x, y in simplified:
            lon, lat = _pixel_to_lon_lat(x, y, small.width, small.height, georef)
            polygon.append(_lat_lng_to_mercator(lat, lon))
        if len(polygon) < 3:
            continue
        west, north = _pixel_to_lon_lat(float(xs.min()), float(ys.min()), small.width, small.height, georef)
        east, south = _pixel_to_lon_lat(float(xs.max() + 1), float(ys.max() + 1), small.width, small.height, georef)
        min_x, min_y = _lat_lng_to_mercator(south, west)
        max_x, max_y = _lat_lng_to_mercator(north, east)
        area = int(component.sum())
        contours.append(
            UrbanContour(
                contour_id=f"urban-contour-{index:03d}",
                bbox=(west, south, east, north),
                bbox_mercator=(min_x, min_y, max_x, max_y),
                polygon_mercator=tuple(polygon),
                area_pixels=area,
                urban_score=float(score[component].mean()),
            )
        )
    return sorted(contours, key=lambda contour: (contour.area_pixels, contour.urban_score), reverse=True)[:max_contours]


def detect_urban_envelopes(
    image: Image.Image,
    georef: GeoReference,
    *,
    max_width: int = 180,
    threshold: float = 0.52,
    connect_iterations: int | None = None,
    min_area_ratio: float = 0.012,
    max_contours: int = 24,
    method: str = "palette_pattern",
) -> list[UrbanContour]:
    if method == "texture":
        mask_result = build_urban_mask(image, max_width=max_width, threshold=threshold)
    elif method == "palette_pattern":
        mask_result = build_urban_palette_pattern_mask(image, max_width=max_width, threshold=threshold)
    elif method == "road_veins":
        mask_result = build_urban_road_vein_mask(image, max_width=max_width, threshold=threshold)
    elif method == "road_seeded":
        mask_result = build_urban_road_seeded_mask(image, max_width=max_width, threshold=threshold)
    elif method == "road_density":
        mask_result = build_urban_road_density_mask(image, max_width=max_width, threshold=threshold)
    else:
        raise ValueError(f"Unknown urban envelope method: {method}")
    small = mask_result.image
    score = mask_result.score
    iterations = connect_iterations if connect_iterations is not None else max(2, round(small.width / 45))
    mask = _close(mask_result.mask, iterations=iterations)
    mask = _open(mask, iterations=max(1, iterations // 3))
    min_area_pixels = max(24, round(mask.size * min_area_ratio))

    contours: list[UrbanContour] = []
    for index, component in enumerate(_component_masks(mask, min_area=min_area_pixels), start=1):
        ys, xs = np.where(component)
        boundary = _stitch_boundary_loop(component)
        simplified = _rdp(boundary + [boundary[0]], epsilon=max(1.6, small.width / 130))[:-1]
        if len(simplified) > 120:
            step = max(1, len(simplified) // 120)
            simplified = simplified[::step]
        polygon: list[tuple[float, float]] = []
        for x, y in simplified:
            lon, lat = _pixel_to_lon_lat(x, y, small.width, small.height, georef)
            polygon.append(_lat_lng_to_mercator(lat, lon))
        if len(polygon) < 3:
            continue
        west, north = _pixel_to_lon_lat(float(xs.min()), float(ys.min()), small.width, small.height, georef)
        east, south = _pixel_to_lon_lat(float(xs.max() + 1), float(ys.max() + 1), small.width, small.height, georef)
        min_x, min_y = _lat_lng_to_mercator(south, west)
        max_x, max_y = _lat_lng_to_mercator(north, east)
        contours.append(
            UrbanContour(
                contour_id=f"urban-envelope-{index:03d}",
                bbox=(west, south, east, north),
                bbox_mercator=(min_x, min_y, max_x, max_y),
                polygon_mercator=tuple(polygon),
                area_pixels=int(component.sum()),
                urban_score=float(score[component].mean()),
            )
        )
    return sorted(contours, key=lambda contour: (contour.area_pixels, contour.urban_score), reverse=True)[:max_contours]


def build_urban_mask(
    image: Image.Image,
    *,
    max_width: int = 320,
    threshold: float = 0.52,
) -> UrbanMaskResult:
    small = _resize_for_mask(image, max_width=max_width)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    gray_image = small.convert("L")
    edges = np.asarray(gray_image.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    brightness = (red + green + blue) / 3
    color_range = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    local_mean = _box_blur(brightness, radius=3)
    local_variance = _box_blur((brightness - local_mean) ** 2, radius=3)
    local_std = np.sqrt(np.maximum(0, local_variance))
    grayness = 1 - np.clip(color_range * 3.3, 0, 1)
    local_texture = _box_blur(edges, radius=2)
    green_dominance = np.maximum(0, green - np.maximum(red, blue))
    blue_dominance = np.maximum(0, blue - np.maximum(red, green))
    road_brightness = 1 - np.clip(np.abs(brightness - 0.53) / 0.34, 0, 1)
    dark_water = np.clip((0.24 - brightness) / 0.24, 0, 1) * np.clip((blue + green - red) * 2.2, 0, 1)
    bright_snow = np.maximum(0, brightness - 0.76) * grayness
    vegetation = np.clip(green_dominance * 4.5 + np.maximum(0, green - 0.36) * 1.2 - color_range * 0.15, 0, 1)
    built_texture = np.clip(local_texture * 1.7 + local_std * 2.4, 0, 1)

    score = built_texture * 0.55 + grayness * 0.42 + road_brightness * 0.22
    score -= vegetation * 0.85 + blue_dominance * 2.45 + dark_water * 1.4 + bright_snow * 2.25
    score = np.clip(score, 0, 1)
    adaptive_threshold = max(threshold, float(np.quantile(score, 0.76)))
    mask = score >= adaptive_threshold
    mask &= brightness > 0.24
    mask &= vegetation < 0.58
    mask = _open(_close(mask, iterations=1), iterations=1)
    mask = _close(mask, iterations=1)
    return UrbanMaskResult(image=small, score=score, mask=mask, threshold=adaptive_threshold)


def build_urban_palette_pattern_mask(
    image: Image.Image,
    *,
    max_width: int = 220,
    threshold: float = 0.5,
) -> UrbanMaskResult:
    small = _resize_for_mask(image, max_width=max_width)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    brightness = (red + green + blue) / 3
    color_range = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    grayness = 1 - np.clip(color_range * 3.0, 0, 1)
    green_dominance = np.maximum(0, green - np.maximum(red, blue))
    blue_dominance = np.maximum(0, blue - np.maximum(red, green))

    concrete = grayness * (1 - np.clip(np.abs(brightness - 0.56) / 0.28, 0, 1))
    asphalt = grayness * (1 - np.clip(np.abs(brightness - 0.34) / 0.22, 0, 1))
    roof = np.clip((red - green) * 3.2 + (red - blue) * 2.2 + (brightness - 0.28) * 1.4, 0, 1)
    built_palette = np.maximum.reduce((concrete, asphalt, roof * 0.82))

    gy, gx = np.gradient(gray)
    vertical_edges = _box_blur(np.abs(gx), radius=2)
    horizontal_edges = _box_blur(np.abs(gy), radius=2)
    edge_density = np.clip((vertical_edges + horizontal_edges) * 5.4, 0, 1)
    orthogonal_balance = 2 * np.minimum(vertical_edges, horizontal_edges) / (vertical_edges + horizontal_edges + 1e-4)

    local_mean = _box_blur(gray, radius=4)
    local_std = np.sqrt(np.maximum(0, _box_blur((gray - local_mean) ** 2, radius=4)))
    repetitive_detail = np.clip(local_std * 5.0 + edge_density * 0.55, 0, 1)
    block_pattern = np.clip(edge_density * (0.45 + orthogonal_balance * 0.9), 0, 1)

    vegetation = np.clip(green_dominance * 5.0 + np.maximum(0, green - 0.34) * 1.5 - grayness * 0.2, 0, 1)
    dark_water = np.clip((0.27 - brightness) / 0.27, 0, 1) * np.clip((blue + green - red) * 2.4, 0, 1)
    pale_water = np.clip(blue_dominance * 3.4 + np.maximum(0, green - red) * 1.5 - grayness * 0.28, 0, 1)
    water = np.maximum(dark_water, pale_water)
    snow_or_cloud = np.maximum(0, brightness - 0.78) * grayness

    score = built_palette * 0.5 + block_pattern * 0.44 + repetitive_detail * 0.24
    score -= vegetation * 0.92 + water * 1.35 + snow_or_cloud * 2.1 + blue_dominance * 1.8
    score = np.clip(score, 0, 1)

    smoothed_score = _box_blur(score, radius=max(2, round(small.width / 90)))
    adaptive_threshold = max(threshold, float(np.quantile(smoothed_score, 0.64)))
    mask = smoothed_score >= adaptive_threshold
    mask &= vegetation < 0.62
    mask &= water < 0.45
    mask = _open(_close(mask, iterations=2), iterations=1)
    return UrbanMaskResult(image=small, score=smoothed_score, mask=mask, threshold=adaptive_threshold)


def _line_response(values: np.ndarray, dy: int, dx: int, radius: int = 2) -> np.ndarray:
    along_forward = _shift_values(values, dy, dx)
    along_backward = _shift_values(values, -dy, -dx)
    line_continuity = (values + along_forward + along_backward) / 3.0

    side_dy = -dx
    side_dx = dy
    side_a = _shift_values(values, side_dy * radius, side_dx * radius)
    side_b = _shift_values(values, -side_dy * radius, -side_dx * radius)
    side_contrast = line_continuity - (side_a + side_b) / 2.0
    return np.clip(line_continuity * 0.65 + side_contrast * 2.4, 0, 1)


def _bright_ridge_response(gray: np.ndarray, radius: int) -> np.ndarray:
    smooth = _box_blur(gray, radius=radius)
    gy, gx = np.gradient(smooth)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)
    gxy = (gxy + gyx) / 2

    trace = gxx + gyy
    delta = np.sqrt(np.maximum(0, (gxx - gyy) ** 2 + 4 * gxy * gxy))
    lambda_a = (trace - delta) / 2
    lambda_b = (trace + delta) / 2
    abs_a = np.abs(lambda_a)
    abs_b = np.abs(lambda_b)
    lambda_small = np.where(abs_a <= abs_b, lambda_a, lambda_b)
    lambda_large = np.where(abs_a <= abs_b, lambda_b, lambda_a)

    ridge_polarity = np.clip(-lambda_large * 48 * radius, 0, 1)
    line_shape = 1 - np.clip(np.abs(lambda_small) / (np.abs(lambda_large) + 1e-5), 0, 1)
    strength = np.clip(np.sqrt(lambda_small * lambda_small + lambda_large * lambda_large) * 18 * radius, 0, 1)
    return np.clip(ridge_polarity * line_shape * (0.35 + strength * 0.65), 0, 1)


def _road_vein_scores(image: Image.Image, max_width: int) -> tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    small = _resize_for_mask(image, max_width=max_width)
    rgb = np.asarray(small, dtype=np.float32) / 255.0
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]
    brightness = (red + green + blue) / 3
    color_range = np.max(rgb, axis=2) - np.min(rgb, axis=2)
    grayness = 1 - np.clip(color_range * 3.0, 0, 1)
    green_dominance = np.maximum(0, green - np.maximum(red, blue))
    blue_dominance = np.maximum(0, blue - np.maximum(red, green))

    concrete = grayness * (1 - np.clip(np.abs(brightness - 0.58) / 0.32, 0, 1))
    asphalt = grayness * (1 - np.clip(np.abs(brightness - 0.36) / 0.24, 0, 1))
    built_palette = np.maximum(concrete, asphalt)

    gy, gx = np.gradient(gray)
    edge_density = np.clip(_box_blur(np.abs(gx) + np.abs(gy), radius=2) * 5.8, 0, 1)
    straight_line = np.maximum.reduce(
        (
            _line_response(built_palette, 0, 1),
            _line_response(built_palette, 1, 0),
            _line_response(built_palette, 1, 1),
            _line_response(built_palette, 1, -1),
        )
    )
    bright_line = np.maximum.reduce(
        (
            _line_response(gray, 0, 1),
            _line_response(gray, 1, 0),
            _line_response(gray, 1, 1),
            _line_response(gray, 1, -1),
        )
    )
    dark_line = np.maximum.reduce(
        (
            _line_response(1 - gray, 0, 1),
            _line_response(1 - gray, 1, 0),
            _line_response(1 - gray, 1, 1),
            _line_response(1 - gray, 1, -1),
        )
    )
    thin_line = np.maximum(bright_line, dark_line * built_palette * 0.32) * np.clip(grayness * 1.15, 0, 1)
    ridge_line = np.maximum.reduce((_bright_ridge_response(gray, 1), _bright_ridge_response(gray, 2), _bright_ridge_response(gray, 3)))
    road_candidates = np.clip(ridge_line * 0.8 + thin_line * 0.5 + straight_line * 0.24 + edge_density * built_palette * 0.16, 0, 1)
    road_network = _box_blur(road_candidates, radius=max(3, round(small.width / 70)))

    local_mean = _box_blur(gray, radius=4)
    local_std = np.sqrt(np.maximum(0, _box_blur((gray - local_mean) ** 2, radius=4)))
    detail = np.clip(local_std * 4.6 + edge_density * 0.35, 0, 1)

    vegetation_color = np.clip(green_dominance * 5.5 + np.maximum(0, green - 0.34) * 1.7 - grayness * 0.2, 0, 1)
    organic_dark = ((brightness < 0.42) & (blue + 0.015 < green) & (blue + 0.01 < red) & (edge_density < 0.55)).astype(np.float32)
    organic_mass = _box_blur(organic_dark, radius=max(4, round(small.width / 55)))
    water = np.clip((0.27 - brightness) / 0.27, 0, 1) * np.clip((blue + green - red) * 2.4, 0, 1)
    snow_or_cloud = np.maximum(0, brightness - 0.78) * grayness

    score = road_network * 0.66 + road_candidates * 0.34 + detail * built_palette * 0.28
    score -= vegetation_color * 0.72 + organic_mass * 0.95 + water * 1.45 + snow_or_cloud * 2.1 + blue_dominance * 1.8
    score = np.clip(score, 0, 1)
    return small, score, road_candidates, road_network, organic_mass, water


def build_urban_road_vein_mask(
    image: Image.Image,
    *,
    max_width: int = 220,
    threshold: float = 0.5,
) -> UrbanMaskResult:
    small, score, _road_candidates, road_network, organic_mass, water = _road_vein_scores(image, max_width)

    smoothed_score = _box_blur(score, radius=max(2, round(small.width / 95)))
    adaptive_threshold = max(min(0.18, threshold * 0.32), float(np.quantile(smoothed_score, 0.68)))
    mask = smoothed_score >= adaptive_threshold
    mask &= road_network > np.quantile(road_network, 0.5)
    mask &= organic_mass < 0.78
    mask &= water < 0.45
    mask = _open(_close(mask, iterations=2), iterations=1)
    return UrbanMaskResult(image=small, score=smoothed_score, mask=mask, threshold=adaptive_threshold)


def build_urban_road_line_mask(
    image: Image.Image,
    *,
    max_width: int = 320,
    threshold: float = 0.5,
) -> UrbanMaskResult:
    small, score, road_candidates, _road_network, organic_mass, _water = _road_vein_scores(image, max_width)
    line_score = np.clip(road_candidates * (0.65 + score * 0.45) * (1 - organic_mass * 0.72) * (1 - _water * 0.88), 0, 1)
    adaptive_threshold = max(min(0.34, threshold * 0.52), float(np.quantile(line_score, 0.70)))
    mask = line_score >= adaptive_threshold
    mask &= organic_mass < 0.56
    mask &= _water < 0.36
    return UrbanMaskResult(image=small, score=line_score, mask=mask, threshold=adaptive_threshold)


def build_urban_road_seeded_mask(
    image: Image.Image,
    *,
    max_width: int = 320,
    threshold: float = 0.5,
) -> UrbanMaskResult:
    palette = build_urban_palette_pattern_mask(image, max_width=max_width, threshold=threshold)
    road = build_urban_road_line_mask(image, max_width=max_width, threshold=threshold)

    vicinity_iterations = max(3, round(palette.image.width / 44))
    road_vicinity = _dilate(road.mask, iterations=vicinity_iterations)
    mask = (palette.mask & road_vicinity) | _dilate(road.mask, iterations=1)
    mask = _open(_close(mask, iterations=max(3, round(palette.image.width / 64))), iterations=1)

    score = np.clip(palette.score * 0.48 + road.score * 0.9, 0, 1)
    return UrbanMaskResult(image=palette.image, score=score, mask=mask, threshold=threshold)


def build_urban_road_density_mask(
    image: Image.Image,
    *,
    max_width: int = 320,
    threshold: float = 0.5,
) -> UrbanMaskResult:
    small, score, road_candidates, road_network, organic_mass, water = _road_vein_scores(image, max_width)
    line_score = np.clip(road_candidates * (0.65 + score * 0.45) * (1 - organic_mass * 0.72) * (1 - water * 0.88), 0, 1)
    line_threshold = max(min(0.34, threshold * 0.52), float(np.quantile(line_score, 0.70)))
    line_mask = (line_score >= line_threshold) & (organic_mass < 0.56) & (water < 0.36)

    # City regions have many nearby road-like strokes. Rivers, trails, and mountain
    # ridges can be line-like too, but they usually do not form a dense local mesh.
    density_radius = max(5, round(small.width / 34))
    broad_radius = max(density_radius + 2, round(small.width / 22))
    line_density = _box_blur(line_mask.astype(np.float32), radius=density_radius)
    broad_density = _box_blur(line_mask.astype(np.float32), radius=broad_radius)
    network_density = _box_blur(road_network, radius=density_radius)

    urban_density = np.clip(line_density * 1.25 + broad_density * 0.78 + network_density * 0.42 - organic_mass * 0.62 - water * 1.0, 0, 1)
    adaptive_threshold = max(min(0.08, threshold * 0.16), float(np.quantile(urban_density, 0.70)))
    mask = urban_density >= adaptive_threshold
    mask &= organic_mass < 0.70
    mask &= water < 0.42
    mask = _open(_close(mask, iterations=max(2, round(small.width / 80))), iterations=1)

    min_component_area = max(32, round(mask.size * 0.004))
    kept = np.zeros(mask.shape, dtype=bool)
    for component in _component_masks(mask, min_area=min_component_area):
        kept |= component
    return UrbanMaskResult(image=small, score=urban_density, mask=kept, threshold=adaptive_threshold)
